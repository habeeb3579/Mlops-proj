import json
import boto3
import os
import uuid
import pickle
import gc
from datetime import datetime
from typing import Union, Tuple, List, Dict, Optional
import numpy as np

# Initialize clients at module level for reuse
sqs = boto3.client('sqs', region_name=os.environ.get('REGION', 'us-east-1'))

# Global model cache
model = None
use_remote = os.getenv("USE_REMOTE_MODEL", "false").lower() == "true"

def load_model_once():
    """Load model once and cache globally with error handling"""
    global model
    if model is None:
        try:
            if use_remote:
                # Import mlflow only if needed to save memory
                import mlflow
                tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
                model_name = os.environ.get("MLFLOW_MODEL_NAME", "nyc-taxi-regressor-weighted-main9")
                mlflow.set_tracking_uri(tracking_uri)
                print(f"📡 Loading model from MLflow: {model_name}@production")
                model = mlflow.pyfunc.load_model(f"models:/{model_name}@production")
                print(f"✅ Loaded MLflow model")
            else:
                print("📦 Loading model from local pickle file...")
                model_path = "/var/task/model.pkl"  # Lambda container path
                if not os.path.exists(model_path):
                    model_path = "model.pkl"  # Fallback for local testing
                
                with open(model_path, "rb") as f:
                    model = pickle.load(f)
                print("✅ Loaded local pickle model")
            
            # Force garbage collection after model loading
            gc.collect()
            
        except Exception as e:
            print(f"❌ Model loading error: {str(e)}")
            raise
    
    return model

def parse_datetime_fast(dt_str: str) -> Optional[datetime]:
    """Optimized datetime parsing with minimal try/except overhead"""
    if not dt_str:
        return None
    
    try:
        # Handle ISO format first (most common)
        if "T" in dt_str:
            return datetime.fromisoformat(dt_str.replace("Z", "+00:00"))
        else:
            # Handle space-separated format
            return datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S")
    except Exception:
        print(f"⚠️ Could not parse datetime: {dt_str}")
        return None

def calculate_duration(pickup_str: str, dropoff_str: str) -> Optional[float]:
    """Calculate duration in minutes with validation"""
    pickup_dt = parse_datetime_fast(pickup_str)
    dropoff_dt = parse_datetime_fast(dropoff_str)
    
    if not pickup_dt or not dropoff_dt:
        return None
    
    try:
        duration = (dropoff_dt - pickup_dt).total_seconds() / 60.0
        # Filter outliers (1-60 minutes)
        return duration if 1 <= duration <= 60 else None
    except Exception:
        return None

def prepare_single_feature(row: Dict) -> Optional[Dict]:
    """Process a single row efficiently"""
    # Extract datetime fields
    pickup = row.get("lpep_pickup_datetime") or row.get("tpep_pickup_datetime")
    dropoff = row.get("lpep_dropoff_datetime") or row.get("tpep_dropoff_datetime")
    
    if not pickup or not dropoff:
        return None
    
    # Calculate duration
    duration = calculate_duration(pickup, dropoff)
    if duration is None:
        return None
    
    # Create feature dict with minimal copying
    feature = {
        "duration": duration,
        "ride_id": str(uuid.uuid4())
    }
    
    # Add location IDs as strings
    if "PULocationID" in row:
        feature["PULocationID"] = str(row["PULocationID"])
    if "DOLocationID" in row:
        feature["DOLocationID"] = str(row["DOLocationID"])
    
    # Create PU_DO combination if both locations exist
    if "PULocationID" in feature and "DOLocationID" in feature:
        feature["PU_DO"] = f"{feature['PULocationID']}_{feature['DOLocationID']}"
    
    # Copy other relevant fields without deep copying
    for key in ["passenger_count", "trip_distance", "fare_amount", "total_amount"]:
        if key in row:
            feature[key] = row[key]
    
    return feature

def prepare_features_batch(data: Union[Dict, List[Dict]]) -> Tuple[List[Dict], int]:
    """Optimized batch feature preparation"""
    if isinstance(data, dict):
        data = [data]
    
    features = []
    processed_count = 0
    
    for row in data:
        try:
            feature = prepare_single_feature(row)
            if feature:
                features.append(feature)
                processed_count += 1
        except Exception as e:
            print(f"⚠️ Error processing row: {str(e)}")
            continue
    
    return features, processed_count

def make_prediction_batch(features: List[Dict]) -> List[float]:
    """Make predictions for a batch of features"""
    try:
        model_instance = load_model_once()
        
        # Convert features to format expected by model
        # This depends on your model's expected input format
        predictions = model_instance.predict(features)
        
        # Ensure predictions is a list
        if hasattr(predictions, 'tolist'):
            return predictions.tolist()
        elif isinstance(predictions, (list, tuple)):
            return list(predictions)
        else:
            return [float(predictions)]
            
    except Exception as e:
        print(f"❌ Prediction error: {str(e)}")
        raise

def process_message_optimized(message_data: Dict) -> Dict:
    """Process a single message with optimized memory usage"""
    try:
        # Prepare features
        features, count = prepare_features_batch(message_data["data"])
        
        if not features:
            raise ValueError("No valid features extracted from data")
        
        # Make prediction
        predictions = make_prediction_batch(features)
        
        if not predictions:
            raise ValueError("No predictions generated")
        
        # Get model version info
        model_version = "local-pickle"
        if hasattr(model, 'metadata') and hasattr(model.metadata, 'run_id'):
            model_version = model.metadata.run_id
        
        result = {
            "ride_id": features[0]["ride_id"],
            "prediction": float(predictions[0]),
            "model_version": model_version,
            "processed_features": count,
            "received_at": datetime.utcnow().isoformat()
        }
        
        return result
        
    except Exception as e:
        print(f"❌ Processing error: {str(e)}")
        raise

def process_records_batch(records: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    """Process multiple records efficiently"""
    processed_records = []
    failed_records = []
    
    for record in records:
        message_id = record.get("messageId", "unknown")
        
        try:
            message_body = json.loads(record["body"])
            result = process_message_optimized(message_body)
            
            processed_records.append({
                "messageId": message_id,
                "result": result,
                "status": "processed"
            })
            
            print(f"✅ Processed {message_id}: prediction={result['prediction']:.4f}")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Failed {message_id}: {error_msg}")
            
            failed_records.append({
                "messageId": message_id,
                "error": error_msg,
                "status": "failed"
            })
    
    return processed_records, failed_records

def lambda_handler(event, context):
    """Optimized Lambda handler with better memory management"""
    try:
        # Log memory info if available
        if hasattr(context, 'memory_limit_in_mb'):
            print(f"🔧 Lambda memory limit: {context.memory_limit_in_mb}MB")
        
        records = event.get("Records", [])
        
        if not records:
            print("⚠️ No records to process")
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "No records to process",
                    "timestamp": datetime.utcnow().isoformat()
                })
            }
        
        print(f"📊 Processing {len(records)} records")
        
        # Process records in batch
        processed_records, failed_records = process_records_batch(records)
        
        # Build response
        response_body = {
            "processed": len(processed_records),
            "failed": len(failed_records),
            "total_records": len(records),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Include results only if there aren't too many (to avoid large responses)
        if len(processed_records) <= 10:
            response_body["results"] = processed_records
        
        if failed_records:
            response_body["failures"] = failed_records[:5]  # Limit failure details
        
        print(f"📈 Summary: {len(processed_records)} processed, {len(failed_records)} failed")
        
        return {
            "statusCode": 200,
            "body": json.dumps(response_body)
        }
        
    except Exception as e:
        error_msg = str(e)
        print(f"🔥 Lambda handler error: {error_msg}")
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": error_msg,
                "timestamp": datetime.utcnow().isoformat()
            })
        }
    
    finally:
        # Force garbage collection at the end
        gc.collect()