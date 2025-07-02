# lambda_functions/consumer.py

import json
import os
import uuid
import pickle
import logging
from datetime import datetime
from typing import Union, Tuple, List, Dict, Optional
from functools import lru_cache

# Core imports only - others imported when needed
import boto3
#from botocore.exceptions import BotoCoreError, ClientError
import pandas as pd

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Format logs for AWS CloudWatch
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '[%(levelname)s] %(asctime)s - %(name)s - %(message)s'
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Environment variables with defaults
REGION = os.getenv('AWS_REGION', os.getenv('REGION', 'us-east-1'))
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_MODEL_NAME = os.getenv("MLFLOW_MODEL_NAME", "nyc-taxi-regressor-weighted-main9")
USE_REMOTE_MODEL = os.getenv("USE_REMOTE_MODEL", "false").lower() == "true"
QUEUE_URL = os.getenv('QUEUE_URL')

# AWS clients with error handling
try:
    sqs = boto3.client('sqs', region_name=REGION)
    dynamodb = boto3.resource('dynamodb', region_name=REGION)
    logger.info(f"✅ Initialized AWS clients for region: {REGION}")
except Exception as e:
    logger.error(f"❌ Failed to initialize AWS clients: {str(e)}")
    raise

# Global model variable
model = None

def _import_mlflow():
    """Import MLflow only when needed."""
    try:
        import mlflow
        return mlflow
    except ImportError:
        logger.error("❌ MLflow not installed. Install with: pip install mlflow")
        raise ImportError("MLflow is required for remote model loading but not installed")


def _import_numpy():
    """Import numpy only when needed."""
    try:
        import numpy as np
        return np
    except ImportError:
        logger.error("❌ NumPy not installed. Install with: pip install numpy")
        raise ImportError("NumPy is required for data processing but not installed")

@lru_cache(maxsize=None)
def load_model_once(use_remote: bool = False) -> None:
    """Load model once and cache it globally with error handling."""
    global model
    
    if model is not None:
        logger.info("📦 Model already loaded, skipping reload")
        return
    
    try:
        if use_remote:
            # Import MLflow only when using remote model
            mlflow = _import_mlflow()
            logger.info(f"📡 Loading model from MLflow: {MLFLOW_TRACKING_URI}")
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            model = mlflow.pyfunc.load_model(f"models:/{MLFLOW_MODEL_NAME}@production")
            logger.info(f"✅ Successfully loaded MLflow model: {MLFLOW_MODEL_NAME}@production")
        else:
            logger.info("📦 Loading model from local pickle file...")
            # Lambda function paths for ECR deployment
            possible_paths = [
                "/var/task/model.pkl",      # Default Lambda function directory
                "/opt/model.pkl",           # Lambda layer directory
                "/tmp/model.pkl",           # Temporary directory
                "model.pkl",                # Current working directory
                "./model.pkl"               # Relative path
            ]
            
            model_path = None
            for path in possible_paths:
                if os.path.exists(path):
                    model_path = path
                    logger.info(f"🎯 Found model at: {path}")
                    break
            
            if not model_path:
                available_files = []
                for check_dir in ["/var/task", "/opt", "/tmp", "."]:
                    try:
                        files = os.listdir(check_dir)
                        available_files.extend([f"{check_dir}/{f}" for f in files if f.endswith('.pkl')])
                    except (OSError, PermissionError):
                        continue
                
                error_msg = f"Model file not found. Checked paths: {possible_paths}. Available .pkl files: {available_files}"
                logger.error(f"❌ {error_msg}")
                raise FileNotFoundError(error_msg)
            
            with open(model_path, "rb") as f:
                model = pickle.load(f)
            logger.info(f"✅ Successfully loaded local pickle model from: {model_path}")
            
    except FileNotFoundError:
        logger.error("❌ Model file not found")
        raise
    except Exception as e:
        logger.error(f"❌ Failed to load model: {str(e)}")
        raise

def generate_uuids(n: int) -> List[str]:
    """Generate n unique UUIDs efficiently."""
    return [str(uuid.uuid4()) for _ in range(n)]

def prepare_df_pandas(data: Union[List[Dict], Dict]) -> Tuple[Dict, Optional[float]]:
    """
    Full pandas-based data preparation for complex cases.
    
    Args:
        data: Input data in various formats
        
    Returns:
        Tuple of (processed_data_dict, duration)
    """
    try:
        # Import pandas only when needed
        #np = _import_numpy()
        
        # Convert input to DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            raise ValueError(f"Unsupported input type: {type(data)}")

        # Determine datetime columns
        datetime_mapping = {
            ('lpep_pickup_datetime', 'lpep_dropoff_datetime'): ('lpep_pickup_datetime', 'lpep_dropoff_datetime'),
            ('tpep_pickup_datetime', 'tpep_dropoff_datetime'): ('tpep_pickup_datetime', 'tpep_dropoff_datetime')
        }
        
        pickup_col, dropoff_col = None, None
        for (pickup, dropoff), (p_col, d_col) in datetime_mapping.items():
            if pickup in df.columns and dropoff in df.columns:
                pickup_col, dropoff_col = p_col, d_col
                break

        # Process datetime and calculate duration
        duration = None
        if pickup_col and dropoff_col:
            df[pickup_col] = pd.to_datetime(df[pickup_col], errors='coerce')
            df[dropoff_col] = pd.to_datetime(df[dropoff_col], errors='coerce')
            
            # Calculate duration in minutes
            df["duration"] = (df[dropoff_col] - df[pickup_col]).dt.total_seconds() / 60
            
            # Filter valid durations (1-60 minutes)
            initial_count = len(df)
            df = df[(df["duration"] >= 1) & (df["duration"] <= 60)]
            filtered_count = len(df)
            
            if filtered_count < initial_count:
                logger.warning(f"⚠️ Filtered out {initial_count - filtered_count} records with invalid duration")
            
            if not df.empty:
                duration = df["duration"].iloc[0]

        # Process location IDs
        location_cols = ["PULocationID", "DOLocationID"]
        for col in location_cols:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Create PU_DO combination if both location IDs exist
        if all(col in df.columns for col in location_cols):
            df["PU_DO"] = df["PULocationID"] + "_" + df["DOLocationID"]

        # Generate ride IDs
        df["ride_id"] = generate_uuids(len(df))

        if df.empty:
            logger.warning("⚠️ No valid records after pandas processing")
            return None, None

        # Return first record as dict
        result_dict = df.iloc[0].to_dict()
        
        # Convert numpy types to native Python types
        for key, value in result_dict.items():
            if hasattr(value, 'item'):  # numpy scalar
                result_dict[key] = value.item()
            elif pd.isna(value):
                result_dict[key] = None

        logger.info(f"📊 Prepared data with pandas processing")
        return result_dict, duration

    except Exception as e:
        logger.error(f"❌ Error in pandas data preparation: {str(e)}")
        return None, None

def prepare_data(data: Union[List[Dict], Dict]) -> Tuple[Dict, Optional[float]]:
    """
    Prepare data using the most appropriate method.
    
    Args:
        data: Input data
        
    Returns:
        Tuple of (processed_data_dict, duration)
    """
    
    # Fall back to pandas for complex cases
    logger.info("📊 Using pandas for data preparation")
    return prepare_df_pandas(data)

def process_message(message_data: Dict) -> Dict:
    """
    Process a single message and return prediction result.
    
    Args:
        message_data: Dictionary containing prediction data
        
    Returns:
        Dictionary with prediction results
    """
    try:
        # Ensure model is loaded
        if model is None:
            load_model_once(use_remote=USE_REMOTE_MODEL)

        # Prepare data
        processed_data, duration = prepare_data(message_data["data"])
        
        if processed_data is None:
            raise ValueError("No valid data after preprocessing")

        # For prediction, we might need to convert back to the format the model expects
        
        # MLflow models typically expect pandas DataFrame
        #pd = _import_pandas()
        prediction_input = pd.DataFrame([processed_data])
        
        # Make prediction
        prediction = model.predict(prediction_input)
        
        # Handle prediction result
        if hasattr(prediction, '__iter__') and not isinstance(prediction, str):
            pred_value = float(prediction[0])
        else:
            pred_value = float(prediction)
        
        # Get model version info
        model_version = "unknown"
        if hasattr(model, "metadata") and hasattr(model.metadata, "run_id"):
            model_version = model.metadata.run_id
        elif not USE_REMOTE_MODEL:
            model_version = "local-pickle"

        result = {
            "ride_id": processed_data["ride_id"],
            "prediction": pred_value,
            "model_version": model_version,
            "received_at": datetime.utcnow().isoformat(),
            "use_remote_model": USE_REMOTE_MODEL
        }

        logger.info(f"✅ Prediction completed - ride_id: {result['ride_id']}, prediction: {result['prediction']:.2f}")
        return result

    except Exception as e:
        logger.error(f"❌ Error processing message: {str(e)}")
        raise

def lambda_handler(event, context):
    """
    Main Lambda handler with optimized error handling and logging.
    
    Args:
        event: Lambda event containing SQS records
        context: Lambda context object
        
    Returns:
        Response dictionary with processing results
    """
    start_time = datetime.utcnow()
    processed_records = []
    failed_records = []
    
    try:
        # Load model once at the start
        load_model_once(use_remote=USE_REMOTE_MODEL)
        
        records = event.get("Records", [])
        logger.info(f"🚀 Processing {len(records)} records")
        
        if not records:
            logger.warning("⚠️ No records found in event")
            return {
                "statusCode": 200,
                "body": json.dumps({
                    "message": "No records to process",
                    "timestamp": start_time.isoformat()
                })
            }

        # Process each record
        for i, record in enumerate(records):
            message_id = record.get("messageId", f"unknown-{i}")
            
            try:
                logger.info(f"📝 Processing record {i+1}/{len(records)} - ID: {message_id}")
                
                # Parse message body
                message_body = json.loads(record["body"])
                
                # Validate message structure
                if "data" not in message_body:
                    raise ValueError("Missing 'data' field in message body")
                
                # Process the message
                result = process_message(message_body)
                
                processed_records.append({
                    "messageId": message_id,
                    "result": result,
                    "status": "processed"
                })
                
                logger.info(f"✅ Successfully processed {message_id}")

            except json.JSONDecodeError as e:
                error_msg = f"Invalid JSON in message body: {str(e)}"
                logger.error(f"❌ {error_msg} - ID: {message_id}")
                failed_records.append({
                    "messageId": message_id,
                    "error": error_msg,
                    "status": "failed",
                    "error_type": "json_decode_error"
                })
                
            except Exception as e:
                error_msg = str(e)
                logger.error(f"❌ Processing failed for {message_id}: {error_msg}")
                failed_records.append({
                    "messageId": message_id,
                    "error": error_msg,
                    "status": "failed",
                    "error_type": type(e).__name__
                })

        # Calculate processing time
        processing_time = (datetime.utcnow() - start_time).total_seconds()
        
        # Log summary
        success_count = len(processed_records)
        failure_count = len(failed_records)
        logger.info(f"📊 Processing complete - Success: {success_count}, Failed: {failure_count}, Time: {processing_time:.2f}s")

        response_body = {
            "processed": success_count,
            "failed": failure_count,
            "total_records": len(records),
            "processing_time_seconds": round(processing_time, 2),
            "results": processed_records,
            "failures": failed_records,
            "timestamp": datetime.utcnow().isoformat(),
            "model_config": {
                "use_remote": USE_REMOTE_MODEL,
                "model_name": MLFLOW_MODEL_NAME if USE_REMOTE_MODEL else "local-pickle"
            }
        }

        return {
            "statusCode": 200,
            "body": json.dumps(response_body)
        }

    except Exception as e:
        error_msg = f"Critical Lambda error: {str(e)}"
        logger.error(f"🔥 {error_msg}")
        
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": error_msg,
                "error_type": type(e).__name__,
                "timestamp": datetime.utcnow().isoformat(),
                "processing_time_seconds": (datetime.utcnow() - start_time).total_seconds()
            })
        }

def poll_queue() -> List[Dict]:
    """
    Poll SQS queue for messages with error handling.
    
    Returns:
        List of messages from the queue
    """
    if not QUEUE_URL:
        logger.error("❌ QUEUE_URL environment variable not set")
        return []
    
    try:
        logger.info(f"📥 Polling queue: {QUEUE_URL}")
        response = sqs.receive_message(
            QueueUrl=QUEUE_URL,
            MaxNumberOfMessages=10,
            WaitTimeSeconds=20,
            VisibilityTimeout=30,
            AttributeNames=['All'],
            MessageAttributeNames=['All']
        )
        
        messages = response.get('Messages', [])
        logger.info(f"📨 Received {len(messages)} messages from queue")
        return messages
    except Exception as e:
        logger.error(f"❌ Unexpected error while polling queue: {str(e)}")
        return []

def store_processed_data(data: Dict) -> bool:
    """
    Store processed data in DynamoDB with error handling.
    
    Args:
        data: Processed data to store
        
    Returns:
        Boolean indicating success
    """
    try:
        table_name = os.getenv('DYNAMODB_TABLE_NAME')
        if not table_name:
            logger.warning("⚠️ DYNAMODB_TABLE_NAME not configured, skipping storage")
            return False
            
        table = dynamodb.Table(table_name)
        table.put_item(Item=data)
        logger.info(f"✅ Stored data for ride_id: {data.get('ride_id', 'unknown')}")
        return True
    except Exception as e:
        logger.error(f"❌ Unexpected error storing data: {str(e)}")
        return False

def forward_to_downstream(data: Dict) -> bool:
    """
    Forward processed data to downstream systems.
    
    Args:
        data: Data to forward
        
    Returns:
        Boolean indicating success
    """
    try:
        downstream_queue = os.getenv('DOWNSTREAM_QUEUE_URL')
        if not downstream_queue:
            logger.info("📤 No downstream queue configured, skipping forward")
            return True
            
        response = sqs.send_message(
            QueueUrl=downstream_queue,
            MessageBody=json.dumps(data),
            MessageAttributes={
                'source': {
                    'StringValue': 'lambda-consumer',
                    'DataType': 'String'
                },
                'timestamp': {
                    'StringValue': datetime.utcnow().isoformat(),
                    'DataType': 'String'
                }
            }
        )
        
        logger.info(f"✅ Forwarded data downstream - MessageId: {response['MessageId']}")
        return True
    except Exception as e:
        logger.error(f"❌ Unexpected error forwarding data: {str(e)}")
        return False