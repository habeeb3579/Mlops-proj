# lambda_functions/consumer.py
import json
import boto3
import os
from datetime import datetime

# Initialize AWS clients
sqs = boto3.client('sqs', region_name=os.environ.get('REGION', 'us-east-1'))
dynamodb = boto3.resource('dynamodb', region_name=os.environ.get('REGION', 'us-east-1'))

def lambda_handler(event, context):
    """
    Consumer Lambda function that processes messages from SQS
    Simulates Kinesis stream processing
    """
    
    processed_records = []
    failed_records = []
    
    try:
        # Handle SQS event source mapping
        if 'Records' in event:
            for record in event['Records']:
                try:
                    # Process each SQS message
                    message_body = json.loads(record['body'])
                    
                    # Process the message (your business logic here)
                    result = process_message(message_body)
                    
                    processed_records.append({
                        'messageId': record.get('messageId'),
                        'receiptHandle': record.get('receiptHandle'),
                        'result': result,
                        'status': 'processed'
                    })
                    
                    print(f"Successfully processed message: {record.get('messageId')}")
                    
                except Exception as e:
                    print(f"Error processing record {record.get('messageId')}: {str(e)}")
                    failed_records.append({
                        'messageId': record.get('messageId'),
                        'error': str(e),
                        'status': 'failed'
                    })
        
        # Handle direct invocation (e.g., from EventBridge)
        else:
            # Poll SQS queue for messages
            messages = poll_queue()
            for message in messages:
                try:
                    message_body = json.loads(message['Body'])
                    result = process_message(message_body)
                    
                    # Delete processed message
                    sqs.delete_message(
                        QueueUrl=os.environ['QUEUE_URL'],
                        ReceiptHandle=message['ReceiptHandle']
                    )
                    
                    processed_records.append({
                        'messageId': message.get('MessageId'),
                        'result': result,
                        'status': 'processed'
                    })
                    
                except Exception as e:
                    print(f"Error processing message: {str(e)}")
                    failed_records.append({
                        'messageId': message.get('MessageId'),
                        'error': str(e),
                        'status': 'failed'
                    })
        
        # Return processing summary
        return {
            'statusCode': 200,
            'body': json.dumps({
                'processed': len(processed_records),
                'failed': len(failed_records),
                'processedRecords': processed_records,
                'failedRecords': failed_records,
                'timestamp': datetime.utcnow().isoformat()
            })
        }
        
    except Exception as e:
        print(f"Critical error in consumer: {str(e)}")
        return {
            'statusCode': 500,
            'body': json.dumps({
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            })
        }

def process_message(message_data):
    """
    Process individual message - customize this for your use case
    """
    try:
        # Extract message information
        message_id = message_data.get('id')
        timestamp = message_data.get('timestamp')
        data = message_data.get('data', {})
        source = message_data.get('source', 'unknown')
        
        # Example processing logic
        processed_data = {
            'original_id': message_id,
            'processed_at': datetime.utcnow().isoformat(),
            'source': source,
            'data_keys': list(data.keys()) if isinstance(data, dict) else [],
            'data_size': len(str(data)),
            'processing_status': 'completed'
        }
        
        # Optional: Store in DynamoDB or other service
        # store_processed_data(processed_data)
        
        # Optional: Forward to another queue or service
        # forward_to_downstream(processed_data)
        
        print(f"Processed message {message_id} with {len(processed_data)} fields")
        
        return processed_data
        
    except Exception as e:
        print(f"Error in process_message: {str(e)}")
        raise

def poll_queue():
    """
    Poll SQS queue for messages (used for direct invocation)
    """
    try:
        response = sqs.receive_message(
            QueueUrl=os.environ['QUEUE_URL'],
            MaxNumberOfMessages=2,  # Process up to 10 messages
            WaitTimeSeconds=20,      # Long polling
            VisibilityTimeout=30
        )
        
        return response.get('Messages', [])
        
    except Exception as e:
        print(f"Error polling queue: {str(e)}")
        return []

def store_processed_data(data):
    """
    Optional: Store processed data in DynamoDB
    Uncomment and configure if you need persistent storage
    """
    # table_name = os.environ.get('DYNAMODB_TABLE', 'processed-messages')
    # table = dynamodb.Table(table_name)
    # 
    # table.put_item(Item=data)

def forward_to_downstream(data):
    """
    Optional: Forward processed data to another service
    """
    # Example: Send to another SQS queue, SNS topic, or API
    pass

def batch_process_messages(messages):
    """
    Process multiple messages efficiently
    """
    results = []
    
    for message in messages:
        try:
            result = process_message(message)
            results.append(result)
        except Exception as e:
            print(f"Error in batch processing: {str(e)}")
            results.append({'error': str(e)})
    
    return results