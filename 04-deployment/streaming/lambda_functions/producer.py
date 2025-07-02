# lambda_functions/producer.py
import json
import boto3
import os
import uuid
from datetime import datetime

# Initialize SQS client
sqs = boto3.client('sqs', region_name=os.environ.get('REGION', 'us-east-1'))

def lambda_handler(event, context):
    """
    Producer Lambda function that receives data via API Gateway
    and sends it to SQS queue (Kinesis alternative)
    """
    
    try:
        # Parse the incoming request
        if 'body' in event:
            if isinstance(event['body'], str):
                body = json.loads(event['body'])
            else:
                body = event['body']
        else:
            body = event
        
        # Extract data and configuration
        data = body.get('data', {})
        use_fifo = body.get('useFifo', False)
        message_group_id = body.get('messageGroupId', 'default-group')
        
        # Add metadata to the message
        message_payload = {
            'id': str(uuid.uuid4()),
            'timestamp': datetime.utcnow().isoformat(),
            'data': data,
            'source': 'api-gateway'
        }
        
        # Choose queue based on requirements
        queue_url = os.environ['FIFO_QUEUE_URL'] if use_fifo else os.environ['QUEUE_URL']
        
        # Prepare SQS message
        sqs_message = {
            'QueueUrl': queue_url,
            'MessageBody': json.dumps(message_payload)
        }
        
        # Add FIFO-specific attributes if using FIFO queue
        if use_fifo:
            sqs_message['MessageGroupId'] = message_group_id
            sqs_message['MessageDeduplicationId'] = message_payload['id']
        
        print(message_payload)
        # Send message to SQS
        response = sqs.send_message(**sqs_message)
        
        # Success response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': True,
                'messageId': response.get('MessageId'),
                'queueType': 'FIFO' if use_fifo else 'Standard',
                'timestamp': message_payload['timestamp']
            })
        }
        
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {str(e)}")
        return {
            'statusCode': 400,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': 'Invalid JSON in request body'
            })
        }
        
    except Exception as e:
        print(f"Error processing message: {str(e)}")
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'success': False,
                'error': 'Internal server error'
            })
        }

def batch_send_messages(messages, queue_url, use_fifo=False):
    """
    Send multiple messages in batch for better performance
    """
    entries = []
    
    for i, message in enumerate(messages):
        entry = {
            'Id': str(i),
            'MessageBody': json.dumps(message)
        }
        
        if use_fifo:
            entry['MessageGroupId'] = message.get('messageGroupId', 'batch-group')
            entry['MessageDeduplicationId'] = message.get('id', str(uuid.uuid4()))
        
        entries.append(entry)
    
    # Send batch (max 10 messages per batch)
    for i in range(0, len(entries), 10):
        batch = entries[i:i+10]
        sqs.send_message_batch(
            QueueUrl=queue_url,
            Entries=batch
        )