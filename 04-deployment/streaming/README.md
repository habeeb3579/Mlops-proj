# Free Tier Kinesis Alternative - Deployment Guide

we will use the model deploed locally in 02, se we need to, go to 02-experimental_tracking and run

mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db

if mlflow.db does not exist, you need to run the make_run.ipynb in section 02 (the first cell after ## This works well for local mode as it gives the full path of where the models are saved)

# use lambda_container if you want dockerized method (and comment lambda_new.tf), and vice versa. You can also use lambda.old in place of lambda_new if the dependencies size is below 70MB

initial consumer dockerfile

```bash
resource "local_file" "consumer_dockerfile" {
  content = <<EOF
FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies needed for building Python packages
# Using microdnf for Amazon Linux 2023
RUN microdnf update -y && \
    microdnf install -y gcc gcc-c++ make && \
    microdnf clean all && \
    rm -rf /var/cache/microdnf

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Copy requirements and install dependencies
COPY requirements.txt $${LAMBDA_TASK_ROOT}
RUN pip install -r requirements.txt --no-cache-dir

RUN pip cache purge && \
    rm -rf /tmp/* /var/tmp/* /root/.cache

# Copy function code
COPY consumer.py $${LAMBDA_TASK_ROOT}/lambda_function.py
# Copy model file if it exists
COPY model.pkl $${LAMBDA_TASK_ROOT}/model.pkl

# Memory optimization environment variables
ENV PYTHONUNBUFFERED=1
ENV MALLOC_ARENA_MAX=2
ENV PYTHONDONTWRITEBYTECODE=1

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"]
EOF
  filename = "${path.module}/consumer.Dockerfile"
}
```

## Overview

This Terraform configuration creates a cost-effective streaming data processing system using AWS free tier services:

- **SQS** (instead of Kinesis) - Message queuing and streaming
- **Lambda** - Data processing functions
- **API Gateway** - HTTP endpoint for data ingestion
- **CloudWatch** - Logging and monitoring
- **EventBridge** - Scheduled processing

## Architecture

```
API Gateway → Producer Lambda → SQS Queue → Consumer Lambda
                    ↓              ↓            ↓
               CloudWatch     Dead Letter   Processing
                 Logs          Queue        Results
```

## Prerequisites

1. AWS CLI configured with appropriate permissions
2. Terraform >= 1.0 installed
3. AWS account with free tier available
4. create IAM user and generate access key
5. run aws configure, supply your access key details, set region, set output to json

allow the following policies for the created user

- ✅ AWSLambda_FullAccess
- ✅ IAMFullAccess
- ✅ AmazonSQSFullAccess
- ✅ AmazonAPIGatewayAdministrator
- ✅ CloudWatchLogsFullAccess
- ✅ AmazonEventBridgeFullAccess
- ✅ AmazonEC2ContainerRegistryFullAccess
- ✅ AmazonElasticContainerRegistryPublicFullAccess (for public ecr)

## Deployment Steps

### 1. Clone and Prepare

```bash
# Create project directory
mkdir stream-processor && cd stream-processor

# Create directory structure
mkdir -p lambda_functions
```

### 2. Create Lambda Function Files

Create the lambda function files in the `lambda_functions/` directory:

- `lambda_functions/producer.py` (from Producer Lambda artifact)
- `lambda_functions/consumer.py` (from Consumer Lambda artifact)

### 3. Initialize and Deploy

```bash
# Initialize Terraform
terraform init

# Review planned changes
terraform plan

# Deploy infrastructure
terraform apply

or
terraform apply --var use_private_ecr=true
terraform apply -var="use_private_ecr=true" -auto-approve

```

### 4. Get Endpoint URL

```bash
# Get the API Gateway URL
terraform output api_gateway_url
```

copy the api_gateway_url and use it in test.py

## Usage Examples

### Use test.py

```bash
python test.py
```

### 1. Send Data via HTTP POST

```bash
# Basic data streaming
curl -X POST "https://87xb9tqwac.execute-api.us-east-1.amazonaws.com/dev/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "userId": "12345",
      "event": "page_view",
      "timestamp": "2024-01-01T12:00:00Z",
      "properties": {
        "page": "/home",
        "source": "organic"
      }
    }
  }'
```

### 2. Send Ordered Data (FIFO)

```bash
# For ordered processing
curl -X POST "https://your-api-id.execute-api.us-east-1.amazonaws.com/dev/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "data": {
      "orderId": "order-123",
      "status": "created",
      "amount": 99.99
    },
    "useFifo": true,
    "messageGroupId": "order-processing"
  }'
```

### 3. Batch Processing

```python
import requests
import json

# Python example for batch sending
api_url = "https://your-api-id.execute-api.us-east-1.amazonaws.com/dev/stream"

# Send multiple events
events = [
    {"userId": "user1", "action": "login"},
    {"userId": "user2", "action": "purchase", "amount": 50.00},
    {"userId": "user3", "action": "logout"}
]

for event in events:
    response = requests.post(api_url, json={"data": event})
    print(f"Status: {response.status_code}, Response: {response.json()}")
```

## Monitoring and Debugging

### 1. View Logs

```bash
# Producer Lambda logs
aws logs tail /aws/lambda/stream-processor-producer --follow

# Consumer Lambda logs
aws logs tail /aws/lambda/stream-processor-consumer --follow
```

### 2. Check Queue Status

```bash
# Get queue attributes
aws sqs get-queue-attributes \
  --queue-url $(terraform output -raw sqs_queue_url) \
  --attribute-names All
```

### 3. Monitor API Gateway

```bash
# API Gateway logs
aws logs tail /aws/apigateway/stream-processor --follow
```

## Cost Optimization Tips

1. **Stay Within Free Tier**

   - Monitor usage in AWS Billing dashboard
   - Set up billing alerts at $1, $5, $10
   - Use minimal Lambda memory (128MB)

2. **SQS Optimization**

   - Use long polling (20 seconds)
   - Batch message processing
   - Set appropriate visibility timeouts

3. **Lambda Optimization**

   - Keep functions lightweight
   - Use connection pooling
   - Optimize cold start times

4. **API Gateway Optimization**
   - Implement caching for repeated requests
   - Use request validation
   - Set up throttling

## Scaling Beyond Free Tier

When you exceed free tier limits:

1. **Add DynamoDB** for persistent storage
2. **Implement batching** for cost efficiency
3. **Add SNS** for fan-out patterns
4. **Use Step Functions** for complex workflows
5. **Consider Kinesis** for high-throughput scenarios

## Security Considerations

1. **API Authentication**

   ```hcl
   # Add to api_gateway.tf
   resource "aws_api_gateway_authorizer" "auth" {
     name                   = "api-authorizer"
     rest_api_id           = aws_api_gateway_rest_api.main.id
     authorizer_uri        = aws_lambda_function.authorizer.invoke_arn
     authorizer_credentials = aws_iam_role.api_gateway_role.arn
   }
   ```

2. **VPC Configuration** (if needed)
3. **Encryption at Rest** (already enabled for SQS)
4. **IAM Least Privilege** (already implemented)

## Troubleshooting

### Common Issues

1. **Lambda Timeout**

   - Increase timeout in variables.tf
   - Optimize function code

2. **SQS Messages Not Processing**

   - Check Lambda permissions
   - Verify event source mapping
   - Check dead letter queue

3. **API Gateway 502 Errors**

   - Check Lambda function logs
   - Verify integration configuration

4. **High Costs**
   - Review CloudWatch metrics
   - Optimize batch sizes
   - Implement request caching

## Cleanup

```bash
# Destroy all resources
terraform destroy
```

## Next Steps

1. Customize the consumer function for your use case
2. Add monitoring dashboards
3. Implement error handling and retries
4. Set up CI/CD pipeline
5. Add integration tests

This setup provides a robust, cost-effective alternative to Kinesis that can handle significant workloads within AWS free tier limits.
