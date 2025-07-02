# terraform.tfvars
# Customize these values for your environment

aws_region   = "us-east-1"  # Free tier friendly region
project_name = "stream-processor"
environment  = "dev"

# Tags for resource management
tags = {
  Project     = "stream-processor"
  Environment = "dev"
  ManagedBy   = "terraform"
  Owner       = "Habeeb"
  CostCenter  = "development"
}

# SQS Configuration
sqs_visibility_timeout = 300     # 5 minutes
sqs_message_retention  = 1209600 # 14 days (free tier limit)

# Lambda Configuration (optimized for free tier)
lambda_timeout = 300  # 1 minute
lambda_memory  = 1024 # Minimum memory for cost efficiency

# API Gateway throttling (cost control)
api_throttle_rate_limit  = 100
api_throttle_burst_limit = 200
use_private_ecr = false  # Set to true to use private ECR repositories