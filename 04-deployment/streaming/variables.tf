# variables.tf
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1" # Free tier friendly region
}

variable "project_name" {
  description = "Name of the project"
  type        = string
  default     = "stream-processor"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "dev"
}

variable "tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "stream-processor"
    Environment = "dev"
    ManagedBy   = "terraform"
  }
}

# SQS Configuration
variable "sqs_visibility_timeout" {
  description = "SQS message visibility timeout in seconds"
  type        = number
  default     = 300 # 5 minutes
}

variable "sqs_message_retention" {
  description = "SQS message retention period in seconds"
  type        = number
  default     = 1209600 # 14 days (free tier limit)
}

# Lambda Configuration
variable "lambda_timeout" {
  description = "Lambda function timeout in seconds"
  type        = number
  default     = 60 # Keep low for cost efficiency
}

variable "lambda_memory" {
  description = "Lambda function memory in MB"
  type        = number
  default     = 128 # Minimum for cost efficiency
}

# API Gateway Configuration
variable "api_throttle_rate_limit" {
  description = "API Gateway throttle rate limit"
  type        = number
  default     = 100
}

variable "api_throttle_burst_limit" {
  description = "API Gateway throttle burst limit"
  type        = number
  default     = 200
}

variable "use_private_ecr" {
  description = "Whether to use private ECR repositories (true) or public ECR (false)"
  type        = bool
  default     = false
}