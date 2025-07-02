# main.tf
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# Data source for current AWS account
data "aws_caller_identity" "current" {}
data "aws_region" "current" {}

# CloudWatch Log Groups for Lambda functions
resource "aws_cloudwatch_log_group" "producer_logs" {
  name              = "/aws/lambda/${var.project_name}-producer"
  retention_in_days = 14 # Free tier: 5GB storage included
  
  tags = var.tags
}

resource "aws_cloudwatch_log_group" "consumer_logs" {
  name              = "/aws/lambda/${var.project_name}-consumer"
  retention_in_days = 14
  
  tags = var.tags
}

resource "aws_cloudwatch_log_group" "api_gateway_logs" {
  name              = "/aws/apigateway/${var.project_name}"
  retention_in_days = 14
  
  tags = var.tags
}