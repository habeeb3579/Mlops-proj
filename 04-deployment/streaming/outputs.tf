# outputs.tf
output "api_gateway_url" {
  description = "API Gateway endpoint URL"
  value       = "${aws_api_gateway_stage.main.invoke_url}/stream"
}

output "api_gateway_id" {
  description = "API Gateway REST API ID"
  value       = aws_api_gateway_rest_api.main.id
}

output "sqs_queue_url" {
  description = "Main SQS queue URL"
  value       = aws_sqs_queue.main_queue.url
}

output "sqs_fifo_queue_url" {
  description = "FIFO SQS queue URL"
  value       = aws_sqs_queue.fifo_queue.url
}

output "sqs_dlq_url" {
  description = "Dead letter queue URL"
  value       = aws_sqs_queue.dlq.url
}

output "producer_lambda_arn" {
  description = "Producer Lambda function ARN"
  value       = aws_lambda_function.producer.arn
}

output "consumer_lambda_arn" {
  description = "Consumer Lambda function ARN"
  value       = aws_lambda_function.consumer.arn
}

output "producer_lambda_name" {
  description = "Producer Lambda function name"
  value       = aws_lambda_function.producer.function_name
}

output "consumer_lambda_name" {
  description = "Consumer Lambda function name"
  value       = aws_lambda_function.consumer.function_name
}

output "region" {
  description = "AWS region"
  value       = data.aws_region.current.name
}

output "account_id" {
  description = "AWS account ID"
  value       = data.aws_caller_identity.current.account_id
}

# Cost monitoring outputs
output "free_tier_summary" {
  description = "Free tier usage summary"
  value = {
    api_gateway_requests = "1M requests/month free"
    lambda_requests      = "1M requests/month free"
    lambda_compute_time  = "400,000 GB-seconds/month free"
    sqs_requests        = "1M requests/month free"
    cloudwatch_logs     = "5GB storage free"
    notes = [
      "Monitor usage in AWS billing dashboard",
      "Set up billing alerts for cost control",
      "SQS charges after 1M requests per month",
      "Lambda charges after free tier limits"
    ]
  }
}