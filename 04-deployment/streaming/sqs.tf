# sqs.tf
# Main processing queue (replaces Kinesis stream)
resource "aws_sqs_queue" "main_queue" {
  name                      = "${var.project_name}-main-queue"
  delay_seconds             = 0
  max_message_size          = 262144 # 256KB max
  message_retention_seconds = var.sqs_message_retention
  visibility_timeout_seconds = var.sqs_visibility_timeout
  receive_wait_time_seconds = 20 # Long polling for cost efficiency
  
  # Enable server-side encryption (free)
  sqs_managed_sse_enabled = true
  
  tags = var.tags
}

# Dead letter queue for failed messages
resource "aws_sqs_queue" "dlq" {
  name                      = "${var.project_name}-dlq"
  message_retention_seconds = var.sqs_message_retention
  
  sqs_managed_sse_enabled = true
  
  tags = var.tags
}

# Redrive policy for main queue
resource "aws_sqs_queue_redrive_policy" "main_queue_redrive" {
  queue_url = aws_sqs_queue.main_queue.id
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.dlq.arn
    maxReceiveCount     = 3
  })
}

# Optional: FIFO queue for ordered processing (if needed)
resource "aws_sqs_queue" "fifo_queue" {
  name                        = "${var.project_name}-fifo.fifo"
  fifo_queue                  = true
  content_based_deduplication = true
  delay_seconds               = 0
  max_message_size            = 262144
  message_retention_seconds   = var.sqs_message_retention
  visibility_timeout_seconds  = var.sqs_visibility_timeout
  receive_wait_time_seconds   = 20
  
  sqs_managed_sse_enabled = true
  
  tags = var.tags
}

# Dead letter queue for FIFO
resource "aws_sqs_queue" "fifo_dlq" {
  name       = "${var.project_name}-fifo-dlq.fifo"
  fifo_queue = true
  
  sqs_managed_sse_enabled = true
  
  tags = var.tags
}

resource "aws_sqs_queue_redrive_policy" "fifo_queue_redrive" {
  queue_url = aws_sqs_queue.fifo_queue.id
  redrive_policy = jsonencode({
    deadLetterTargetArn = aws_sqs_queue.fifo_dlq.arn
    maxReceiveCount     = 3
  })
}