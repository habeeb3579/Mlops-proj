# iam.tf
# IAM role for Lambda producer function
resource "aws_iam_role" "lambda_producer_role" {
  name = "${var.project_name}-lambda-producer-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# IAM role for Lambda consumer function
resource "aws_iam_role" "lambda_consumer_role" {
  name = "${var.project_name}-lambda-consumer-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "lambda.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# Producer Lambda policy - can send to SQS
resource "aws_iam_policy" "lambda_producer_policy" {
  name = "${var.project_name}-lambda-producer-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
      },
      {
        Effect = "Allow"
        Action = [
          "sqs:SendMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = [
          aws_sqs_queue.main_queue.arn,
          aws_sqs_queue.fifo_queue.arn
        ]
      }
    ]
  })
}

# Consumer Lambda policy - can receive from SQS
resource "aws_iam_policy" "lambda_consumer_policy" {
  name = "${var.project_name}-lambda-consumer-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "arn:aws:logs:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:*"
      },
      {
        Effect = "Allow"
        Action = [
          "sqs:ReceiveMessage",
          "sqs:DeleteMessage",
          "sqs:GetQueueAttributes"
        ]
        Resource = [
          aws_sqs_queue.main_queue.arn,
          aws_sqs_queue.fifo_queue.arn,
          aws_sqs_queue.dlq.arn,
          aws_sqs_queue.fifo_dlq.arn
        ]
      }
    ]
  })
}

# Attach policies to roles
resource "aws_iam_role_policy_attachment" "lambda_producer_policy_attachment" {
  role       = aws_iam_role.lambda_producer_role.name
  policy_arn = aws_iam_policy.lambda_producer_policy.arn
}

resource "aws_iam_role_policy_attachment" "lambda_consumer_policy_attachment" {
  role       = aws_iam_role.lambda_consumer_role.name
  policy_arn = aws_iam_policy.lambda_consumer_policy.arn
}

# API Gateway role
resource "aws_iam_role" "api_gateway_role" {
  name = "${var.project_name}-api-gateway-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "apigateway.amazonaws.com"
        }
      }
    ]
  })

  tags = var.tags
}

# API Gateway policy to invoke Lambda
resource "aws_iam_policy" "api_gateway_policy" {
  name = "${var.project_name}-api-gateway-policy"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Effect = "Allow"
        Action = [
          "lambda:InvokeFunction"
        ]
        Resource = "arn:aws:lambda:${data.aws_region.current.name}:${data.aws_caller_identity.current.account_id}:function:${var.project_name}-*"
      },
      {
        Effect = "Allow"
        Action = [
          "logs:CreateLogGroup",
          "logs:CreateLogStream",
          "logs:PutLogEvents"
        ]
        Resource = "*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "api_gateway_policy_attachment" {
  role       = aws_iam_role.api_gateway_role.name
  policy_arn = aws_iam_policy.api_gateway_policy.arn
}