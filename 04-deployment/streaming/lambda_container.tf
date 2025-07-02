# lambda_container.tf - Using Container Images for ML workloads with dynamic ECR support

# Create Dockerfile for consumer with build dependencies
resource "local_file" "consumer_dockerfile" {
  content = <<EOF
FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies needed for building Python packages with pandas
# Using microdnf for Amazon Linux 2023
RUN microdnf update -y && \
    microdnf install -y \
        gcc \
        gcc-c++ \
        make \
        gcc-gfortran \
        openblas-devel \
        lapack-devel \
        python3-devel \
        cmake \
        pkgconfig && \
    microdnf clean all && \
    rm -rf /var/cache/microdnf

# Upgrade pip and install build tools
RUN pip install --upgrade pip setuptools wheel

# Set environment variables for numpy/pandas compilation
ENV NPY_NUM_BUILD_JOBS=4
ENV OPENBLAS_NUM_THREADS=1

# Copy requirements and install dependencies
COPY requirements.txt $${LAMBDA_TASK_ROOT}

# Install pandas and numpy with pre-built wheels if possible, fallback to source
RUN pip install --no-cache-dir \
    --only-binary=:all: \
    pandas numpy || \
    pip install --no-cache-dir pandas numpy

# Install remaining requirements
RUN pip install -r requirements.txt --no-cache-dir

# Clean up build dependencies and cache
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

# Create Dockerfile for producer (lighter dependencies, but still with build tools for consistency)
resource "local_file" "producer_dockerfile" {
  content = <<EOF
FROM public.ecr.aws/lambda/python:3.12

# Install system dependencies (lighter for producer)
# Using microdnf for Amazon Linux 2023
RUN microdnf update -y && \
    microdnf install -y gcc && \
    microdnf clean all

# Upgrade pip to latest version
RUN pip install --upgrade pip

# Install lighter dependencies for producer
RUN pip install requests boto3 pandas numpy --no-cache-dir

# Copy function code
COPY producer.py $${LAMBDA_TASK_ROOT}/lambda_function.py

# Set the CMD to your handler
CMD ["lambda_function.lambda_handler"]
EOF
  filename = "${path.module}/producer.Dockerfile"
}

# Create requirements.txt
resource "local_file" "requirements" {
  content = <<EOF
requests==2.31.0
xgboost==2.1.4
#mlflow==2.22.0
#cloudpickle==3.1.1
numpy==1.26.4
#psutil==5.9.8
scikit-learn==1.5.1
boto3
#scipy==1.13.1
#pandas==2.0.3
EOF
  filename = "${path.module}/requirements.txt"
}

# Private ECR Repository for consumer (only created when use_private_ecr is true)
resource "aws_ecr_repository" "consumer_repo" {
  count = var.use_private_ecr ? 1 : 0
  name  = "${var.project_name}-consumer"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  tags = var.tags
}

# Private ECR Repository for producer (only created when use_private_ecr is true)
resource "aws_ecr_repository" "producer_repo" {
  count = var.use_private_ecr ? 1 : 0
  name  = "${var.project_name}-producer"
  
  image_scanning_configuration {
    scan_on_push = true
  }
  
  tags = var.tags
}

# Public ECR Repository for consumer (only created when use_private_ecr is false)
resource "aws_ecrpublic_repository" "consumer_repo_public" {
  count           = var.use_private_ecr ? 0 : 1
  repository_name = "${var.project_name}-consumer"

  catalog_data {
    about_text        = "Consumer Lambda function for ${var.project_name}"
    architectures     = ["x86_64"]
    description       = "ML workload consumer for stream processing"
    operating_systems = ["Linux"]
    usage_text        = "Container image for AWS Lambda consumer function"
  }

  tags = var.tags
}

# Public ECR Repository for producer (only created when use_private_ecr is false)
resource "aws_ecrpublic_repository" "producer_repo_public" {
  count           = var.use_private_ecr ? 0 : 1
  repository_name = "${var.project_name}-producer"

  catalog_data {
    about_text        = "Producer Lambda function for ${var.project_name}"
    architectures     = ["x86_64"]
    description       = "Stream processing producer function"
    operating_systems = ["Linux"]
    usage_text        = "Container image for AWS Lambda producer function"
  }

  tags = var.tags
}

# Local values for repository URLs and login commands
locals {
  consumer_repo_url = var.use_private_ecr ? aws_ecr_repository.consumer_repo[0].repository_url : aws_ecrpublic_repository.consumer_repo_public[0].repository_uri
  producer_repo_url = var.use_private_ecr ? aws_ecr_repository.producer_repo[0].repository_url : aws_ecrpublic_repository.producer_repo_public[0].repository_uri
  
  # ECR login commands differ between private and public
  private_consumer_login = "aws ecr get-login-password --region ${data.aws_region.current.name} | docker login --username AWS --password-stdin ${split("/", local.consumer_repo_url)[0]}"
  public_consumer_login  = "aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws"
  consumer_login_cmd     = var.use_private_ecr ? local.private_consumer_login : local.public_consumer_login
  
  private_producer_login = "aws ecr get-login-password --region ${data.aws_region.current.name} | docker login --username AWS --password-stdin ${split("/", local.producer_repo_url)[0]}"
  public_producer_login  = "aws ecr-public get-login-password --region us-east-1 | docker login --username AWS --password-stdin public.ecr.aws"
  producer_login_cmd     = var.use_private_ecr ? local.private_producer_login : local.public_producer_login
}

# Build and push consumer image
resource "null_resource" "consumer_image" {
  depends_on = [
    local_file.consumer_dockerfile,
    local_file.requirements,
    aws_ecr_repository.consumer_repo,
    aws_ecrpublic_repository.consumer_repo_public,
  ]
  
  triggers = {
    dockerfile = local_file.consumer_dockerfile.content_md5
    requirements = local_file.requirements.content_md5
    consumer_code = filemd5("${path.module}/lambda_functions/consumer.py")
    model_hash = filemd5("${path.module}/lambda_functions/model.pkl")
    use_private_ecr = var.use_private_ecr
    repo_url = local.consumer_repo_url
  }

  provisioner "local-exec" {
    command = <<EOF
set -e  # Exit on any error

echo "🔐 Logging into ECR..."
${local.consumer_login_cmd}

echo "📋 Copying files to build context..."
cp ${path.module}/lambda_functions/consumer.py ${path.module}/consumer.py
cp ${path.module}/lambda_functions/model.pkl ${path.module}/model.pkl

echo "📦 Building consumer image..."
docker build -f ${path.module}/consumer.Dockerfile -t ${local.consumer_repo_url}:latest ${path.module}

echo "🚀 Pushing consumer image..."
docker push ${local.consumer_repo_url}:latest

echo "🧹 Cleaning up build files..."
rm -f ${path.module}/consumer.py ${path.module}/model.pkl

echo "✅ Consumer image build and push completed successfully"
EOF
  }

  provisioner "local-exec" {
    when    = destroy
    command = <<EOF
# Cleanup on destroy
rm -f ${path.module}/consumer.py ${path.module}/model.pkl
EOF
  }
}

# Build and push producer image
resource "null_resource" "producer_image" {
  depends_on = [
    local_file.producer_dockerfile,
    aws_ecr_repository.producer_repo,
    aws_ecrpublic_repository.producer_repo_public,
  ]
  
  triggers = {
    dockerfile = local_file.producer_dockerfile.content_md5
    producer_code = filemd5("${path.module}/lambda_functions/producer.py")
    use_private_ecr = var.use_private_ecr
    repo_url = local.producer_repo_url
  }

  provisioner "local-exec" {
    command = <<EOF
set -e  # Exit on any error

echo "🔐 Logging into ECR..."
${local.producer_login_cmd}

echo "📋 Copying files to build context..."
cp ${path.module}/lambda_functions/producer.py ${path.module}/producer.py

echo "📦 Building producer image..."
docker build -f ${path.module}/producer.Dockerfile -t ${local.producer_repo_url}:latest ${path.module}

echo "🚀 Pushing producer image..."
docker push ${local.producer_repo_url}:latest

echo "🧹 Cleaning up build files..."
rm -f ${path.module}/producer.py

echo "✅ Producer image build and push completed successfully"
EOF
  }

  provisioner "local-exec" {
    when    = destroy
    command = <<EOF
# Cleanup on destroy
rm -f ${path.module}/producer.py
EOF
  }
}

# Consumer Lambda function using container image
resource "aws_lambda_function" "consumer" {
  function_name = "${var.project_name}-consumer"
  role         = aws_iam_role.lambda_consumer_role.arn
  package_type = "Image"
  image_uri    = "${local.consumer_repo_url}:latest"
  timeout      = var.lambda_timeout
  memory_size  = var.lambda_memory

  environment {
    variables = {
      QUEUE_URL = aws_sqs_queue.main_queue.url
      REGION    = data.aws_region.current.name
      USE_REMOTE_MODEL = "false"
      PYTHONUNBUFFERED = "1"
      MALLOC_ARENA_MAX = "2"
      PYTHONDONTWRITEBYTECODE = "1"
    }
  }

   # Add reserved concurrency to limit memory usage
  #reserved_concurrent_executions = 10

  depends_on = [
    null_resource.consumer_image,
    aws_iam_role_policy_attachment.lambda_consumer_policy_attachment,
    aws_cloudwatch_log_group.consumer_logs,
  ]

  tags = var.tags

  # Ensure the image exists before creating the function
  lifecycle {
    replace_triggered_by = [
      null_resource.consumer_image
    ]
  }
}

# Producer Lambda function using container image
resource "aws_lambda_function" "producer" {
  function_name = "${var.project_name}-producer"
  role         = aws_iam_role.lambda_producer_role.arn
  package_type = "Image"
  image_uri    = "${local.producer_repo_url}:latest"
  timeout      = var.lambda_timeout
  memory_size  = var.lambda_memory

  environment {
    variables = {
      QUEUE_URL      = aws_sqs_queue.main_queue.url
      FIFO_QUEUE_URL = aws_sqs_queue.fifo_queue.url
      REGION         = data.aws_region.current.name
    }
  }

  depends_on = [
    null_resource.producer_image,
    aws_iam_role_policy_attachment.lambda_producer_policy_attachment,
    aws_cloudwatch_log_group.producer_logs,
  ]

  tags = var.tags

  # Ensure the image exists before creating the function
  lifecycle {
    replace_triggered_by = [
      null_resource.producer_image
    ]
  }
}

# SQS Event source mapping for consumer
resource "aws_lambda_event_source_mapping" "sqs_trigger" {
  event_source_arn = aws_sqs_queue.main_queue.arn
  function_name    = aws_lambda_function.consumer.arn
  batch_size       = 10
  
  maximum_batching_window_in_seconds = 5
}

# EventBridge rule for scheduled processing
resource "aws_cloudwatch_event_rule" "scheduled_processing" {
  name                = "${var.project_name}-scheduled-processing"
  description         = "Trigger consumer processing every 5 minutes"
  schedule_expression = "rate(5 minutes)"
  
  tags = var.tags
}

# EventBridge target for consumer Lambda
resource "aws_cloudwatch_event_target" "lambda_target" {
  rule      = aws_cloudwatch_event_rule.scheduled_processing.name
  target_id = "ConsumerLambdaTarget"
  arn       = aws_lambda_function.consumer.arn
}

# Lambda permission for EventBridge
resource "aws_lambda_permission" "allow_eventbridge" {
  statement_id  = "AllowExecutionFromEventBridge"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.consumer.function_name
  principal     = "events.amazonaws.com"
  source_arn    = aws_cloudwatch_event_rule.scheduled_processing.arn
}

# Lambda permission for API Gateway
resource "aws_lambda_permission" "api_gateway_producer" {
  statement_id  = "AllowExecutionFromAPIGateway"
  action        = "lambda:InvokeFunction"
  function_name = aws_lambda_function.producer.function_name
  principal     = "apigateway.amazonaws.com"
  source_arn    = "${aws_api_gateway_rest_api.main.execution_arn}/*/*"
}