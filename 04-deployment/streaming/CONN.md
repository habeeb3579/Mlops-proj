## 🔗 Overall Architecture

Clients send POST requests to an API Gateway /stream endpoint.
API Gateway invokes the Producer Lambda.
Producer Lambda sends messages to SQS queues.
Consumer Lambda is triggered by SQS, processes the messages.
Logs are recorded in CloudWatch.
IAM roles/policies secure access among components.

## 📡 api\*gateway.tf – API Gateway Configuration

aws*api_gateway_rest_api.main: Creates a REST API.
aws_api_gateway_resource.stream: Defines /stream endpoint path.
aws_api_gateway_method.*: Configures HTTP methods (POST, OPTIONS) and validation for the /stream resource.
aws_api_gateway_model.stream_model: JSON schema to validate request body.
aws_api_gateway_integration.lambda_integration: Connects API Gateway POST to producer Lambda using AWS_PROXY (full request passthrough).
aws_api_gateway_integration.*\_response & method_response\**: Sets CORS headers and response settings.
aws_api_gateway_deployment.main: Deploys the API config.
aws_api_gateway_stage.main: Deploys to a named stage like dev.

## 🔐 iam.tf – IAM Roles and Policies

aws_iam_role.lambda_producer_role: Trusts Lambda to assume it.
aws_iam_policy.lambda_producer_policy: Allows sending to SQS + logging.
aws_iam_role.lambda_consumer_role + policy: Allows reading from SQS + logging.
aws_iam_role.api_gateway_role + policy: Allows API Gateway to invoke Lambda and write logs.
aws_iam_role_policy_attachment: Binds the policies to the roles.

## 🧠 lambda.tf – Producer and Consumer Logic

data.archive*file.*: Zips the Lambda source code from Python files.
aws*lambda_function.producer: Sends messages to SQS.
aws_lambda_function.consumer: Triggered by SQS, processes messages.
aws_lambda_event_source_mapping.sqs_trigger: Binds the main SQS queue to the consumer Lambda.
aws_cloudwatch_event_rule (optional): Triggers consumer Lambda every 5 mins.
aws_lambda_permission.*: Allows API Gateway or EventBridge to invoke Lambdas.

## 📬 sqs.tf – Queues for Asynchronous Communication

aws*sqs_queue.main_queue: Standard queue for general message flow.
aws_sqs_queue.dlq: Dead-letter queue for main queue errors.
aws_sqs_queue_redrive_policy.*: Moves failed messages to DLQ after 3 attempts.
aws_sqs_queue.fifo_queue & fifo_dlq: For ordered delivery if needed.

## 📊 main.tf – Provider and Log Groups

Sets up AWS provider and region.
data.aws_caller_identity & data.aws_region: Fetch metadata.
aws_cloudwatch_log_group.\*: Capture logs from Lambda/API Gateway (14-day retention).

## 📤 outputs.tf – Useful Outputs

API URL, Lambda names/ARNs, queue URLs.
Shows AWS region/account and free-tier usage summary.

## ⚙️ terraform.tfvars & variables.tf

Customizes region, names, timeouts, memory, tags, and free-tier settings.
Controls SQS visibility, Lambda timeout/memory, and API rate limits.

## 💡 How They Are Connected

API Gateway → Producer Lambda → SQS → Consumer Lambda.
Each component is IAM-bound with least privilege.
CloudWatch monitors Lambda and API behavior.
Redrive policies help handle failures reliably.
