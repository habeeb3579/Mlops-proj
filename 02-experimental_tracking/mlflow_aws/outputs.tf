output "mlflow_server_public_ip" {
  description = "Public IP of the MLflow server"
  value       = aws_instance.mlflow_server.public_ip
}

output "mlflow_tracking_uri" {
  description = "MLflow tracking URI"
  value       = "http://${aws_instance.mlflow_server.public_ip}:5000"
}

output "mlflow_db_connection" {
  description = "PostgreSQL URI for backend store"
  value       = "postgresql://${var.mlflow_db_user}:${var.mlflow_db_password}@${aws_db_instance.mlflow_db.address}:5432/${var.mlflow_db_name}"
  sensitive   = true
}

output "mlflow_artifact_store" {
  description = "S3 URI for MLflow artifacts"
  value       = "s3://${aws_s3_bucket.mlflow_artifact_store.id}"
}
