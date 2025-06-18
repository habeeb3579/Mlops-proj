variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "mlflow_db_name" {
  description = "Name of the MLflow PostgreSQL database"
  type        = string
  default     = "mlflow"
}

variable "mlflow_db_user" {
  description = "Username for the MLflow PostgreSQL database"
  type        = string
  default     = "mlflow"
}

variable "mlflow_db_password" {
  description = "Password for the MLflow PostgreSQL database"
  type        = string
  sensitive   = true
}

variable "mlflow_key_pair" {
  description = "EC2 key pair name for SSH access"
  type        = string
}

variable "ec2_ami" {
  description = "AMI ID for the EC2 instance"
  type        = string
  default     = "ami-0c2b8ca1dad447f8a" # Amazon Linux 2, us-east-1
}
