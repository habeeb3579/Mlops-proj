terraform {
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

resource "aws_s3_bucket" "mlflow_artifact_store" {
  bucket        = "${var.mlflow_db_name}-mlflow-artifacts"
  force_destroy = true
}

resource "aws_security_group" "mlflow_sg" {
  name        = "mlflow-sg"
  description = "Allow inbound access for MLflow"

  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 5000
    to_port     = 5000
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  ingress {
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_db_instance" "mlflow_db" {
  identifier           = "mlflow-db"
  allocated_storage    = 20
  engine               = "postgres"
  engine_version       = "13.4"
  instance_class       = "db.t3.micro"
  username             = var.mlflow_db_user
  password             = var.mlflow_db_password
  db_name              = var.mlflow_db_name
  publicly_accessible  = true
  skip_final_snapshot  = true
  deletion_protection  = false

  vpc_security_group_ids = [aws_security_group.mlflow_sg.id]
}

resource "aws_instance" "mlflow_server" {
  ami                    = var.ec2_ami
  instance_type          = "t2.micro"
  key_name               = var.mlflow_key_pair
  security_groups        = [aws_security_group.mlflow_sg.name]

  user_data              = file("${path.module}/user_data.sh")

  tags = {
    Name = "mlflow-tracking-server"
  }
}