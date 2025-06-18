# MLflow Tracking Server on AWS Free Tier with Terraform

This repository provisions a cost-efficient, production-grade [MLflow Tracking Server](https://mlflow.org/) using AWS Free Tier resources via Terraform.

It sets up:

- A `t2.micro` EC2 instance to host the MLflow tracking server
- A `db.t3.micro` PostgreSQL RDS instance as the backend store
- An S3 bucket as the default artifact store
- A systemd service to run MLflow server at boot
- A security group to allow traffic on ports 22, 80, and 5000

---

## 📦 Folder Structure

.
├── main.tf # Core infrastructure resources
├── variables.tf # Input variables
├── outputs.tf # Output values
├── terraform.tfvars # Your actual variable values (ignored in .gitignore)
├── user_data.sh # EC2 startup script to install and run MLflow
└── README.md # This file

---

## 🛠 Prerequisites

1. **Terraform**: >= 1.3  
   [Install Terraform](https://learn.hashicorp.com/tutorials/terraform/install-cli)

2. **AWS CLI** with credentials configured:
   ```bash
   aws configure
   An AWS Key Pair (create in EC2 → Key Pairs).
   Name it e.g., mlflow-key and save the .pem file locally.
   🔐 terraform.tfvars Example
   ```

Create a terraform.tfvars file to set your variables:

aws_region = "us-east-1"
mlflow_db_name = "mlflow"
mlflow_db_user = "mlflow"
mlflow_db_password = "YourStrongPasswordHere"
mlflow_key_pair = "mlflow-key" # Must match your actual AWS EC2 key pair
ec2_ami = "ami-0c2b8ca1dad447f8a" # Amazon Linux 2 (us-east-1)

### 🚀 Usage

1. Initialize Terraform
   terraform init
2. Review the Plan
   terraform plan
3. Apply the Infrastructure
   terraform apply
   Confirm with yes when prompted.

Terraform will:

Create a security group
Provision a PostgreSQL RDS instance (db.t3.micro)
Create an S3 bucket for artifacts
Launch a t2.micro EC2 instance
Install and run MLflow via user_data.sh

### 🌐 Access the MLflow Tracking UI

After successful deployment, Terraform will output:

Public IP of the MLflow server
Tracking URI
Artifact S3 URI
PostgreSQL URI (backend store)
Example:

mlflow_tracking_uri = "http://34.229.99.99:5000"
Open that URL in your browser to use MLflow.

### 🧹 Cleanup

To avoid AWS charges (especially for RDS after Free Tier hours):

```bash
terraform destroy
```

### 🧪 Example: Use the Tracking Server in Python

```bash
import mlflow

mlflow.set_tracking_uri("http://<mlflow_server_ip>:5000")
mlflow.set_experiment("my-experiment")

with mlflow.start_run():
    mlflow.log_param("alpha", 0.5)
    mlflow.log_metric("accuracy", 0.92)
```
