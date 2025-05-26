# Required variables
project_id         = "your-gcp-project-id"
credentials        = "./path/to/your/service-account-key.json"
mlflow_db_password = "your-secure-password"  # Create a strong password here

# Optional - uncomment and modify these if you want to change the defaults
region             = "us-central1"
zone               = "us-central1-a"
# mlflow_db_name         = "mlflow"
# mlflow_db_user         = "mlflow"
# mlflow_db_tier         = "db-f1-micro"
# mlflow_vm_machine_type = "e2-medium"