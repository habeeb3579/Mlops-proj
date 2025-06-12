#!/bin/bash

# Colors for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}=== MLflow on GCP Setup Script ===${NC}"

# Check if gcloud is installed
if ! command -v gcloud &> /dev/null; then
    echo "Google Cloud SDK is not installed. Please install it first:"
    echo "https://cloud.google.com/sdk/docs/install"
    exit 1
fi

# Check if terraform is installed
if ! command -v terraform &> /dev/null; then
    echo "Terraform is not installed. Please install it first:"
    echo "https://learn.hashicorp.com/tutorials/terraform/install-cli"
    exit 1
fi

# Check if user is logged in to gcloud
if ! gcloud auth list --filter=status:ACTIVE --format="value(account)" &> /dev/null; then
    echo "You are not logged in to Google Cloud. Please login first:"
    gcloud auth login
fi

# Get the current project
current_project=$(gcloud config get-value project)
echo -e "${GREEN}Current GCP project:${NC} $current_project"

# Confirm project
read -p "Do you want to use this project for MLflow deployment? (y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Please set your desired project with: gcloud config set project YOUR_PROJECT_ID"
    exit 1
fi

# Check if the tfvars file exists
if [ ! -f terraform.tfvars ]; then
    echo "terraform.tfvars file not found. Creating a template..."
    
    # Generate a random password
    random_password=$(openssl rand -base64 12)
    
    cat > terraform.tfvars << EOF
project_id         = "$current_project"
region             = "us-central1"
zone               = "us-central1-a"
mlflow_db_password = "$random_password"  # Auto-generated password, please change if needed
EOF
    
    echo -e "${YELLOW}Created terraform.tfvars file with default values.${NC}"
    echo -e "${YELLOW}Please review and edit this file if necessary before continuing.${NC}"
    
    read -p "Press Enter to continue or Ctrl+C to abort and edit the file..."
fi

# Initialize terraform
echo -e "\n${GREEN}Initializing Terraform...${NC}"
terraform init

# Plan terraform
echo -e "\n${GREEN}Creating Terraform plan...${NC}"
terraform plan -out=tfplan

# Ask for confirmation
echo -e "\n${YELLOW}Review the plan above carefully.${NC}"
read -p "Do you want to apply this Terraform plan? (y/n): " confirm
if [[ $confirm != [yY] && $confirm != [yY][eE][sS] ]]; then
    echo "Terraform apply cancelled."
    exit 0
fi

# Apply terraform
echo -e "\n${GREEN}Applying Terraform plan...${NC}"
terraform apply tfplan

# Output success message
if [ $? -eq 0 ]; then
    echo -e "\n${GREEN}=== MLflow deployment successful! ===${NC}"
    
    # Extract outputs
    mlflow_url=$(terraform output -raw mlflow_tracking_uri)
    gcs_bucket=$(terraform output -raw mlflow_artifact_store)
    
    echo -e "\n${YELLOW}MLflow UI:${NC} $mlflow_url"
    echo -e "${YELLOW}MLflow Artifact Store:${NC} $gcs_bucket"
    
    echo -e "\n${GREEN}Wait a few minutes for the VM to complete the installation process.${NC}"
    echo -e "${GREEN}You can check the startup script progress with:${NC}"
    echo "gcloud compute ssh mlflow-tracking-server --command='sudo tail -f /var/log/syslog'"
    
    echo -e "\n${YELLOW}To connect your local environment to this MLflow server:${NC}"
    echo "export MLFLOW_TRACKING_URI=$mlflow_url"
else
    echo -e "\n${YELLOW}Terraform apply failed. Please check the error messages above.${NC}"
fi

echo -e "\n${GREEN}Done!${NC}"