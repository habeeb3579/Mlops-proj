terraform {
  required_providers {
    google = {
      source  = "hashicorp/google"
      version = "~> 4.0"
    }
  }
}

# Variables definition
variable "project_id" {
  description = "The GCP project ID"
  type        = string
}

variable "credentials" {
  description = "Path to the GCP service account key file"
  type        = string
}

variable "region" {
  description = "The GCP region"
  type        = string
  default     = "us-central1"
}

variable "zone" {
  description = "The GCP zone"
  type        = string
  default     = "us-central1-a"
}

variable "mlflow_db_tier" {
  description = "The tier of the PostgreSQL instance"
  type        = string
  default     = "db-f1-micro" # Smallest machine type, adjust based on your needs
}

variable "mlflow_db_name" {
  description = "MLflow database name"
  type        = string
  default     = "mlflow"
}

variable "mlflow_db_user" {
  description = "MLflow database user"
  type        = string
  default     = "mlflow"
}

variable "mlflow_db_password" {
  description = "MLflow database password"
  type        = string
  sensitive   = true
}

variable "mlflow_vm_machine_type" {
  description = "The machine type for the MLflow tracking server VM"
  type        = string
  default     = "e2-medium" # 2 vCPUs, 4 GB RAM
}

# Configure the Google Cloud provider with service account key
provider "google" {
  credentials = file(var.credentials)
  project     = var.project_id
  region      = var.region
  zone        = var.zone
}

# Enable required APIs
resource "google_project_service" "compute" {
  service            = "compute.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "sql" {
  service            = "sql-component.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "sqladmin" {
  service            = "sqladmin.googleapis.com"
  disable_on_destroy = false
}

resource "google_project_service" "storage" {
  service            = "storage.googleapis.com"
  disable_on_destroy = false
}

# Create a GCS bucket for MLflow artifacts
resource "google_storage_bucket" "mlflow_artifact_store" {
  name          = "${var.project_id}-mlflow-artifacts"
  location      = var.region
  force_destroy = true
  
  uniform_bucket_level_access = true
  
  depends_on = [google_project_service.storage]
}

# Create a Cloud SQL PostgreSQL instance for MLflow metadata
resource "google_sql_database_instance" "mlflow_db" {
  name             = "mlflow-postgresql"
  database_version = "POSTGRES_13"
  region           = var.region
  
  settings {
    tier = var.mlflow_db_tier
    
    backup_configuration {
      enabled = true
    }
    
    ip_configuration {
      ipv4_enabled = true
      authorized_networks {
        name  = "all"
        value = "0.0.0.0/0"
      }
    }
  }

  deletion_protection = false # Set to true in production
  
  depends_on = [google_project_service.sqladmin]
}

# Create a database
resource "google_sql_database" "mlflow_database" {
  name     = var.mlflow_db_name
  instance = google_sql_database_instance.mlflow_db.name
}

# Create a user
resource "google_sql_user" "mlflow_user" {
  name     = var.mlflow_db_user
  instance = google_sql_database_instance.mlflow_db.name
  password = var.mlflow_db_password
}

# Create a Compute Engine VM to run MLflow tracking server
resource "google_compute_instance" "mlflow_tracking_server" {
  name         = "mlflow-tracking-server"
  machine_type = var.mlflow_vm_machine_type
  
  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-11"
      size  = 20 # GB
    }
  }
  
  network_interface {
    network = "default"
    access_config {
      # This will give the instance a public IP
    }
  }
  
  # Grant the VM service account access to the GCS bucket
  service_account {
    scopes = ["storage-rw", "cloud-platform"]
  }

  metadata_startup_script = <<-EOF
    #!/bin/bash
    apt-get update && apt-get install -y python3-pip python3-dev python3-venv libpq-dev gcc
    
    # Install Google Cloud SDK
    curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-345.0.0-linux-x86_64.tar.gz
    tar zxvf google-cloud-sdk-345.0.0-linux-x86_64.tar.gz
    ./google-cloud-sdk/install.sh --quiet
    
    # Create a Python virtual environment for MLflow
    python3 -m venv /opt/mlflow
    source /opt/mlflow/bin/activate
    
    # Install MLflow and dependencies
    pip install mlflow psycopg2-binary google-cloud-storage
    
    # Extract database connection info
    DB_IP="${google_sql_database_instance.mlflow_db.public_ip_address}"
    DB_NAME="${var.mlflow_db_name}"
    DB_USER="${var.mlflow_db_user}"
    DB_PASSWORD="${var.mlflow_db_password}"
    GCS_BUCKET="${google_storage_bucket.mlflow_artifact_store.name}"
    
    # Write MLflow service file
    cat > /etc/systemd/system/mlflow.service << SERVICEEOF
    [Unit]
    Description=MLflow Tracking Server
    After=network.target
    
    [Service]
    User=root
    WorkingDirectory=/opt/mlflow
    ExecStart=/opt/mlflow/bin/mlflow server \
      --backend-store-uri postgresql://$DB_USER:$DB_PASSWORD@$DB_IP:5432/$DB_NAME \
      --default-artifact-root gs://$GCS_BUCKET \
      --host 0.0.0.0
    Restart=on-failure
    
    [Install]
    WantedBy=multi-user.target
    SERVICEEOF
    
    # Enable and start MLflow service
    systemctl daemon-reload
    systemctl enable mlflow
    systemctl start mlflow
    
    # Install and configure Nginx as a reverse proxy (optional but recommended)
    apt-get install -y nginx
    
    cat > /etc/nginx/sites-available/mlflow << NGINXEOF
    server {
        listen 80;
        
        location / {
            proxy_pass http://localhost:5000;
            proxy_set_header Host \$host;
            proxy_set_header X-Real-IP \$remote_addr;
            proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
        }
    }
    NGINXEOF
    
    ln -s /etc/nginx/sites-available/mlflow /etc/nginx/sites-enabled/
    systemctl restart nginx
  EOF
  
  tags = ["mlflow", "http-server"]
  
  depends_on = [google_project_service.compute]
}

# Create firewall rule to allow access to MLflow UI
resource "google_compute_firewall" "mlflow_firewall" {
  name    = "allow-mlflow"
  network = "default"

  allow {
    protocol = "tcp"
    ports    = ["80", "5000"]
  }

  source_ranges = ["0.0.0.0/0"]
  target_tags   = ["http-server"]
}

# Outputs
output "mlflow_tracking_server_ip" {
  value = google_compute_instance.mlflow_tracking_server.network_interface[0].access_config[0].nat_ip
  description = "The public IP address of the MLflow tracking server"
}

output "mlflow_db_connection" {
  value = "postgresql://${var.mlflow_db_user}:${var.mlflow_db_password}@${google_sql_database_instance.mlflow_db.public_ip_address}:5432/${var.mlflow_db_name}"
  description = "The PostgreSQL connection string for MLflow (sensitive)"
  sensitive = true
}

output "mlflow_artifact_store" {
  value = "gs://${google_storage_bucket.mlflow_artifact_store.name}"
  description = "The GCS bucket URI for MLflow artifacts"
}

output "mlflow_tracking_uri" {
  value = "http://${google_compute_instance.mlflow_tracking_server.network_interface[0].access_config[0].nat_ip}"
  description = "The MLflow tracking URI"
}