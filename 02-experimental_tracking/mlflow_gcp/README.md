# MLflow on GCP Setup with Service Account Key

## Prerequisites

1. **Terraform installed** - [Download here](https://www.terraform.io/downloads.html)
2. **GCP Account** with billing enabled
3. **Service Account Key** (we'll create this)

## Step 1: Create a GCP Project (Optional)

If you want to create a new project:

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click on the project dropdown at the top
3. Click "New Project"
4. Enter a project name and ID
5. Click "Create"

## Step 2: Create a Service Account and Key

1. In the Google Cloud Console, go to **IAM & Admin** > **Service Accounts**
2. Click **"Create Service Account"**
3. Enter a name like `mlflow-terraform`
4. Click **"Create and Continue"**
5. Add these roles:
   - **Project Editor** (or more specific roles like Compute Admin, Storage Admin, Cloud SQL Admin)
   - **Service Account User**
6. Click **"Continue"** then **"Done"**
7. Click on the created service account
8. Go to the **"Keys"** tab
9. Click **"Add Key"** > **"Create new key"**
10. Choose **JSON** format
11. Click **"Create"** - this downloads your key file
12. Save this file securely (e.g., as `mlflow-sa-key.json`)

## Step 3: Setup Terraform Files

1. Create a new directory for your MLflow deployment
2. Save the modified Terraform configuration as `main.tf`
3. Create a `terraform.tfvars` file with your specific values:

```hcl
project_id         = "your-actual-project-id"
credentials        = "./mlflow-sa-key.json"  # Path to your key file
mlflow_db_password = "YourSecurePassword123!"
region             = "us-central1"
zone               = "us-central1-a"
```

## Step 4: Deploy MLflow

1. **Initialize Terraform:**

   ```bash
   terraform init
   ```

2. **Plan the deployment:**

   ```bash
   terraform plan
   ```

3. **Apply the deployment:**

   ```bash
   terraform apply
   ```

4. **Type `yes` when prompted**

## Step 5: Access MLflow

After deployment (wait 5-10 minutes for VM setup):

1. Get the MLflow URL:

   ```bash
   terraform output mlflow_tracking_uri
   ```

2. Open the URL in your browser

3. **Connect your local environment:**
   ```bash
   export MLFLOW_TRACKING_URI=$(terraform output -raw mlflow_tracking_uri)
   ```

## Step 6: Test Your Setup

Create a simple test script:

```bash
python3 example.py
```

## Troubleshooting

### Check VM startup progress:

```bash
# If you have gcloud installed (optional)
gcloud compute ssh mlflow-tracking-server --project=YOUR_PROJECT_ID --command='sudo tail -f /var/log/syslog'
```

### Check MLflow service status:

```bash
gcloud compute ssh mlflow-tracking-server --project=YOUR_PROJECT_ID --command='sudo systemctl status mlflow'
```

### Common Issues:

1. **"APIs not enabled"** - The Terraform script now automatically enables required APIs
2. **"Insufficient permissions"** - Make sure your service account has the required roles
3. **Database connection issues** - Check if the database IP configuration allows connections

## Security Notes

- **Never commit your service account key file to version control**
- **Add your key file to `.gitignore`**
- **Use more restrictive firewall rules in production**
- **Enable deletion protection for the database in production**
- **Use stronger machine types for production workloads**

## Cleanup

To destroy all resources:

```bash
terraform destroy
```

## Cost Optimization

For minimal costs, the default configuration uses:

- `db-f1-micro` for PostgreSQL (shared-core, 0.6 GB RAM)
- `e2-medium` for the VM (2 vCPUs, 4 GB RAM)

You can adjust these in your `terraform.tfvars` file.

## File Structure

Your project should look like:

```
mlflow-gcp/
├── main.tf
├── terraform.tfvars
├── mlflow-sa-key.json
└── .gitignore (add *.json to ignore key files)
```
