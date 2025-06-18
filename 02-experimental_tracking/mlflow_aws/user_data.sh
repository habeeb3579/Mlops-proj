#!/bin/bash
yum update -y
amazon-linux-extras enable python3.8
yum install -y python3.8 postgresql nginx git
python3.8 -m ensurepip
python3.8 -m pip install mlflow boto3 psycopg2-binary

cat > /etc/systemd/system/mlflow.service << SERVICEEOF
[Unit]
Description=MLflow Server
After=network.target

[Service]
ExecStart=/usr/local/bin/mlflow server \
  --backend-store-uri postgresql://${MLFLOW_DB_USER}:${MLFLOW_DB_PASSWORD}@${MLFLOW_DB_HOST}:5432/${MLFLOW_DB_NAME} \
  --default-artifact-root s3://${MLFLOW_BUCKET} \
  --host 0.0.0.0
Restart=always

[Install]
WantedBy=multi-user.target
SERVICEEOF

systemctl daemon-reexec
systemctl daemon-reload
systemctl enable mlflow
systemctl start mlflow
