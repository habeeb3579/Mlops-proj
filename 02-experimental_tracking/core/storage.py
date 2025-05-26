class StorageConfig:
    """Configuration for various storage backends for MLflow"""
    
    @staticmethod
    def get_tracking_uri(storage_type: str, **kwargs) -> str:
        """
        Generate the appropriate tracking URI based on storage type
        
        Args:
            storage_type: Type of storage ("sqlite", "postgresql", "aws", "gcp", "local")
            **kwargs: Additional parameters specific to the storage type
            
        Returns:
            str: MLflow tracking URI
        """
        if storage_type == "sqlite":
            db_path = kwargs.get("db_path", "mlflow.db")
            return f"sqlite:///{db_path}"
        
        elif storage_type == "postgresql":
            host = kwargs.get("host", "localhost")
            port = kwargs.get("port", 5432)
            database = kwargs.get("database", "mlflow")
            user = kwargs.get("user", "mlflow")
            password = kwargs.get("password", "mlflow")
            return f"postgresql://{user}:{password}@{host}:{port}/{database}"
        
        elif storage_type == "aws":
            s3_bucket = kwargs.get("s3_bucket", "mlflow-artifacts")
            region = kwargs.get("region", "us-east-1")
            # For AWS, typically use RDS for tracking and S3 for artifacts
            return f"https://{s3_bucket}.s3.{region}.amazonaws.com"
        
        elif storage_type == "gcp":
            project = kwargs.get("project", "mlflow-project")
            bucket = kwargs.get("bucket", "mlflow-artifacts")
            # For GCP, can use Cloud SQL for tracking and GCS for artifacts
            return f"gs://{bucket}"
        
        else:  # local
            return kwargs.get("tracking_uri", "mlruns")
    
    @staticmethod
    def get_artifact_location(storage_type: str, **kwargs) -> str:
        """
        Generate the appropriate artifact storage location based on storage type
        
        Args:
            storage_type: Type of storage ("local", "s3", "gcs")
            **kwargs: Additional parameters specific to the storage type
            
        Returns:
            str: Artifact location URI
        """
        if storage_type == "s3":
            bucket = kwargs.get("s3_bucket", "mlflow-artifacts")
            prefix = kwargs.get("prefix", "artifacts")
            return f"s3://{bucket}/{prefix}"
        
        elif storage_type == "gcs":
            bucket = kwargs.get("bucket", "mlflow-artifacts")
            prefix = kwargs.get("prefix", "artifacts")
            return f"gs://{bucket}/{prefix}"
        
        else:  # local
            return kwargs.get("artifact_location", "mlruns")
