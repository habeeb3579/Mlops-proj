class StorageConfig:
    """Configuration for various storage backends for MLflow"""

    @staticmethod
    def get_tracking_uri(storage_type: str, **kwargs) -> str:
        """
        Generate the appropriate tracking URI based on storage type.

        Args:
            storage_type: One of ("sqlite", "postgresql", "aws", "gcp", "local", "remote")
            **kwargs: Parameters like db_path, host, port, user, password, etc.

        Returns:
            str: MLflow tracking URI
        """
        if storage_type == "sqlite":
            return f"sqlite:///{kwargs.get('db_path', 'mlflow.db')}"

        if storage_type == "postgresql":
            return "postgresql://{user}:{password}@{host}:{port}/{database}".format(
                user=kwargs.get("user", "mlflow"),
                password=kwargs.get("password", "mlflow"),
                host=kwargs.get("host", "localhost"),
                port=kwargs.get("port", 5432),
                database=kwargs.get("database", "mlflow")
            )

        # Covers aws, gcp, remote, or custom server-based URIs
        return kwargs.get("tracking_uri", f"http://{kwargs.get('host', 'localhost')}:{kwargs.get('port', 5000)}")

    @staticmethod
    def get_artifact_location(storage_type: str, **kwargs) -> str:
        """
        Generate the appropriate artifact storage location based on storage type.

        Args:
            storage_type: One of ("local", "s3", "gcs")
            **kwargs: Parameters like bucket, s3_bucket, prefix, etc.

        Returns:
            str: Artifact location URI
        """
        def join_path(base: str, prefix: str | None) -> str:
            return f"{base}/{prefix}" if prefix not in ["none", "None", ""] else base

        if storage_type == "s3":
            return join_path(f"s3://{kwargs.get('s3_bucket', 'mlflow-artifacts')}", kwargs.get("prefix"))

        if storage_type == "gcs":
            return join_path(f"gs://{kwargs.get('bucket', 'mlflow-artifacts')}", kwargs.get("prefix"))

        return kwargs.get("artifact_location", "mlruns")