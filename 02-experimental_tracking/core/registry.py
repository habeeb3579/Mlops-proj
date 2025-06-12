import numpy as np
import os
from typing import Dict, List, Tuple, Union, Optional, Any
import pandas as pd
import yaml
import urllib.parse

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from utils.logger import get_logger

# GCS and S3 support
from google.cloud import storage
import boto3

# Type aliases for clarity
ModelType = Any  # Generic model type
ArrayLike = Union[List, np.ndarray, pd.Series]
DictConfig = Dict[str, Any]

logger = get_logger(__name__)

class ModelRegistry:
    """Manages model registration and versioning in MLflow"""
    
    def __init__(self, client: MlflowClient):
        self.client = client
    
    def register_model(self, run_id: str, model_uri: str, name: str) -> None:
        """
        Register a model with MLflow
        
        Args:
            run_id: ID of the run containing the model
            model_uri: URI pointing to the model
            name: Name to register the model under
        """
        result = mlflow.register_model(model_uri=model_uri, name=name)
        logger.info(f"Model registered as {name} v{result.version}")
        return result
    
    def transition_to_production(self, name: str, version: int, archive_existing: bool = False) -> None:
        """
        Transition a model version to production stage
        
        Args:
            name: Name of the registered model
            version: Version to transition
            archive_existing: Whether to archive existing production models
        """
        self.client.transition_model_version_stage(
            name=name,
            version=version,
            stage="Production",
            archive_existing_versions=archive_existing
        )
        logger.info(f"Model {name} v{version} transitioned to Production stage")
    
    def set_alias(self, name: str, version: int, alias: str) -> None:
        """
        Set an alias for a model version
        
        Args:
            name: Name of the registered model
            version: Version to set alias for
            alias: Alias to set
        """
        self.client.set_registered_model_alias(name, alias, version)
        logger.info(f"Set alias '{alias}' for {name} v{version}")
    
    
    def find_best_models(self, experiment_id: str, metric: str = "rmse", max_results: int = 5) -> List:
        # Step 1: Retrieve a broad list of recent runs (e.g. 200)
        runs = self.client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",  # No filtering here due to MLflow limitations
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=200
        )

        # Step 2: Python-side filtering
        valid_runs = []
        for run in runs:
            is_not_nested = "mlflow.parentRunId" not in run.data.tags
            has_metric = metric in run.data.metrics

            if is_not_nested and has_metric:
                valid_runs.append(run)

        # Step 3: Sort by metric value (ascending)
        valid_runs.sort(key=lambda r: r.data.metrics[metric])

        return valid_runs[:max_results]


    # def get_model_size(self, run_id: str, artifact_path: str = "model") -> float:
    #     """
    #     Get the size of a model artifact in kilobytes (KB).

    #     - First attempts to read `model_size_bytes` from MLmodel.
    #     - If not available, computes the total size of files in the model directory.

    #     Args:
    #         run_id (str): MLflow run ID
    #         artifact_path (str): Artifact path of the model (default: "model")

    #     Returns:
    #         float: Size in kilobytes (KB)
    #     """
    #     run = self.client.get_run(run_id)

    #     # Step 1: Try to read size from MLmodel
    #     artifact_uri = run.info.artifact_uri
    #     parsed = urllib.parse.urlparse(artifact_uri)
    #     model_dir = os.path.join(parsed.path, artifact_path)
    #     mlmodel_path = os.path.join(model_dir, "MLmodel")

    #     if os.path.exists(mlmodel_path):
    #         try:
    #             with open(mlmodel_path, "r") as f:
    #                 mlmodel = yaml.safe_load(f)
    #             size_bytes = mlmodel.get("model_size_bytes")
    #             if size_bytes is not None:
    #                 return int(size_bytes) / 1024.0
    #         except Exception as e:
    #             print(f"[Warning] Failed to read model_size_bytes from MLmodel: {e}")

    #     # Step 2: Fallback to manual calculation
    #     if parsed.scheme != "file":
    #         raise NotImplementedError("Manual size calculation only supported for local files.")

    #     total_size = 0
    #     for dirpath, _, filenames in os.walk(model_dir):
    #         for fname in filenames:
    #             fpath = os.path.join(dirpath, fname)
    #             total_size += os.path.getsize(fpath)

    #     return total_size / 1024.0  # Return KB

    def get_model_size(self, run_id: str, artifact_path: str = "model") -> float:
        """
        Get the size of a model artifact in kilobytes (KB).

        - First attempts to read `model_size_bytes` from MLmodel file.
        - If not available, computes size from underlying storage (local, GCS, or S3).

        Args:
            run_id (str): MLflow run ID
            artifact_path (str): Path to model artifact

        Returns:
            float: Size in KB
        """
        run = self.client.get_run(run_id)
        artifact_uri = run.info.artifact_uri
        parsed = urllib.parse.urlparse(artifact_uri)

        scheme = parsed.scheme
        bucket_or_host = parsed.netloc
        base_path = parsed.path.lstrip("/")
        full_prefix = os.path.join(base_path, artifact_path)
        mlmodel_filename = os.path.join(full_prefix, "MLmodel")

        try:
            if scheme == "file":
                mlmodel_path = os.path.join(parsed.path, artifact_path, "MLmodel")
                if os.path.exists(mlmodel_path):
                    with open(mlmodel_path, "r") as f:
                        mlmodel = yaml.safe_load(f)
                    size_bytes = mlmodel.get("model_size_bytes")
                    if size_bytes is not None:
                        return int(size_bytes) / 1024.0

            elif scheme == "gs":
                client = storage.Client()
                bucket = client.bucket(bucket_or_host)
                blob = bucket.blob(mlmodel_filename)
                content = blob.download_as_text()
                mlmodel = yaml.safe_load(content)
                size_bytes = mlmodel.get("model_size_bytes")
                if size_bytes is not None:
                    return int(size_bytes) / 1024.0

            elif scheme == "s3":
                s3 = boto3.client("s3")
                response = s3.get_object(Bucket=bucket_or_host, Key=mlmodel_filename)
                content = response["Body"].read().decode("utf-8")
                mlmodel = yaml.safe_load(content)
                size_bytes = mlmodel.get("model_size_bytes")
                if size_bytes is not None:
                    return int(size_bytes) / 1024.0

        except Exception as e:
            logger.warning(f"Could not read model_size_bytes from MLmodel: {e}")

        # Fallback: compute manually
        if scheme == "file":
            model_dir = os.path.join(parsed.path, artifact_path)
            total_size = 0
            for dirpath, _, filenames in os.walk(model_dir):
                for fname in filenames:
                    fpath = os.path.join(dirpath, fname)
                    total_size += os.path.getsize(fpath)
            return total_size / 1024.0

        elif scheme == "gs":
            return self._get_gcs_size(bucket_or_host, full_prefix)

        elif scheme == "s3":
            return self._get_s3_size(bucket_or_host, full_prefix)

        raise NotImplementedError(f"Unsupported artifact scheme: {scheme}")

    
    def _get_gcs_size(self, bucket_name: str, prefix: str) -> float:
        client = storage.Client()
        total_size = sum(blob.size for blob in client.list_blobs(bucket_name, prefix=prefix))
        return total_size / 1024.0

    def _get_s3_size(self, bucket_name: str, prefix: str) -> float:
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        total_size = 0
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                total_size += obj["Size"]
        return total_size / 1024.0

    def find_production_candidate(self, experiment_id: str, metric: str = "rmse",
                              size_weight: float = 0.2,
                              performance_weight: float = 0.8,
                              max_candidates: int = 10) -> Dict:
        """
        Select the best model run based on performance and size.

        Args:
            experiment_id (str): MLflow experiment ID
            metric (str): Performance metric (e.g., 'rmse')
            size_weight (float): Weight to assign to model size (smaller is better)
            performance_weight (float): Weight to assign to metric performance (better is lower for RMSE)
            max_candidates (int): Number of runs to evaluate

        Returns:
            Dict: Best run info or None
        """
        # Step 1: Get valid parent runs with the desired metric
        all_runs = self.client.search_runs(
            experiment_ids=[experiment_id],
            filter_string="",
            run_view_type=ViewType.ACTIVE_ONLY,
            max_results=200
        )

        parent_runs = [
            run for run in all_runs
            if "mlflow.parentRunId" not in run.data.tags and metric in run.data.metrics
        ]

        if not parent_runs:
            logger.warning("No parent runs with the specified metric found.")
            return None

        # Step 2: Compute RMSE and model size for each run
        candidates = []
        for run in parent_runs[:max_candidates]:
            run_id = run.info.run_id
            try:
                rmse_val = run.data.metrics[metric]
                size_kb = self.get_model_size(run_id, artifact_path="model")
                candidates.append({
                    "run": run,
                    "rmse": rmse_val,
                    "size_kb": size_kb
                })
            except Exception as e:
                logger.warning(f"Skipping run {run_id} due to error: {e}")

        if not candidates:
            logger.warning("No candidates with valid model sizes and metrics.")
            return None

        # Step 3: Normalize RMSE and model size (min-max)
        rmses = [c["rmse"] for c in candidates]
        sizes = [c["size_kb"] for c in candidates]
        min_rmse, max_rmse = min(rmses), max(rmses)
        min_size, max_size = min(sizes), max(sizes)

        for c in candidates:
            c["rmse_score"] = 1 - (c["rmse"] - min_rmse) / (max_rmse - min_rmse + 1e-8)
            c["size_score"] = 1 - (c["size_kb"] - min_size) / (max_size - min_size + 1e-8)
            c["combined_score"] = (
                performance_weight * c["rmse_score"] + size_weight * c["size_score"]
            )

        # Step 4: Select best candidate
        best = max(candidates, key=lambda x: x["combined_score"])
        run = best["run"]

        return {
            "run_id": run.info.run_id,
            "metrics": run.data.metrics,
            "params": run.data.params,
            "size_kb": best["size_kb"],
            "rmse": best["rmse"],
            "combined_score": best["combined_score"]
        }
