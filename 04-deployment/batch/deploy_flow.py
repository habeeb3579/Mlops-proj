"""
Script to deploy the taxi model flow using proper Prefect deployment methods
"""

from score_updated import apply_model_flow
from prefect.client.schemas.objects import ConcurrencyLimitConfig, ConcurrencyLimitStrategy
from prefect import flow, deploy
import os

def deploy_flow():
    """Deploy the flow using flow.serve() method for local development"""
    
    print("🚀 Serving apply_model_flow locally...")
    print("This will create a deployment and start a worker automatically.")
    print("Press Ctrl+C to stop.")
    
    try:
        # Use serve() instead of deploy() - this handles local deployments better
        apply_model_flow.serve(
            name="taxi-model-deployment",
            parameters={
                "taxi": "green",
                "year": 2021,
                "month": 4,
                "tracking_server": "http://localhost:5000",
                "model_name": "nyc-taxi-regressor-weighted-main9",
                "deployment_type": "local"
            },
            description="Local deployment for applying ML models to taxi data",
            tags=["ml", "taxi", "batch-processing", "local"],
            version="1.0.0",
            # Optional: Add scheduling (uncomment to use)
            # cron="0 2 * * *",  # Daily at 2 AM
            # interval=3600,     # Every hour (in seconds)
        )
        
    except KeyboardInterrupt:
        print("\n🛑 Flow serving stopped.")
    except Exception as e:
        print(f"❌ Serving failed: {str(e)}")
        raise

def deploy_with_process_work_pool():
    """Deploy using a process-based work pool with proper storage configuration"""
    
    print("🚀 Deploying apply_model_flow with process work pool...")
    
    try:
        # Get current working directory for remote storage
        remote_source = "/home/habeeb/Mlops-proj/04-deployment/batch"
        entrypoint = "score_updated.py:apply_model_flow"
        
        # Deploy with a process work pool using local file system storage
        apply_model_flow.from_source(
        source=remote_source,
        entrypoint=entrypoint 
        ).deploy(
                name="taxi-model-deployment-process",
                parameters={
                    "taxi": "green",
                    "year": 2021,
                    "month": 4,
                    "tracking_server": "http://localhost:5000",
                    "model_name": "nyc-taxi-regressor-weighted-main9",
                    "deployment_type": "local"
                },
                description="Process-based deployment for applying ML models to taxi data",
                tags=["ml", "taxi", "batch-processing", "process-pool"],
                version="1.0.0",
                work_pool_name="my-process-pool",
            )
        
        print("✅ Deployment created successfully!")
        print("   Flow Name: apply_model_flow")
        print("   Deployment Name: taxi-model-deployment-process")
        print("   Full Deployment ID: apply-model-flow/taxi-model-deployment-process")
        
        print("\n📋 Next steps:")
        print("1. Start a Prefect worker:")
        print("   prefect worker start --pool default-process-pool")
        print("\n2. Or run directly:")
        print("   prefect deployment run 'apply-model-flow/taxi-model-deployment-process'")
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        raise


def deploy_with_docker_work_pool():
    """Deploy using a docker-based work pool with proper storage configuration"""
    
    print("🚀 Deploying apply_model_flow with docker work pool...")
    
    try:
        
        # Deploy with a docker work pool using local file system storage
        apply_model_flow.deploy(
            name="taxi-model-deployment-docker",
            parameters={
                "taxi": "green",
                "year": 2021,
                "month": 4,
                "tracking_server": "http://localhost:5000",
                "model_name": "nyc-taxi-regressor-weighted-main9",
                "deployment_type": "local"
            },
            description="Process-based deployment for applying ML models to taxi data",
            tags=["ml", "taxi", "batch-processing", "process-pool"],
            version="1.0.0",
            work_pool_name="my-docker-pool",
            image="my-mlflow-prefect-image:latest", #"my-registry.com/my-docker-image:my-tag", # set to your image registry
            push=False, # switch to True to push to your image registry
            # Run once a day at midnight
            #cron="0 0 * * *",
        )
        
        print("✅ Deployment created successfully!")
        print("   Flow Name: apply-model-flow")
        print("   Deployment Name: taxi-model-deployment-docker")
        print("   Full Deployment ID: apply-model-flow/taxi-model-deployment-docker")
        
        print("\n📋 Next steps:")
        print("1. Start a Prefect worker:")
        print("   prefect worker start --pool my-docker-pool")
        print("\n2. Or run directly:")
        print("   prefect deployment run 'apply-model-flow/taxi-model-deployment-docker'")
        
    except Exception as e:
        print(f"❌ Deployment failed: {str(e)}")
        raise


def deploy_with_remote_storage():
    """Deploy using remote storage"""
    
    print("🚀 Deploying apply_model_flow with remote storage...")
    
    # You'll need to update these with your actual details (S3, GS, Github)
    remote_source = "" #"https://github.com/username/repository.git"
    entrypoint = "" #"path/to/your/flow.py:your_flow_function"
    
    try:
        flow.from_source(
        source=remote_source,
        entrypoint=entrypoint 
    ).deploy(
            name="taxi-model-deployment-remote",
            parameters={
                "taxi": "green",
                "year": 2021,
                "month": 4,
                "tracking_server": "http://localhost:5000",
                "model_name": "nyc-taxi-regressor-weighted-main9",
                "deployment_type": "local"
            },
            description="GitHub-based deployment for applying ML models to taxi data",
            tags=["ml", "taxi", "batch-processing", "github"],
            version="1.0.0",
            work_pool_name="default-process-pool",
        )
        
        print("✅ remote deployment created successfully!")
        print("   Flow Name: apply-model-flow")
        print("   Deployment Name: taxi-model-deployment-remote")
        
    except Exception as e:
        print(f"❌ remote deployment failed: {str(e)}")
        raise

def create_work_pool():
    """Helper function to create a process work pool if it doesn't exist"""
    
    import subprocess
    
    try:
        print("📋 Creating default-process-pool work pool...")
        result = subprocess.run([
            "prefect", "work-pool", "create", 
            "--type", "process",
            "default-process-pool"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Work pool created successfully!")
        else:
            print(f"ℹ️  Work pool might already exist: {result.stderr}")
            
    except Exception as e:
        print(f"⚠️  Could not create work pool: {str(e)}")

if __name__ == "__main__":
    import sys
    
    # Parse command line arguments
    deployment_type = sys.argv[1] if len(sys.argv) > 1 else "serve"
    
    if deployment_type == "serve":
        # python deploy_flow.py serve (default)
        deploy_flow()

    elif deployment_type == "process":
        # python deploy_flow.py process
        #create_work_pool()  # Ensure work pool exists
        deploy_with_process_work_pool()
        
    elif deployment_type == "docker":
        # python deploy_flow.py docker
        #create_work_pool()  # Ensure work pool exists
        deploy_with_docker_work_pool()
        
    elif deployment_type == "remote":
        # python deploy_flow.py remote
        deploy_with_remote_storage()
        
    else:
        print("❌ Unknown deployment type. Available options:")
        print("   python deploy_flow.py serve    # Local serving (default)")
        print("   python deploy_flow.py process  # Process work pool")
        print("   python deploy_flow.py docker   # Docker deployment")
        print("   python deploy_flow.py remote   # remote storage")