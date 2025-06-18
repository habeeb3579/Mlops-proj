# backfill_script.py
from prefect import get_client
from prefect.deployments import run_deployment
from prefect.client.schemas.filters import DeploymentFilter
from datetime import datetime
import asyncio

async def get_deployment_id(deployment_name: str, flow_name: str):
    """Get deployment ID by name"""
    async with get_client() as client:
        deployments = await client.read_deployments(
            deployment_filter=DeploymentFilter(
                name={"any_": [deployment_name]}
            )
        )
        
        if not deployments:
            raise ValueError(f"No deployment found with name: {deployment_name}")
        
        return deployments[0].id

async def trigger_backfill_flow(
    deployment_id: str,
    taxi: str, 
    year: int, 
    month: int, 
    run_date: str, 
    tracking_server: str, 
    model_name: str
):
    """Trigger a single flow run for backfill using deployment ID"""
    async with get_client() as client:
        # Construct the run date
        if run_date:
            run_date_iso = datetime.fromisoformat(run_date).isoformat()
        else:
            run_date_iso = None

        # Create flow run with parameters
        flow_run = await client.create_flow_run_from_deployment(
            deployment_id=deployment_id,
            parameters={
                "taxi": taxi,
                "year": year,
                "month": month,
                "run_date": run_date_iso,
                "tracking_server": tracking_server,
                "model_name": model_name,
                "deployment_type": "local"
            }
        )
        
        print(f"✅ Created flow run {flow_run.id} for {taxi} taxi, {year}-{month:02d}")
        return flow_run.id

async def trigger_backfill_flow_v2(
    deployment_name: str,
    taxi: str, 
    year: int, 
    month: int, 
    run_date: str, 
    tracking_server: str, 
    model_name: str
):
    """Alternative method: Trigger flow run using run_deployment"""
    
    # Construct the run date
    if run_date:
        run_date_iso = datetime.fromisoformat(run_date).isoformat()
    else:
        run_date_iso = None

    try:
        # Use run_deployment method
        flow_run = await run_deployment(
            name=deployment_name,  # Format: "flow_name/deployment_name"
            parameters={
                "taxi": taxi,
                "year": year,
                "month": month,
                "run_date": run_date_iso,
                "tracking_server": tracking_server,
                "model_name": model_name,
                "deployment_type": "local"
            },
            timeout=0,  # Don't wait for completion
        )
        
        if flow_run is None:
            raise ValueError(f"run_deployment returned None - check deployment name format")
            
        print(f"✅ Created flow run {flow_run.id} for {taxi} taxi, {year}-{month:02d}")
        return flow_run.id
        
    except Exception as e:
        print(f"❌ Error in trigger_backfill_flow_v2: {str(e)}")
        raise

async def backfill_months(
    start_month: int, 
    end_month: int, 
    year: int, 
    taxi: str, 
    tracking_server: str, 
    model_name: str, 
    deployment_name: str = "apply-model-flow/taxi-model-deployment",
    run_date: str = None,
    concurrent: bool = True,
    use_run_deployment: bool = True
):
    """
    Main function to trigger backfill for multiple months
    
    Args:
        start_month: Starting month (1-12)
        end_month: Ending month (1-12) 
        year: Year for backfill
        taxi: Taxi type (green, yellow)
        tracking_server: MLflow tracking server URI
        model_name: MLflow model name
        deployment_name: Full deployment name (flow_name/deployment_name)
        run_date: Optional run date
        concurrent: Whether to run months concurrently or sequentially
        use_run_deployment: Use run_deployment() vs client method
    """
    
    print(f"🚀 Starting backfill for {taxi} taxi from {year}-{start_month:02d} to {year}-{end_month:02d}")
    print(f"📋 Using deployment: {deployment_name}")
    
    try:
        if use_run_deployment:
            # Method 1: Using run_deployment (simpler)
            if concurrent:
                tasks = []
                for month in range(start_month, end_month + 1):
                    tasks.append(
                        trigger_backfill_flow_v2(
                            deployment_name, taxi, year, month, run_date, tracking_server, model_name
                        )
                    )
                
                flow_run_ids = await asyncio.gather(*tasks)
                print(f"🎉 Successfully created {len(flow_run_ids)} concurrent flow runs")
                
            else:
                flow_run_ids = []
                for month in range(start_month, end_month + 1):
                    flow_run_id = await trigger_backfill_flow_v2(
                        deployment_name, taxi, year, month, run_date, tracking_server, model_name
                    )
                    flow_run_ids.append(flow_run_id)
                
                print(f"🎉 Successfully created {len(flow_run_ids)} sequential flow runs")
        
        else:
            # Method 2: Using client.create_flow_run_from_deployment (original method)
            # Extract flow name and deployment name
            flow_name = deployment_name.split('/')[0]
            dep_name = deployment_name.split('/')[1]
            
            deployment_id = await get_deployment_id(dep_name, flow_name)
            print(f"📋 Found deployment ID: {deployment_id}")
            
            if concurrent:
                tasks = []
                for month in range(start_month, end_month + 1):
                    tasks.append(
                        trigger_backfill_flow(
                            deployment_id, taxi, year, month, run_date, tracking_server, model_name
                        )
                    )
                
                flow_run_ids = await asyncio.gather(*tasks)
                print(f"🎉 Successfully created {len(flow_run_ids)} concurrent flow runs")
                
            else:
                flow_run_ids = []
                for month in range(start_month, end_month + 1):
                    flow_run_id = await trigger_backfill_flow(
                        deployment_id, taxi, year, month, run_date, tracking_server, model_name
                    )
                    flow_run_ids.append(flow_run_id)
                
                print(f"🎉 Successfully created {len(flow_run_ids)} sequential flow runs")
        
        return flow_run_ids
        
    except Exception as e:
        print(f"❌ Error during backfill: {str(e)}")
        raise

async def check_flow_run_status(flow_run_ids: list):
    """Check the status of flow runs"""
    async with get_client() as client:
        print("\n📊 Checking flow run statuses:")
        for run_id in flow_run_ids:
            flow_run = await client.read_flow_run(run_id)
            print(f"  {run_id}: {flow_run.state_type}")

async def list_deployments():
    """List all available deployments"""
    async with get_client() as client:
        deployments = await client.read_deployments()
        print("📋 Available deployments:")
        for deployment in deployments:
            print(f"deployment: {deployment}")
            print(f"  - {deployment.name} (ID: {deployment.id})")
            print(f"    Flow: {deployment.flow_id}")
        return deployments

if __name__ == "__main__":
    # Define parameters for backfill
    start_month = 7  # Start month (April)
    end_month = 9    # End month (June)
    year = 2021      # Year for backfill
    taxi_type = "green"  # Taxi type (green or yellow)
    tracking_server = "http://localhost:5000"  # MLflow tracking URI
    model_name = "nyc-taxi-regressor-weighted-main9"  # MLflow model name
    
    async def main():
        try:
            # First, list available deployments to verify our deployment exists
            print("🔍 Checking available deployments...")
            deployments = await list_deployments()
            
            if not deployments:
                print("❌ No deployments found. Make sure to run 'python deploy_flow.py' first.")
                return
            
            # Run the backfill process
            flow_run_ids = await backfill_months(
                start_month=start_month,
                end_month=end_month,
                year=year,
                taxi=taxi_type,
                tracking_server=tracking_server,
                model_name=model_name,
                deployment_name="apply-model-flow/taxi-model-deployment",  # Full deployment name
                concurrent=True,  # Set to False for sequential execution
                use_run_deployment=False  # Use the simpler run_deployment method
            )
            
            # Optional: Check status of created runs
            print("\n⏳ Waiting a moment before checking status...")
            await asyncio.sleep(2)
            await check_flow_run_status(flow_run_ids)
            
        except Exception as e:
            print(f"❌ Backfill failed: {str(e)}")
            print("\n💡 Troubleshooting tips:")
            print("1. Make sure Prefect server is running: prefect server start")
            print("2. Make sure the flow is deployed: python deploy_flow.py")
            print("3. Make sure a worker is running: prefect worker start --pool default")
    
    asyncio.run(main())