First run (local path of:)

/home/habeeb/Mlops-proj/02-experimental_tracking/make_run.ipynb

then

/home/habeeb/Mlops-proj/02-experimental_tracking/local_deployment.ipynb (optional, used to set tracking_uri so that we can access it elsewhere)

this is where we are getting the mlflow registered model from.

# Method 1: Serve without creating a workpool (Recommended but this is temporary and not permanently deployed)

## Terminal 1: Start Prefect server

```bash
prefect server start
```

## Terminal 2: Serve flow

```bash
python deploy_flow.py serve
```

## to test

Run

```bash
run_score.ipynb
```

then

```bash
 test_temporary_deployment.ipynb
```

# Method 2: Deploy + Process OR Docker Worker (see https://docs.prefect.io/v3/how-to-guides/deployments/deploy-via-python)

## Terminal 1: Start Prefect server

```bash
prefect server start
```

## Terminal 2: Create/Start worker

create a docker workpool

```bash
prefect work-pool create --type docker my-docker-pool
```

or

```bash
# Create a process work pool
prefect work-pool create --type process my-process-pool
```

```bash
prefect worker start --pool my-docker-pool
```

or

```bash
prefect worker start --pool my-process-pool
```

## Terminal 3: Serve the flow

```bash
python deploy_flow.py docker
```

or

```bash
python deploy_flow.py process
```

## Terminal 4: Run backfill

```bash
python backfill_script.py
```

## Note: use hyphen instead of underscore when running prefect deployment run apply-model-flow/taxi-model-deployment
