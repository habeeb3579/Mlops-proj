# MLOps Zoomcamp 2023 Week 3

![Prefect logo](./images/logo.svg)

---

This repo contains Python code to accompany the videos that show how to use Prefect for MLOps. We will create workflows that you can orchestrate and observe..

# Setup

## Clone the repo

Clone the repo locally.

## Install packages

In a conda environment with Python 3.10.12 or similar, install all package dependencies with

```bash
pip install -r requirements.txt
```

## Start the Prefect server locally

Create another window and activate your conda environment. Start the Prefect API server locally with

```bash
prefect server start
```

```bash
prefect project init
```

create a process workpool named zoompool

```bash
prefect deploy 3.4/orchestrate.py:main_flow -n taxi1 -p zoompool
```

if deploying using deployment.yaml file, use

```bash
prefect deploy --all
```

```bash
prefect deployment run main_flow/taxi1
```

start the workpool

```bash
prefect worker start --pool "zoompool"
```

see available blocks

```bash
prefect block ls
prefect block type ls
```

```bash
prefect block register -m prefect-aws
prefect block register -m prefect_gcp
```

## Optional: use Prefect Cloud for added capabilties

Signup and use for free at https://app.prefect.cloud
