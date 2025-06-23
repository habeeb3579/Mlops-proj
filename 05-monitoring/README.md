## ✅ Instructions

**Create the environment:**

```bash
conda env create -f conda.yaml
```

**Activate the environment:**

```bash
conda activate ml-monitoring
```

**Register it as a Jupyter kernel:**

```bash
python -m ipykernel install --user --name=ml-monitoring --display-name "Python (ml-monitoring)"
```

## Docker-compose

remove folder mounted to a container

```bash
docker run --rm -v /home/habeeb/Mlops-proj:/mnt alpine sh -c "rm -rf /mnt/05-monitoring/services/config"
```

docker-compose up --build

## mlflow_evidently

```bash
mlflow ui --port 5001 --backend-store-uri sqlite:///mlflow.db
```
