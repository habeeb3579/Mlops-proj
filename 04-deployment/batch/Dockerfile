FROM continuumio/miniconda3:latest

WORKDIR /app

# Copy and create conda environment
COPY conda.yaml .
RUN conda env create -f conda.yaml

# Ensure shell uses conda env
SHELL ["conda", "run", "-n", "mlflow-env", "/bin/bash", "-c"]

# Copy source code
COPY . .

# Expose ports (adjust as needed)
EXPOSE 4200
EXPOSE 5000

CMD ["python", "score_updated.py"]