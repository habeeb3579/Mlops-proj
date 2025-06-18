FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

# Install pipenv and system dependencies
#RUN pip install --upgrade pip && pip install pipenv \
#    && apt-get update && apt-get install -y gcc

#RUN pip install -U pip
#RUN pip install pipenv

RUN pip install --upgrade pip

# Install system dependencies
RUN apt-get update && apt-get install -y gcc

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app

#COPY ["Pipfile", "Pipfile.lock", "./"]
#RUN pipenv install --system --deploy

COPY ["starter.py", "app.py", "./"]

EXPOSE 9696

CMD ["gunicorn", "-k", "uvicorn.workers.UvicornWorker", "app:app", "--bind", "0.0.0.0:9696"]