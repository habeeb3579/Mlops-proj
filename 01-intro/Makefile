# Makefile for NYC Taxi Duration Predictor

# Default variables
PYTHON := python3
SCRIPT := NYC_trip_duration_prediction.py
DATA_YEAR := 2021
DATA_MONTH := 1
MODEL := linear_regression
TARGET_TRANSFORM := log
RANDOM_STATE := 42

ENV_NAME := nyc-taxi-env
PYTHON_VERSION := 3.9
CONDA_HOME := /opt/conda/bin  # Update if different
WITH_IPYKERNEL := true

# Directory structure
.PHONY: setup
setup:
	mkdir -p models plots

.PHONY: install-requirements
install-requirements:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt


# Add to .PHONY targets
.PHONY: create-env install-conda-requirements

create-env:
	@echo "Creating Conda environment '$(ENV_NAME)'..."
	. $(CONDA_HOME)/etc/profile.d/conda.sh && \
	conda create --name $(ENV_NAME) python=$(PYTHON_VERSION) -y && \
	conda activate $(ENV_NAME); \
	if [ "$(WITH_IPYKERNEL)" = "true" ]; then \
		conda install -y ipykernel && \
		python -m ipykernel install --user --name $(ENV_NAME) --display-name "Python ($(ENV_NAME))"; \
	fi

install-conda-requirements: create-env
	. $(CONDA_HOME)/etc/profile.d/conda.sh && \
	conda activate $(ENV_NAME) && \
	pip install -r requirements.txt

# Train various models
.PHONY: train-linear train-ridge train-lasso train-rf train-gb

train-linear: setup
	$(PYTHON) $(SCRIPT) train \
		--train-year $(DATA_YEAR) \
		--train-month $(DATA_MONTH) \
		--model linear_regression \
		--target-transform $(TARGET_TRANSFORM) \
		--random-state $(RANDOM_STATE)

train-ridge: setup
	$(PYTHON) $(SCRIPT) train \
		--train-year $(DATA_YEAR) \
		--train-month $(DATA_MONTH) \
		--model ridge \
		--target-transform $(TARGET_TRANSFORM) \
		--random-state $(RANDOM_STATE)

train-lasso: setup
	$(PYTHON) $(SCRIPT) train \
		--train-year $(DATA_YEAR) \
		--train-month $(DATA_MONTH) \
		--model lasso \
		--target-transform $(TARGET_TRANSFORM) \
		--random-state $(RANDOM_STATE)

train-rf: setup
	$(PYTHON) $(SCRIPT) train \
		--train-year $(DATA_YEAR) \
		--train-month $(DATA_MONTH) \
		--model random_forest \
		--target-transform $(TARGET_TRANSFORM) \
		--random-state $(RANDOM_STATE)

train-gb: setup
	$(PYTHON) $(SCRIPT) train \
		--train-year $(DATA_YEAR) \
		--train-month $(DATA_MONTH) \
		--model gradient_boosting \
		--target-transform $(TARGET_TRANSFORM) \
		--random-state $(RANDOM_STATE)

# Train all models at once
.PHONY: train-all
train-all: train-linear train-ridge train-lasso train-rf train-gb

# Train with specific year/month
.PHONY: train-custom
train-custom: setup
	$(PYTHON) $(SCRIPT) train \
		--train-year $(DATA_YEAR) \
		--train-month $(DATA_MONTH) \
		--model $(MODEL) \
		--target-transform $(TARGET_TRANSFORM) \
		--random-state $(RANDOM_STATE)

# Predict using a pre-trained model (requires MODEL_PATH)
.PHONY: predict
predict:
	@if [ -z "$(MODEL_PATH)" ]; then \
		echo "Error: MODEL_PATH is required. Use make predict MODEL_PATH=path/to/model"; \
		exit 1; \
	fi
	$(PYTHON) $(SCRIPT) predict \
		--model-path $(MODEL_PATH) \
		--year $(DATA_YEAR) \
		--month $(DATA_MONTH)

# Example with specified model path
.PHONY: predict-example
predict-example:
	@echo "This is an example - you should modify MODEL_PATH to point to your actual model file"
	$(PYTHON) $(SCRIPT) predict \
		--model-path models/nyc_taxi_duration_linear_regression_20250507_000000.joblib \
		--year 2021 \
		--month 2

# Clean up
.PHONY: clean
clean:
	rm -rf __pycache__
	
.PHONY: clean-all
clean-all: clean
	rm -rf models plots

# Show help
.PHONY: help
help:
	@echo "NYC Taxi Duration Predictor Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  setup            - Create necessary directories"
	@echo "  train-linear     - Train a linear regression model"
	@echo "  train-ridge      - Train a ridge regression model"
	@echo "  train-lasso      - Train a lasso regression model"
	@echo "  train-rf         - Train a random forest model"
	@echo "  train-gb         - Train a gradient boosting model"
	@echo "  train-all        - Train all model types"
	@echo "  train-custom     - Train with custom parameters"
	@echo "                     Example: make train-custom DATA_YEAR=2022 DATA_MONTH=3 MODEL=ridge"
	@echo "  predict          - Predict using a pre-trained model"
	@echo "                     Usage: make predict MODEL_PATH=path/to/model DATA_YEAR=2022 DATA_MONTH=3"
	@echo "  predict-example  - Example prediction command (needs modification for your model)"
	@echo "  clean            - Remove Python cache files"
	@echo "  clean-all        - Remove all generated files (models, plots, cache)"
	@echo "  create-env         - Create Conda environment"
	@echo "  install-conda-requirements - Install dependencies in Conda environment"
	@echo "  install-requirements - Install Python dependencies from requirements.txt"
	@echo ""
	@echo "Default Variables (can be overridden):"
	@echo "  PYTHON=$(PYTHON)"
	@echo "  SCRIPT=$(SCRIPT)"
	@echo "  DATA_YEAR=$(DATA_YEAR)"
	@echo "  DATA_MONTH=$(DATA_MONTH)"
	@echo "  MODEL=$(MODEL)"
	@echo "  TARGET_TRANSFORM=$(TARGET_TRANSFORM)"
	@echo "  RANDOM_STATE=$(RANDOM_STATE)"