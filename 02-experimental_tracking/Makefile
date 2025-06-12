# Makefile for NYC Taxi Trip Duration Prediction with MLflow
# 
# This Makefile provides shortcuts for common operations related to
# training and managing ML models for NYC taxi trip duration prediction.

# Default configuration
PYTHON := python
TRAIN_YEAR := 2021
TRAIN_MONTH := 1
VAL_YEAR := 2021
VAL_MONTH := 2
TAXI_TYPE := green
TRACKING_STORE := gcp
DB_PATH := mlflow.db
HOST := localhost
PORT := 5000
TRACKING_URI := http://35.224.212.79:5000
ARTIFACT_STORE := local
BUCKET := dezoomfinal-mlflow-artifacts
PREFIX := none
ARTIFACT_LOCATION := gs://dezoomfinal-mlflow-artifacts
EXPERIMENT_NAME := nyc-taxi-exp-weighted-main9
MODEL_NAME := nyc-taxi-regressor-weighted-main9
CAT_TRANSFORMER := onehot
NUM_TRANSFORMER := none
MODELS := all
TUNE := true
REGISTER := true
MAX_EVALS := 5

# Directory structure
DATA_DIR := data
MODELS_DIR := models_newest
MLRUNS_DIR := mlruns

# Help command
help:
	@echo "NYC Taxi Trip Duration Prediction with MLflow - Makefile Commands"
	@echo ""
	@echo "Basic Commands:"
	@echo "  run-all                 Run experiment with all models"
	@echo "  run-models              Run experiment with specific models (set MODELS=model1,model2)"
	@echo "  run-custom              Run experiment with custom parameters"
	@echo "  run-quick               Run experiment with LinearRegression only (no tuning)"
	@echo "  register-models         Register best model without training"
	@echo "  serve-model             Serve the production model using MLflow"
	@echo "  mlflow-ui               Launch the MLflow UI"
	@echo ""
	@echo "Data Commands:"
	@echo "  download-data           Download NYC taxi data"
	@echo "  prepare-data            Prepare data for modeling"
	@echo ""
	@echo "Storage Commands:"
	@echo "  setup-sqlite            Set up SQLite tracking backend"
	@echo "  setup-postgres          Set up PostgreSQL tracking backend"
	@echo "  setup-aws               Set up AWS S3 tracking backend"
	@echo "  setup-gcp               Set up GCP tracking backend"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  clean                   Remove temporary files and SQLite DB"
	@echo "  clean-all               Remove all generated files including models and data"
	@echo "  requirements            Generate requirements.txt"
	@echo ""
	@echo "Environment Variables:"
	@echo "  MODELS                  Models to train (comma-separated or single model, default: all)"
	@echo "  TRACKING_STORE          Tracking backend (sqlite, postgresql, aws, gcp, local)"
	@echo "  ARTIFACT_STORE          Artifact store type (local, s3, gcs)"
	@echo "  BUCKET                  S3 or GCS bucket name"
	@echo "  PREFIX                  Subdirectory in cloud bucket"
	@echo "  ARTIFACT_LOCATION       Local artifact directory (if local)"
	@echo "  HOST                    MLflow tracking server host"
	@echo "  PORT                    MLflow tracking server port"
	@echo "  TRACKING_URI            Direct tracking URI (optional for remote)"
	@echo ""

# Ensure directories exist
$(DATA_DIR):
	mkdir -p $(DATA_DIR)

$(MODELS_DIR):
	mkdir -p $(MODELS_DIR)

# Convert boolean flags to argparse flags
define get_tune_flag
$(if $(filter $(TUNE),false),--no-tune)
endef

define get_register_flag
$(if $(filter $(REGISTER),false),--no-register)
endef

# Core Commands
run-all: $(DATA_DIR) $(MODELS_DIR)
	$(PYTHON) -m main \
		--train-year $(TRAIN_YEAR) --train-month $(TRAIN_MONTH) \
		--val-year $(VAL_YEAR) --val-month $(VAL_MONTH) \
		--taxi $(TAXI_TYPE) \
		--tracking-store $(TRACKING_STORE) --db-path $(DB_PATH) \
		--host $(HOST) --port $(PORT) --tracking-uri $(TRACKING_URI) \
		--artifact-store $(ARTIFACT_STORE) --bucket $(BUCKET) --prefix $(PREFIX) --artifact-location $(ARTIFACT_LOCATION) \
		--experiment-name $(EXPERIMENT_NAME) --model-name $(MODEL_NAME) \
		--categorical-transformer $(CAT_TRANSFORMER) --numerical-transformer $(NUM_TRANSFORMER) \
		--models all --max-evals $(MAX_EVALS) \
		$(call get_tune_flag) $(call get_register_flag)

run-models: $(DATA_DIR) $(MODELS_DIR)
	$(PYTHON) -m main \
		--train-year $(TRAIN_YEAR) --train-month $(TRAIN_MONTH) \
		--val-year $(VAL_YEAR) --val-month $(VAL_MONTH) \
		--taxi $(TAXI_TYPE) \
		--tracking-store $(TRACKING_STORE) --db-path $(DB_PATH) \
		--host $(HOST) --port $(PORT) --tracking-uri $(TRACKING_URI) \
		--artifact-store $(ARTIFACT_STORE) --bucket $(BUCKET) --prefix $(PREFIX) --artifact-location $(ARTIFACT_LOCATION) \
		--experiment-name $(EXPERIMENT_NAME) --model-name $(MODEL_NAME) \
		--categorical-transformer $(CAT_TRANSFORMER) --numerical-transformer $(NUM_TRANSFORMER) \
		--models $(MODELS) --max-evals $(MAX_EVALS) \
		$(call get_tune_flag) $(call get_register_flag)

run-custom: $(DATA_DIR) $(MODELS_DIR)
	$(PYTHON) -m main \
		--train-year $(TRAIN_YEAR) --train-month $(TRAIN_MONTH) \
		--val-year $(VAL_YEAR) --val-month $(VAL_MONTH) \
		--taxi $(TAXI_TYPE) \
		--tracking-store $(TRACKING_STORE) --db-path $(DB_PATH) \
		--host $(HOST) --port $(PORT) --tracking-uri $(TRACKING_URI) \
		--artifact-store $(ARTIFACT_STORE) --bucket $(BUCKET) --prefix $(PREFIX) --artifact-location $(ARTIFACT_LOCATION) \
		--experiment-name $(EXPERIMENT_NAME) --model-name $(MODEL_NAME) \
		--categorical-transformer $(CAT_TRANSFORMER) --numerical-transformer $(NUM_TRANSFORMER) \
		--models $(MODELS) --max-evals $(MAX_EVALS) \
		$(call get_tune_flag) $(call get_register_flag)

run-quick: $(DATA_DIR) $(MODELS_DIR)
	$(PYTHON) -m main \
		--train-year $(TRAIN_YEAR) --train-month $(TRAIN_MONTH) \
		--val-year $(VAL_YEAR) --val-month $(VAL_MONTH) \
		--taxi $(TAXI_TYPE) \
		--tracking-store $(TRACKING_STORE) --db-path $(DB_PATH) \
		--host $(HOST) --port $(PORT) --tracking-uri $(TRACKING_URI) \
		--artifact-store $(ARTIFACT_STORE) --bucket $(BUCKET) --prefix $(PREFIX) --artifact-location $(ARTIFACT_LOCATION) \
		--experiment-name $(EXPERIMENT_NAME) --model-name $(MODEL_NAME) \
		--categorical-transformer $(CAT_TRANSFORMER) --numerical-transformer $(NUM_TRANSFORMER) \
		--models LinearRegression --no-tune

register-models:
	$(PYTHON) -c "from core.manager import ExperimentManager; manager = ExperimentManager(tracking_config={'tracking_store': '$(TRACKING_STORE)', 'db_path': '$(DB_PATH)', 'host': '$(HOST)', 'port': $(PORT), 'tracking_uri': '$(TRACKING_URI)'}, experiment_name='$(EXPERIMENT_NAME)', model_registry_name='$(MODEL_NAME)'); manager.register_best_model()"

serve-model:
	@echo "Starting MLflow model server for production model $(MODEL_NAME)"
	mlflow models serve -m "models:/$(MODEL_NAME)/Production" -p 5001

mlflow-ui:
	@echo "Starting MLflow UI with tracking store: $(TRACKING_STORE)"
	mlflow ui --backend-store-uri sqlite:///$(DB_PATH)

# Data commands
download-data: $(DATA_DIR)
	$(PYTHON) -c "from core.experiment import NYCTaxiDurationExperiment; experiment = NYCTaxiDurationExperiment(data_dir='$(DATA_DIR)'); experiment.download_data($(TRAIN_YEAR), $(TRAIN_MONTH), '$(TAXI_TYPE)'); experiment.download_data($(VAL_YEAR), $(VAL_MONTH), '$(TAXI_TYPE)')"

prepare-data: download-data
	$(PYTHON) -c "from core.experiment import NYCTaxiDurationExperiment; experiment = NYCTaxiDurationExperiment(data_dir='$(DATA_DIR)'); train_data = experiment.download_data($(TRAIN_YEAR), $(TRAIN_MONTH), '$(TAXI_TYPE)'); experiment.prepare_data(train_data); val_data = experiment.download_data($(VAL_YEAR), $(VAL_MONTH), '$(TAXI_TYPE)'); experiment.prepare_data(val_data)"

# Storage setup commands
setup-sqlite:
	@echo "Setting up SQLite tracking backend at $(DB_PATH)"
	$(PYTHON) -c "import mlflow; mlflow.set_tracking_uri('sqlite:///$(DB_PATH)')"

setup-postgres:
	@echo "Setting up PostgreSQL tracking backend"
	$(PYTHON) -c "import mlflow; from core.storage_new import StorageConfig; print(StorageConfig.get_tracking_uri('postgresql', host='$(DB_HOST)', port=$(DB_PORT), database='$(DB_NAME)', user='$(DB_USER)', password='$(DB_PASSWORD)'))"

setup-aws:
	@echo "Setting up AWS S3 tracking backend"
	$(PYTHON) -c "import mlflow; from core.storage_new import StorageConfig; uri = StorageConfig.get_tracking_uri('aws', host='$(HOST)', port=$(PORT)); print(uri)"

setup-gcp:
	@echo "Setting up GCP tracking backend"
	$(PYTHON) -c "import mlflow; from core.storage_new import StorageConfig; uri = StorageConfig.get_tracking_uri('gcp', host='$(HOST)', port=$(PORT)); print(uri)"

# Maintenance
clean:
	@echo "Cleaning up temporary files"
	rm -f $(DB_PATH)
	rm -f *.pyc
	rm -rf __pycache__

clean-all: clean
	@echo "Cleaning up all generated files"
	rm -rf $(MLRUNS_DIR)
	rm -rf $(MODELS_DIR)
	if [ "$(force)" = "true" ]; then rm -rf $(DATA_DIR); fi

requirements:
	@echo "Generating requirements.txt"
	@echo "mlflow>=2.0.0" > requirements.txt
	@echo "scikit-learn>=1.0.0" >> requirements.txt
	@echo "xgboost>=1.5.0" >> requirements.txt
	@echo "pandas>=1.3.0" >> requirements.txt
	@echo "numpy>=1.20.0" >> requirements.txt
	@echo "hyperopt>=0.2.5" >> requirements.txt
	@echo "psycopg2-binary>=2.9.0" >> requirements.txt
	@echo "boto3>=1.24.0" >> requirements.txt
	@echo "google-cloud-storage>=2.0.0" >> requirements.txt

.PHONY: help run-all run-models run-custom run-quick register-models serve-model mlflow-ui download-data prepare-data setup-sqlite setup-postgres setup-aws setup-gcp clean clean-all requirements
