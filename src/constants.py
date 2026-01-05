import os


ROOT_DIR = os.path.dirname(os.path.abspath(__name__))
PROJECT_ROOT = os.path.dirname(ROOT_DIR)
PATH_MODELS = os.path.join(PROJECT_ROOT, "models")
PATH_DATASETS = os.path.join(PROJECT_ROOT, "datasets")
PATH_EXTRACTED = os.path.join(PROJECT_ROOT, "extracted_features")
PATH_LOGS = os.path.join(PROJECT_ROOT, "logs")
MODEL_PATH = os.path.join(PATH_MODELS, "snn_final.pt")

PASSWORD = ".tie5Roanl"

TARGET_FAR = 0.01