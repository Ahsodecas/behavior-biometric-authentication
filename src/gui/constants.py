import os


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(ROOT_DIR))
PATH_MODELS = os.path.join(PROJECT_ROOT, "models")
PATH_DATASETS = os.path.join(PROJECT_ROOT, "datasets")
PATH_EXTRACTED = os.path.join(PROJECT_ROOT, "extracted_features")

MODEL_PATH = os.path.join(PATH_MODELS, "snn_final.pt")

PASSWORD = ".tie5Roanl"