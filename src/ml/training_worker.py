from PyQt5.QtCore import QThread, pyqtSignal

from src.ml.data_preprocessor import DataPreprocessor
from src.ml.model_trainer import ModelTrainer


class TrainingWorker(QThread):
    dataProcFinished = pyqtSignal()
    trainFinished = pyqtSignal()

    def __init__(self, trainer: "ModelTrainer", preprocessor: "DataPreprocessor", username):
        super().__init__()
        self.trainer = trainer
        self.preprocessor = preprocessor
        self.username = username

    def run(self):
        self.preprocessor.build_training_csv()
        self.dataProcFinished.emit()
        self.trainer.initialize()
        self.trainer.train()
        self.trainFinished.emit()