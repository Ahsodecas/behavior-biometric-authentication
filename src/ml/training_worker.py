from PyQt5.QtCore import QThread, pyqtSignal


class TrainingWorker(QThread):
    finished = pyqtSignal()

    def __init__(self, trainer: "ModelTrainer"):
        super().__init__()
        self.trainer = trainer

    def run(self):
        self.trainer.train()
        self.finished.emit()