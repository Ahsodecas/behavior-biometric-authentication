# src/utils/background_auth_manager.py

import os
import csv
import time
from threading import Thread, Event
import torch
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

from src.utils.data_utility import DataUtility
from src.auth.authentication_decision_maker import AuthenticationDecisionMaker


class BackgroundAuthManager(QObject):
    """
    Handles continuous background authentication:
    - Collects mouse/keyboard data periodically
    - Saves raw/features
    - Requests authentication
    - Updates UI via signal
    """

    status_update = pyqtSignal(str)  # Signal to update the floating window

    def __init__(self, username, authenticator_model_path, interval=300):
        super().__init__()
        self.username = username
        self.interval = interval  # seconds
        self.timer = QTimer()
        self.timer.timeout.connect(self.run_cycle)

        # Core utilities
        self.data_utility = DataUtility(username=username)

        # Thread control
        self._stop_event = Event()

    def start(self):
        self.status_update.emit("Background auth started.")
        self.data_utility.start_background_collection()
        self.timer.start(self.interval * 1000)

    def stop(self):
        self.timer.stop()
        self._stop_event.set()
        self.data_utility.stop_background_collection()
        self.status_update.emit("Background auth stopped.")

    def run_cycle(self):
        """
        Collect features, save data, authenticate user, update status label.
        """
        try:
            self.data_utility.mouse_data_collector.save_to_csv(filename="mouse.csv")
            self.data_utility.reset()

        except Exception as e:
            self.status_update.emit(f"Error: {str(e)}")
