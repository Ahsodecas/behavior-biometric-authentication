# src/utils/background_auth_manager.py

import time
from threading import Event
from PyQt5.QtCore import QObject, pyqtSignal, QTimer

from src.utils.data_utility import DataUtility


class BackgroundAuthManager(QObject):
    """
    Dummy background authentication manager.
    Cycles through authentication states on a timer and updates UI.
    """

    status_update = pyqtSignal(str)

    STATUSES = [
        ("● Idle", "idle"),
        ("● Collecting data", "collecting"),
        ("● Feature extraction", "features"),
        ("● Authentication", "auth"),
    ]

    def __init__(self, username, authenticator_model_path=None, interval=2):
        super().__init__()

        self.username = username
        self.interval = interval  # seconds between state changes
        self._stop_event = Event()

        self.data_utility = DataUtility(username=username)

        self._state_index = 0

        self.timer = QTimer()
        self.timer.timeout.connect(self._advance_state)

    def start(self):
        """Start dummy background authentication."""
        self._state_index = 0
        self.status_update.emit(self.STATUSES[0][0])

        self.data_utility.start_background_collection()
        self.timer.start(self.interval * 1000)

    def stop(self):
        """Stop background authentication."""
        self.timer.stop()
        self._stop_event.set()
        self.data_utility.stop_background_collection()
        self.status_update.emit("● Idle")

    def _advance_state(self):
        """
        Dummy logic that cycles through auth states.
        """
        if self._stop_event.is_set():
            return

        self._state_index = (self._state_index + 1) % len(self.STATUSES)
        label, state = self.STATUSES[self._state_index]

        self.status_update.emit(label)

        if state == "collecting":
            pass  # simulate data collection
        elif state == "features":
            pass  # simulate feature extraction
        elif state == "auth":
            pass  # simulate authentication
