# src/utils/background_auth_manager.py

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from threading import Event, Thread
from PyQt5.QtCore import QObject, pyqtSignal

from src.utils.data_utility import DataUtility


class BackgroundAuthManager(QObject):
    """
    Background authentication manager:
    - Collects mouse data for a fixed duration
    - Saves raw CSV
    - Runs authentication model
    - Emits UI status updates
    """

    status_update = pyqtSignal(str)
    auth_result = pyqtSignal(bool, float)  # (accepted, mean_score)

    # =========================
    # Authentication config
    # =========================
    WINDOW_SIZE = 128
    STEP_SIZE = 64
    DECISION_THRESHOLD = 0.7

    def __init__(self, username, data_utility, authenticator_model_path):
        super().__init__()

        self.username = username
        self.data_utility = data_utility
        self.authenticator_model_path = authenticator_model_path

        self._stop_event = Event()
        self._thread = None

        self.model = tf.keras.models.load_model(self.authenticator_model_path)

    # =========================
    # Public API
    # =========================
    def start(self):
        self._stop_event.clear()
        self.status_update.emit("● Collecting data")

        self._thread = Thread(
            target=self._collect_data_for_duration,
            daemon=True
        )
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self.data_utility.stop_background_collection()
        self.status_update.emit("● Idle")

    # =========================
    # Collection + Authentication
    # =========================
    def _collect_data_for_duration(self, duration_minutes=3):
        print("Start collecting data...")
        self.data_utility.start_background_collection()

        start_time = time.time()
        duration_seconds = duration_minutes * 60

        while (
            not self._stop_event.is_set()
            and (time.time() - start_time) < duration_seconds
        ):
            time.sleep(1)

        self.data_utility.stop_background_collection()
        print("Stopped collecting data...")
        self.data_utility.save_mouse_raw_csv(filename="mouse_raw.csv")

        df = self.data_utility.mouse_data_collector.get_data()

        self.status_update.emit("● Authenticating")
        print("Authenticating...")
        accepted, mean_score = self._authenticate(df)
        print(f"Authenticated: {accepted}, mean_score: {mean_score}")

        self.auth_result.emit(bool(accepted), float(mean_score))


    # =========================
    # Authentication Logic
    # =========================
    def _authenticate(self, df):
        if df.empty:
            return False, 0.0

        signal = self._compute_velocity(df)
        windows = self._create_windows(
            signal,
            self.WINDOW_SIZE,
            self.STEP_SIZE
        )

        if len(windows) == 0:
            return False, 0.0

        scores = self.model.predict(windows, verbose=0).ravel()
        mean_score = scores.mean()

        accepted = mean_score >= self.DECISION_THRESHOLD
        return accepted, float(mean_score)

    # =========================
    # Feature Engineering
    # =========================
    @staticmethod
    def _compute_velocity(df):
        df = df.copy()
        df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")

        df["dx"] = df["x"].diff()
        df["dy"] = df["y"].diff()
        df["dt"] = df["timestamp"].diff()

        df = df[df["dt"] > 0]

        df["vx"] = df["dx"] / df["dt"]
        df["vy"] = df["dy"] / df["dt"]

        return df[["vx", "vy"]].dropna().values

    # =========================
    # Windowing
    # =========================
    @staticmethod
    def _create_windows(signal, window_size, step_size):
        windows = []
        for start in range(0, len(signal) - window_size, step_size):
            windows.append(signal[start:start + window_size])
        return np.array(windows)
