# src/utils/background_auth_manager.py

import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from threading import Event, Thread
from PyQt5.QtCore import QObject, pyqtSignal
import src.constants as constants
from src.utils.data_utility import DataUtility
from src.auth.authentication_decision_maker import AuthenticationDecisionMaker


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

    def __init__(self, username, data_utility, authenticator_model_path):
        super().__init__()

        self.username = username
        self.data_utility = data_utility
        self.authenticator_model_path = authenticator_model_path

        self.data_utility = DataUtility(username=username)

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
    def _collect_data_for_duration(self, duration_minutes=0.7):
        if self._stop_event.is_set():
            return

        self.data_utility.start_background_collection()

        start_time = time.time()
        duration_seconds = duration_minutes * 60

        while (
                not self._stop_event.is_set()
                and (time.time() - start_time) < duration_seconds
        ):
            time.sleep(1)

        if self._stop_event.is_set():
            return

        self.data_utility.stop_background_collection()
        self.data_utility.save_mouse_raw_csv(filename="mouse_raw.csv")

        df = self.data_utility.mouse_data_collector.get_data()

        if self._stop_event.is_set():
            return

        self.status_update.emit("● Authenticating")

        accepted, mean_score = self._authenticate(df)

        if self._stop_event.is_set():
            return

        self.auth_result.emit(bool(accepted), float(mean_score))

    # =========================
    # Authentication Logic
    # =========================
    def _authenticate(self, df):
        if df.empty:
            return False, 0.0

        signal = self._compute_velocity(df)
        mu_s = signal.mean(axis=0)
        sigma_s = signal.std(axis=0) + 1e-6
        signal = (signal - mu_s) / sigma_s

        windows = self._create_windows(
            signal,
            self.WINDOW_SIZE,
            self.STEP_SIZE
        )

        if len(windows) == 0:
            return False, 0.0

        scores = self.model.predict(windows, verbose=0).ravel()

        print(
            scores.min(),
            np.percentile(scores, 10),
            scores.mean(),
            np.percentile(scores, 90),
            scores.max()
        )

        self.threshold = float(
            np.load(
                os.path.join(
                    constants.PATH_METRICS,
                    f"{self.username}_mouse_threshold.npy"
                )
            )
        )
        print("[BACKGROUND AUTH MANGER] threshold: ", self.threshold)

        aggregated_score = np.percentile(scores, 20)
        accepted = aggregated_score >= self.threshold

        return accepted, float(aggregated_score)

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
