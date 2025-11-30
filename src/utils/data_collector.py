import os
import time
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.key_stroke_event import KeyStrokeEvent


class DataCollector:
    def __init__(self, username=None):
        self.data = []
        self.session_start_time = None
        self.last_event_time = None         # For inter-event interval
        self.username = username
        self.data_dir = "collected_data"
        self._create_data_directory()
        self.rep_counter = 0

    def _create_data_directory(self):
        """Create the data directory if it doesn't exist."""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

    def start_session(self):
        """Initialize a new data collection session."""
        self.data = []
        self.session_start_time = time.time()
        self.last_event_time = self.session_start_time

    def collect_key_event(self, qt_event, event_type):
        """
        Collect a raw key event from PyQt5 during password typing.

        Parameters:
        - qt_event: QKeyEvent object
        - event_type: 'press' or 'release'
        """
        current_time = time.time()
        session_elapsed = current_time - self.session_start_time

        key_char = qt_event.text() if qt_event.text() else None
        key_code = qt_event.key()  # Numeric Qt key code (e.g. Qt.Key_A)

        self.data.append(KeyStrokeEvent(key_code, event_type, current_time, session_elapsed, key_char))
        self.last_event_time = current_time

    def clear_for_next_rep(self, failed=False):
        """
        Clear only raw data and extracted features between repetitions,
        keeping the rep_counter intact.
        """
        self.data = []
        self.session_start_time = time.time()
        self.last_event_time = self.session_start_time
        if not failed:
            self.rep_counter += 1

    def save_key_raw_csv(self, filename=None):
        """
        Save the raw keystroke event data to a CSV file.
        """
        try:
            if not self.data:
                raise ValueError("No raw keystroke data to save.")


            # Directory structure
            user = self.username or "unknown_user"
            user_dir = os.path.join(self.data_dir, user)
            os.makedirs(user_dir, exist_ok=True)

            if filename is None:
                filename = os.path.join(user_dir, "rep_{self.rep_counter}.csv")
            else:
                filename = os.path.join(user_dir, filename)

            with open(filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                writer.writerow(["key", "keysym", "event_type", "timestamp", "session_elapsed_time"])

                for evt in self.data:
                    row = evt.to_dict()
                    writer.writerow([
                        row["key"],
                        row["keysym"],
                        row["event_type"],
                        row["timestamp"],
                        row["session_elapsed_time"]
                    ])

            print(f"Raw data saved to {filename}")
            return filename

        except Exception as e:
            print(f"[ERROR] Failed to save raw keystroke data: {e}")
            return None

