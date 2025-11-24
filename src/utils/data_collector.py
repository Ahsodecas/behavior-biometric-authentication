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

        # Extract key info from QKeyEvent
        key_char = qt_event.text() if qt_event.text() else None
        key_code = qt_event.key()  # Numeric Qt key code (e.g. Qt.Key_A)

        # Append to your collected data
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

    def preprocess_data(self):
        """Return shallow copy of raw events for saving/export."""
        return [dict(e) for e in self.data]

    def save_session_csv(self, append=False):
        """Save both raw keystroke events and extracted features to CSV files."""
        raw_saved = True #self.save_raw_csv(append)
        features_saved = self.save_features_csv(append=append)
        return raw_saved and features_saved

    def save_raw_csv(self, append=False):
        """Save raw keystroke events and features for the current session to a CSV file."""
        if not self.username or not self.data or not self.features:
            return False

        user_dir = os.path.join(self.data_dir, self.username)
        os.makedirs(user_dir, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(user_dir, f"session_{timestamp}_raw.csv")

        fieldnames = [
            'timestamp',
            'session_elapsed_time',
            'inter_event_interval',
            'event_type',
            'key',
            'keysym',
            'hold_time'
        ]

        mode = 'a' if append and os.path.exists(filename) else 'w'
        with open(filename, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # Only write header if creating new file
            if mode == 'w':
                writer.writeheader()
            for e in self.data:
                writer.writerow({
                    'timestamp': datetime.fromtimestamp(e['timestamp']).isoformat(),
                    'session_elapsed_time': e.get('session_elapsed_time'),
                    'inter_event_interval': e.get('inter_event_interval'),
                    'event_type': e.get('event_type'),
                    'key': e.get('key'),
                    'keysym': e.get('keysym'),
                    'hold_time': e.get('hold_time')
                })

        print(f"Raw data saved to {filename} (append={append})")
        return True

    def clear_data(self):
        """Clear collected data for the next session."""
        self.data = []
        self.session_start_time = None
        self.last_event_time = None
