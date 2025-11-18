import os
import time
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime

class DataCollector:
    def __init__(self, username=None):
        self.data = []                      # List of raw events (press/release)
        self.features = []                  # Extracted features
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
        self.key_press_times = {}

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

        # Record event data
        event_record = {
            'key': key_char or '',
            'keysym': str(key_code),
            'event_type': event_type,
            'timestamp': current_time,
            'session_elapsed_time': round(session_elapsed, 6),
        }

        # Append to your collected data
        self.data.append(event_record)
        self.last_event_time = current_time

    def clear_for_next_rep(self, failed=False):
        """
        Clear only raw data and extracted features between repetitions,
        keeping the rep_counter intact.
        """
        self.data = []
        self.features = {}
        self.session_start_time = time.time()
        self.last_event_time = self.session_start_time
        self.key_press_times = {}
        if not failed:
            self.rep_counter += 1


    def extract_features(self):
        """
        Transform raw key events into keystroke dynamics features:
        - H.{key}: hold time (press → release for same key, supports overlaps)
        - DD.{k1}.{k2}: time between consecutive press events
        - UD.{k1}.{k2}: time between release of k1 and press of k2
        """
        import pandas as pd

        df = pd.DataFrame(self.data)
        df = df.sort_values('timestamp').reset_index(drop=True)

        features = {}
        press_times = {}   # {key: timestamp of last press}
        last_press_key = None
        last_press_time = None
        last_release_key = None
        last_release_time = None
        shift_pressed = False

        shift_keys = ['Shift', 'Shift_R', '\uE008', '16777248']
        org_key = None

        for _, row in df.iterrows():
            key = row['key'] or row['keysym']
            key = str(key).lower()
            event_type = row['event_type']
            timestamp = row['timestamp']

            if key in shift_keys:
                if event_type == 'press':
                    shift_pressed = True
                elif event_type == 'release':
                    shift_pressed = False
                    org_key = None
                continue

            if shift_pressed:
                org_key = key
                key = f"Shift.{key}"

            # --- Handle key press ---
            if event_type == 'press':
                # Down-Down latency: between previous press and this press
                if last_press_key is not None:
                    dd_key = f"DD.{last_press_key}.{key}"
                    features[dd_key] = timestamp - last_press_time

                # Up-Down latency: between previous release and this press
                if last_release_key is not None:
                    ud_key = f"UD.{last_release_key}.{key}"
                    features[ud_key] = timestamp - last_release_time

                # Record this press
                press_times[org_key if org_key is not None else key] = timestamp
                last_press_key = key
                last_press_time = timestamp

            # --- Handle key release ---
            elif event_type == 'release':
                if key in press_times:
                    hold_time = timestamp - press_times[key]
                    h_key = f"H.{key}"
                    features[h_key] = hold_time
                    del press_times[key]

                last_release_key = key
                last_release_time = timestamp

        metadata = {
            "subject": self.username,
            "sessionIndex": 1,
            "rep": self.rep_counter
        }

        self.features = {**metadata, **features}

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

    def save_features_csv(self, filename=None, append=True):
        """
        Save extracted features (self.features) to a CSV file.
        During enrollment, all samples go into a single file with append=True.
        """
        if not self.username:
            print("save_features_csv: username not set")
            return False
        if not self.features:
            print("save_features_csv: no features to save")
            return False

        user_dir = os.path.join(self.data_dir, self.username)
        os.makedirs(user_dir, exist_ok=True)

        # Use a single enrollment file per user
        if filename is None:
            filename = os.path.join(user_dir, f"{time.time()}_{self.username}_features.csv")
        elif not os.path.isabs(filename):
            filename = os.path.join(user_dir, filename)

        # Sort keys to ensure consistent CSV column order
        feature_keys = list(self.features.keys())
        fieldnames = feature_keys

        mode = 'a' if append and os.path.exists(filename) else 'w'
        try:
            with open(filename, mode, newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                # Only write header if creating new file
                if mode == 'w':
                    writer.writeheader()
                row = {}
                for k in feature_keys:
                    v = self.features.get(k)
                    if isinstance(v, (list, dict)):
                        row[k] = json.dumps(v, ensure_ascii=False)
                    else:
                        row[k] = v
                writer.writerow(row)
        except Exception as e:
            print(f"Failed to save features CSV: {e}")
            return False

        print(f"Features saved to {filename} (append={append})")
        return True

    def clear_data(self):
        """Clear collected data for the next session."""
        self.data = []
        self.session_start_time = None
        self.last_event_time = None
        self.key_press_times = {}
