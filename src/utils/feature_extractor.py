import os
import time
import csv
import json
import pandas as pd
from datetime import datetime
from src.utils.extracted_features import ExtractedFeatures


class FeatureExtractor:

    # Determine project root reliably

    CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
    FEATURES_DIR = os.path.join(PROJECT_ROOT, "extracted_features")

    def __init__(self, username=None, raw_key_data=None, raw_mouse_data=None, rep_counter=None):

        self.key_features = ExtractedFeatures()
        self.mouse_features = ExtractedFeatures()

        self.username = username
        self.raw_key_data = raw_key_data
        self.raw_mouse_data = raw_mouse_data

        self.rep_counter = rep_counter or 0

        self.create_features_directory()

    def create_features_directory(self):
        """Create global features folder inside project root."""
        os.makedirs(self.FEATURES_DIR, exist_ok=True)

    def set_username(self, username: str):
        self.username = username
        print(f"USERNAME IN FEATURE EXTRACTOR SET {username}")

    def extract_key_features(self):
        df = pd.DataFrame([e.to_dict() for e in self.raw_key_data])
        df = df.sort_values('timestamp').reset_index(drop=True)

        features = {}
        press_times = {}
        last_press_key = None
        last_press_time = None
        last_release_key = None
        last_release_time = None
        shift_pressed = False

        shift_keys = ['Shift', 'Shift_R', '\uE008', '16777248']

        for _, row in df.iterrows():
            raw_key = row['key'] or row['keysym']
            event_type = row['event_type']
            timestamp = row['timestamp']

            if raw_key in shift_keys:
                shift_pressed = (event_type == 'press')
                continue

            key = str(raw_key)
            if len(key) == 1:
                key = key.lower()
            if shift_pressed:
                key = f"Shift.{key}"

            if event_type == 'press':
                if last_press_key is not None:
                    features[f"DD.{last_press_key}.{key}"] = timestamp - last_press_time

                if last_release_key is not None:
                    features[f"UD.{last_release_key}.{key}"] = timestamp - last_release_time

                press_times[key] = timestamp
                last_press_key = key
                last_press_time = timestamp

            elif event_type == 'release':
                if key in press_times:
                    features[f"H.{key}"] = timestamp - press_times[key]
                    del press_times[key]

                last_release_key = key
                last_release_time = timestamp

        metadata = {
            "subject": self.username,
            "sessionIndex": 1,
            "generated": 0,
            "rep": self.rep_counter
        }

        self.key_features.update(metadata, features)

    def save_key_features_csv(self, filename=None, append=False):
        """Always save under project_root/extracted_features/<username>"""
        if not self.username:
            print("save_features_csv: username not set")
            return False

        if not self.key_features:
            print("save_features_csv: no features to save")
            return False

        # user folder inside global features folder
        user_dir = os.path.join(self.FEATURES_DIR, self.username)
        os.makedirs(user_dir, exist_ok=True)

        # filename rules
        if filename is None:
            filename = f"{time.time()}_{self.username}_features.csv"

        if not os.path.isabs(filename):
            filename = os.path.join(user_dir, filename)

        key_features_list = list(self.key_features.get_keys())
        fieldnames = key_features_list

        mode = 'a' if append and os.path.exists(filename) else 'w'

        try:
            with open(filename, mode, newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if mode == 'w':
                    writer.writeheader()
                row = {
                    k: json.dumps(self.key_features.get_key(k), ensure_ascii=False)
                    if isinstance(self.key_features.get_key(k), (list, dict))
                    else self.key_features.get_key(k)
                    for k in key_features_list
                }
                writer.writerow(row)

            print(f"Features saved to: {filename}")
            return True

        except Exception as e:
            print(f"Failed to save features CSV: {e}")
            return False


    def load_csv_key_features(self, filename: str) -> str:
        username = self.key_features.load_csv_features_all_rows(filename)
        return username

    def preprocess_features_for_synthesis(self):
        return self.key_features.all_features

    def clear_data(self):
        self.key_features.clear()
        self.mouse_features.clear()

    def clear_for_next_rep(self, fault=False):
        self.key_features.clear()
        self.mouse_features.clear()
        self.rep_counter += 1
