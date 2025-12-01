import os
import time
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime
from src.utils.extracted_features import ExtractedFeatures

class FeatureExtractor:

    features_dir = "extracted_features"

    def __init__(self, username=None, raw_key_data = None, raw_mouse_data = None, rep_counter = None):
        self.key_features = ExtractedFeatures()
        self.mouse_features = ExtractedFeatures()
        self.username = username
        self.raw_key_data = raw_key_data
        self.raw_mouse_data = raw_mouse_data
        self.rep_counter = rep_counter or 0
        self.create_features_directory()

    def create_features_directory(self):
        if not os.path.exists(self.features_dir):
            os.makedirs(self.features_dir)

    def extract_key_features(self):
        """
        Transform raw key events into keystroke dynamics features:
        - H.{key}: hold time (press → release for same key, supports overlaps)
        - DD.{k1}.{k2}: time between consecutive press events
        - UD.{k1}.{k2}: time between release of k1 and press of k2
        """

        df = pd.DataFrame([e.to_dict() for e in self.raw_key_data])
        df = df.sort_values('timestamp').reset_index(drop=True)

        features = {}
        press_times = {}  # {key: timestamp of last press}
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
        # is rep_counter needed here?
        # should session_index be modified anywhere?

        self.key_features.update(metadata, features)

    def save_key_features_csv(self, filename=None, append=False):
        """
        Save extracted features (self.features) to a CSV file.
        During enrollment, all samples go into a single file with append=False.
        """
        if not self.username:
            print("save_features_csv: username not set")
            return False
        if not self.key_features:
            print("save_features_csv: no features to save")
            return False

        user_dir = os.path.join(self.features_dir, self.username)
        os.makedirs(user_dir, exist_ok=True)

        # Use a single enrollment file per user
        if filename is None:
            filename = os.path.join(user_dir, f"{time.time()}_{self.username}_features.csv")
        elif not os.path.isabs(filename):
            filename = os.path.join(user_dir, filename)

        # Sort keys to ensure consistent CSV column order
        key_features_list = list(self.key_features.get_keys())
        print(f"key_features_list: {key_features_list}")
        fieldnames = key_features_list

        mode = 'a' if append and os.path.exists(filename) else 'w'
        try:
            with open(filename, mode, newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                # Only write header if creating new file
                if mode == 'w':
                    writer.writeheader()
                row = {}
                for k in key_features_list:
                    v = self.key_features.get_key(k)
                    print("key: " + k + " value: " + str(v))
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

    def prepocess_features_for_synthesis(self):
        return self.key_features.all_features


    def clear_data(self):
        """Clear extracted features for the next session."""
        self.key_features.clear()
        self.mouse_features.clear()

    # FIX add fault functionality
    def clear_for_next_rep(self, fault=False):
        """Clear extracted features for the next rep."""
        self.key_features.clear()
        self.mouse_features.clear()
        self.rep_counter += 1