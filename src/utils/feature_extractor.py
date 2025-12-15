import os
import time
import csv
import json
import pandas as pd
from src.utils.extracted_features import ExtractedFeatures
import src.constants as constants


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

        self.feature_cols = self.generate_required_features()

    def generate_required_features(self):
        """
        Generates names in this order:
        H.k1, DD.k1.k2, UD.k1.k2, H.k2, DD.k2.k3, UD.k2.k3, ..., H.last
        Handles Shift: if char is uppercase, treat it as Shift.<lower>
        """
        password = constants.PASSWORD
        keys = []
        for ch in password:
            if ch.isupper():
                keys.append(f"Shift.{ch.lower()}")
            else:
                keys.append(ch)

        features = []

        for i in range(len(keys)):
            k1 = keys[i]

            # Hold feature always comes first
            features.append(f"H.{k1}")

            # For pairs
            if i < len(keys) - 1:
                k2 = keys[i + 1]
                features.append(f"DD.{k1}.{k2}")
                features.append(f"UD.{k1}.{k2}")

        return features

    def create_features_directory(self):
        """Create global features folder inside project root."""
        os.makedirs(self.FEATURES_DIR, exist_ok=True)

    def set_username(self, username: str):
        self.username = username
        # print(f"USERNAME IN FEATURE EXTRACTOR SET {username}")

    def extract_key_features(self):
        """
        Robust extractor: builds token list from password, finds press/release times
        for each token (handles Shift.X composite), then computes H, DD, UD.
        """
        # --- prepare dataframe of events ---
        df = pd.DataFrame([e.to_dict() for e in self.raw_key_data])
        df = df.sort_values('timestamp').reset_index(drop=True)

        # Normalize raw rows into a list of dicts for easy scanning
        events = []
        for _, r in df.iterrows():
            events.append({
                'raw_key': str(r['key'] or r.get('keysym')),
                'type': r['event_type'],
                'ts': float(r['timestamp'])
            })

        # --- build token list from password (automatic) ---
        password = constants.PASSWORD  # keep this here or read from config
        tokens = []
        for ch in password:
            if ch.isupper():
                tokens.append(f"Shift.{ch.lower()}")
            else:
                tokens.append(ch)

        # --- helpers to find next press/release of a character starting from index ---
        shift_names = {'Shift', 'Shift_R', '16777248', '\uE008'}

        def find_press_release_for_token(start_idx, token):
            """
            Returns (press_ts, release_ts, next_index_to_continue_search)
            If not found returns (None, None, len(events))
            """
            n = len(events)
            # Composite token?
            if token.startswith("Shift."):
                letter = token.split(".", 1)[1]
                # Need to find sequence: a Shift press, then letter press while shift held,
                # then letter release and shift release (order may vary).
                # Strategy: scan for a shift press; from that shift-press index look for the next press of letter.
                idx = start_idx
                while idx < n:
                    # look for shift press
                    if events[idx]['raw_key'] in shift_names and events[idx]['type'] == 'press':
                        shift_press_ts = events[idx]['ts']
                        # track shift release time if found later
                        shift_release_ts = None
                        # find letter press after shift_press
                        j = idx + 1
                        while j < n:
                            # if shift released before letter press, this shift block is not valid
                            if events[j]['raw_key'] in shift_names and events[j]['type'] == 'release':
                                shift_release_ts = events[j]['ts']
                                # if release occurs before letter press, break to outer to search next shift press
                                # but we still continue to check for letter later if release occurs after letter
                                # so don't break here; we'll check timing later
                            if events[j]['type'] == 'press' and str(events[j]['raw_key']).lower() == letter.lower():
                                letter_press_ts = events[j]['ts']
                                # find letter release (after letter press)
                                k = j + 1
                                letter_release_ts = None
                                shift_release_after_letter = None
                                while k < n:
                                    if events[k]['type'] == 'release' and str(
                                            events[k]['raw_key']).lower() == letter.lower():
                                        letter_release_ts = events[k]['ts']
                                        # do not break yet: we still want shift_release time if it occurs after
                                    if events[k]['raw_key'] in shift_names and events[k]['type'] == 'release':
                                        shift_release_after_letter = events[k]['ts']
                                    if letter_release_ts is not None and (
                                            shift_release_after_letter is not None or k >= j + 5):
                                        # we have both or have walked enough — stop
                                        break
                                    k += 1
                                # compute composite release: max of letter_release_ts and shift_release_after_letter (if present)
                                final_release = None
                                if letter_release_ts is not None and shift_release_after_letter is not None:
                                    final_release = max(letter_release_ts, shift_release_after_letter)
                                elif letter_release_ts is not None:
                                    final_release = letter_release_ts
                                elif shift_release_after_letter is not None:
                                    final_release = shift_release_after_letter
                                else:
                                    # no release info yet: use letter_press_ts as placeholder
                                    final_release = letter_press_ts
                                return letter_press_ts, final_release, j + 1
                            j += 1
                    idx += 1
                return None, None, n

            else:
                # simple token: find next press of that key and its release
                key = token
                idx = start_idx
                while idx < n:
                    if events[idx]['type'] == 'press' and str(events[idx]['raw_key']).lower() == str(key).lower():
                        press_ts = events[idx]['ts']
                        # find corresponding release
                        j = idx + 1
                        release_ts = None
                        while j < n:
                            if events[j]['type'] == 'release' and str(events[j]['raw_key']).lower() == str(key).lower():
                                release_ts = events[j]['ts']
                                break
                            j += 1
                        if release_ts is None:
                            release_ts = press_ts  # fallback
                        return press_ts, release_ts, idx + 1
                    idx += 1
                return None, None, n

        # --- find press/release times for each token in order ---
        token_press = {}
        token_release = {}
        search_index = 0
        for tok in tokens:
            p, r, search_index = find_press_release_for_token(search_index, tok)
            token_press[tok] = p
            token_release[tok] = r
            # continue searching from where we left off for the next token

        # --- compute features H, DD, UD ---
        features = {}
        # Holds: H.{token}
        for tok in tokens:
            p = token_press.get(tok)
            r = token_release.get(tok)
            if p is not None and r is not None:
                features[f"H.{tok}"] = r - p

        # Pairs for DD and UD according to token adjacent order
        for i in range(len(tokens) - 1):
            a = tokens[i]
            b = tokens[i + 1]
            pa = token_press.get(a)
            ra = token_release.get(a)
            pb = token_press.get(b)

            # DD.a.b = press(b) - press(a) if both present
            if pa is not None and pb is not None:
                features[f"DD.{a}.{b}"] = pb - pa

            # UD.a.b = press(b) - release(a) only if both present
            if ra is not None and pb is not None:
                # If press(b) occurs earlier than release(a), UD doesn't make sense; leave missing
                if pb >= ra:
                    features[f"UD.{a}.{b}"] = pb - ra

        # --- metadata and save to self.key_features ---
        metadata = {
            "subject": self.username,
            "sessionIndex": 1,
            "generated": 0,
            "rep": self.rep_counter
        }

        self.key_features.update(metadata, features)

    def save_key_features_csv(self, filename=None, append=False):
        """Save metadata + fixed-order keystroke features.
           Missing features are written as 0.
        """
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

        # --------------------------------------------
        # FIXED COLUMN ORDER
        # --------------------------------------------
        metadata_keys = ["subject", "sessionIndex", "generated", "rep"]
        feature_keys = self.feature_cols
        fieldnames = metadata_keys + feature_keys

        mode = 'a' if append and os.path.exists(filename) else 'w'

        try:
            with open(filename, mode, newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)

                if mode == 'w':
                    writer.writeheader()

                # metadata section
                meta = self.key_features.metadata

                # full feature dict
                extracted = self.key_features.data

                # row creation: fill in metadata + features (missing → 0)
                row = {}

                # write metadata
                for m in metadata_keys:
                    row[m] = meta.get(m, "")

                # write ordered features
                for feat in feature_keys:
                    val = extracted.get(feat, 0)
                    if isinstance(val, (list, dict)):
                        row[feat] = json.dumps(val, ensure_ascii=False)
                    else:
                        row[feat] = val

                writer.writerow(row)

            # print(f"Features saved to: {filename}")
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

    def clear_for_next_rep(self, failed=False):
        self.key_features.clear()
        self.mouse_features.clear()
        if not failed:
            self.rep_counter += 1
