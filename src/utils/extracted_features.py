import time
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime

class ExtractedFeatures:
    def __init__(self):
        self.features = {}
        self.all_features = []
        self.metadata = {}
        self.data ={}

    def update(self, metadata, features):
        self.metadata = metadata
        self.features = features
        self.all_features.extend(list(features.items()))
        self.data = {**self.metadata, **self.features}

    def get_keys(self):
        return self.data.keys()

    def get_key(self, key):
        return self.data.get(key)

    def get_features(self):
        return self.features

    def clear(self):
        self.features = {}
        self.metadata = {}
        self.data = {}
        self.all_features = []

    def load_csv_features(self, file_path):
        """
        Load features from a CSV file, convert them to floats/timestamps,
        and return an ExtractedFeatures object.
        """
        metadata = {}
        features = {}

        try:
            df = pd.read_csv(file_path)
            # If multiple rows, take the first one (or iterate if needed)
            row = df.iloc[0].to_dict()

            for key, val in row.items():
                # Handle metadata columns
                if key in ["subject", "sessionIndex", "rep", "generated"]:
                    try:
                        if key == "subject":
                            username = val
                        else:
                            val = int(val)
                    except Exception:
                        pass
                    metadata[key] = val
                else:
                    # Convert feature to numeric float
                    try:
                        num = float(val)
                    except Exception:
                        try:
                            if isinstance(val, str):
                                num = float(pd.to_datetime(val).timestamp())
                            else:
                                raise ValueError(f"Unsupported type for {key}: {val}")
                        except Exception as e:
                            print(f"Could not convert feature {key}: {e}")
                            num = np.nan  # fallback

                    features[key] = num

            self.update(metadata, features)
            return username

        except Exception as e:
            print(f"Failed to load CSV features: {e}")
            return None
