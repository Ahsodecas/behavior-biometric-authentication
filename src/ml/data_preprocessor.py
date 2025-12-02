import os
import csv
import numpy as np
import pandas as pd

from src.utils.data_utility import DataUtility

class DataPreprocessor:
    """
    Creates a unified training dataset by:
      - Loading enrollment samples
      - Generating synthetic versions of them
      - Loading the existing DSL dataset
      - Combining everything into a single CSV for model training
    """

    def __init__(self, enrollment_csv: str, dsl_dataset_csv: str, username: str, output_csv: str, synth_reps=10):
        self.synth_reps = synth_reps
        self.data_utility = DataUtility()
        self.enrollment_csv = enrollment_csv
        self.dsl_dataset_csv = dsl_dataset_csv
        self.username = username
        self.output_csv = output_csv

    # -------------------------------------------------------
    # Load CSV
    # -------------------------------------------------------
    def load_csv(self, path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        return pd.read_csv(path)

    # -------------------------------------------------------
    # Generate synthetic using your existing pipeline
    # -------------------------------------------------------
    def generate_synthetic(self, enrollment_csv, username):
        """
        Uses your existing DataUtility → SyntheticFeaturesGenerator to produce
        synthetic rows saved via FeatureExtractor.save_key_features_csv().
        """

        # Load real enrollment rows
        df = self.load_csv(enrollment_csv)

        # We now load these real rows into FeatureExtractor one by one
        synthetic_file = "synthetic_temp.csv"
        if os.path.exists(synthetic_file):
            os.remove(synthetic_file)

        for _, row in df.iterrows():
            # Feed features into feature extractor
            self.data_utility.feature_extractor.load_row_into_feature_extractor(row)
            # The method above must exist (one small helper added in FeatureExtractor)
            # If not, I will generate it for you.

        # Now DataUtility generates synthetic samples and appends to synthetic_file
        self.data_utility.generate_synthetic_features(
            username=username,
            filename=synthetic_file,
            repetitions=self.synth_reps
        )

        return self.load_csv(synthetic_file)

    # -------------------------------------------------------
    # Main pipeline
    # -------------------------------------------------------
    def build_training_csv(self):

        print("[DataProcessor] Loading real enrollment samples...")
        enroll_df = self.load_csv(self.enrollment_csv)

        print("[DataProcessor] Generating synthetic samples via DataUtility...")
        synth_df = self.generate_synthetic(self.enrollment_csv, self.username)

        print("[DataProcessor] Loading DSL baseline dataset...")
        dsl_df = self.load_csv(self.dsl_dataset_csv)

        print("[DataProcessor] Combining datasets...")
        combined = pd.concat([dsl_df, enroll_df, synth_df], ignore_index=True)

        print(f"[DataProcessor] Writing final training CSV: {self.output_csv}")
        combined.to_csv(self.output_csv, index=False)

        print("[DataProcessor] DONE.")
        return combined

