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

    def __init__(self, enrollment_csv: str, dsl_dataset_csv: str, username: str, output_csv: str, synth_reps=200):
        self.synth_reps = synth_reps
        self.data_utility = DataUtility()
        self.enrollment_csv = enrollment_csv
        self.dsl_dataset_csv = dsl_dataset_csv
        self.username = username
        self.output_csv = output_csv

    # -------------------------------------------------------
    # Safe CSV loader
    # -------------------------------------------------------
    def load_csv(self, path):
        try:
            if not os.path.exists(path):
                raise FileNotFoundError(f"CSV not found: {path}")

            df = pd.read_csv(path)
            print(f"[DataPreprocessor] Loaded CSV: {path} (rows={len(df)})")
            return df

        except Exception as e:
            print(f"[ERROR] Failed to load CSV '{path}': {e}")
            return None

    # -------------------------------------------------------
    # Generate synthetic using your existing pipeline
    # -------------------------------------------------------
    def generate_synthetic(self, enrollment_csv, username):
        try:
            df = self.load_csv(enrollment_csv)
            if df is None or df.empty:
                print("[ERROR] Enrollment CSV is empty or missing. Cannot generate synthetic data.")
                return pd.DataFrame()

            synthetic_file = "synthetic_temp.csv"
            try:
                if os.path.exists(synthetic_file):
                    os.remove(synthetic_file)
            except Exception as e:
                print(f"[WARNING] Failed to delete old synthetic file '{synthetic_file}': {e}")

            username = self.data_utility.feature_extractor.key_features.load_csv_features_all_rows(self.enrollment_csv)

            # Generate synthetic rows
            try:
                self.data_utility.generate_synthetic_features(
                    username=username,
                    filename=synthetic_file,
                    repetitions=self.synth_reps
                )
            except Exception as e:
                print(f"[ERROR] Synthetic generation failed: {e}")
                return pd.DataFrame()

            # Reload generated synthetic file
            synth_df = self.load_csv(f"extracted_features/{username}/{synthetic_file}")
            if synth_df is None:
                return pd.DataFrame()
            return synth_df

        except Exception as e:
            print(f"[ERROR] Unexpected failure in synthetic generation: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------
    # Main pipeline
    # -------------------------------------------------------
    def build_training_csv(self):
        try:
            print("[DataProcessor] Loading real enrollment samples...")
            enroll_df = self.load_csv(self.enrollment_csv)
            if enroll_df is None:
                print("[FATAL] No enrollment dataset. Cannot proceed.")
                return None
            print("[DataProcessor] Generating synthetic samples via DataUtility...")
            synth_df = self.generate_synthetic(self.enrollment_csv, self.username)
            if synth_df is None:
                synth_df = pd.DataFrame()

            print("[DataProcessor] Loading DSL baseline dataset...")
            dsl_df = self.load_csv(self.dsl_dataset_csv)
            if dsl_df is None:
                print("[FATAL] DSL dataset missing. Cannot build training CSV.")
                return None

            print("[DataProcessor] Combining datasets...")
            try:
                combined = pd.concat([dsl_df, enroll_df, synth_df], ignore_index=True)
            except Exception as e:
                print(f"[ERROR] Failed to concat dataframes: {e}")
                return None

            print(f"[DataProcessor] Writing final training CSV: {self.output_csv}")
            try:
                combined.to_csv(self.output_csv, index=False)
            except Exception as e:
                print(f"[ERROR] Failed to save output CSV '{self.output_csv}': {e}")
                return None

            print("[DataProcessor] DONE.")
            return combined

        except Exception as e:
            print(f"[FATAL] Unexpected error in build_training_csv(): {e}")
            return None
