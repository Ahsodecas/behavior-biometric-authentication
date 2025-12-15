import os
import csv
import numpy as np
import pandas as pd
import src.constants as constants
from src.utils.data_utility import DataUtility


class DataPreprocessor:
    """
    Creates a unified training dataset by:
      - Loading enrollment samples
      - Generating synthetic versions of them
      - Generating imposter synthetic samples
      - Combining everything into a single CSV for model training
    """

    def __init__(self, enrollment_csv: str, username: str, output_csv: str, synth_reps=200):
        self.synth_reps = synth_reps
        self.data_utility = DataUtility(username=username)
        self.enrollment_csv = enrollment_csv
        # self.generated_imposter_csv = generated_imposter_csv
        # self.dsl_dataset_csv = dsl_dataset_csv
        self.username = username
        self.output_csv = output_csv

        # Base directory for all generated files
        self.base_dir = os.path.dirname(self.enrollment_csv)

    # -------------------------------------------------------
    # Safe CSV loader
    # -------------------------------------------------------
    def load_csv(self, path):
        if not os.path.exists(path):
            print(f"[ERROR] CSV not found: {path}")
            return None
        try:
            df = pd.read_csv(path)
            print(f"[DataPreprocessor] Loaded CSV: {path} (rows={len(df)})")
            return df

        except Exception as e:
            print(f"[ERROR] Failed to load CSV '{path}': {e}")
            return None

    # -------------------------------------------------------
    # Generate synthetic using existing pipeline
    # -------------------------------------------------------
    def generate_synthetic(self):
        df = self.load_csv(self.enrollment_csv)
        if df is None or df.empty:
            print("[ERROR] Enrollment CSV is empty or missing. Cannot generate synthetic data.")
            return pd.DataFrame()

        synthetic_file = "synthetic_temp.csv"
        synthetic_file_path = os.path.join(constants.PATH_EXTRACTED, f"{self.username}\\{synthetic_file}")
        try:
            if os.path.exists(synthetic_file_path):
                os.remove(synthetic_file_path)
        except Exception as e:
            print(f"[WARNING] Failed to delete old synthetic file '{synthetic_file_path}': {e}")

        print("[INFO] synthetic file path: " + synthetic_file_path)

        self.username = self.data_utility.load_csv_key_features(self.enrollment_csv)

        # Generate synthetic rows
        try:
            self.data_utility.generate_synthetic_features(
                filename=synthetic_file,
                repetitions=self.synth_reps
            )
        except Exception as e:
            print(f"[ERROR] Synthetic generation failed: {e}")
            return pd.DataFrame()

        synth_df = self.load_csv(synthetic_file_path)
        return synth_df if synth_df is not None else pd.DataFrame()

    # -------------------------------------------------------
    # Generate imposter synthetic features
    # -------------------------------------------------------
    def generate_imposter_synthetic(self):
        imposter_file = os.path.join(self.base_dir, "generated_imposter_data.csv")
        try:
            self.data_utility.generate_synthetic_features_imposter_users(
                filename=imposter_file,
                repetitions=self.synth_reps
            )
        except Exception as e:
            print(f"[ERROR] Imposter synthetic generation failed: {e}")
            return pd.DataFrame()

    # -------------------------------------------------------
    # Main pipeline
    # -------------------------------------------------------
    def build_training_csv(self):
        try:
            print("[DataPreprocessor] Loading real enrollment samples...")
            enroll_df = self.load_csv(self.enrollment_csv)
            if enroll_df is None:
                print("[FATAL] No enrollment dataset. Cannot proceed.")
                return None
            print("[DataProcessor] Generating synthetic samples via DataUtility...")
            synth_df = self.generate_synthetic()
            if synth_df is None:
                synth_df = pd.DataFrame()

            print("[DataProcessor] Loading Imposter dataset dataset...")
            generated_imposter_csv = "generated_imposter_data.csv"
            print("[DataPreprocessor] Generating imposter samples...")
            self.data_utility.generate_synthetic_features_imposter_users(
                filename=generated_imposter_csv, repetitions=self.synth_reps)
            dsl_df = self.load_csv(os.path.join(constants.PATH_EXTRACTED, f"{self.username}\\{generated_imposter_csv}"))
            if dsl_df is None:
                print("[FATAL] Imposter dataset missing. Cannot build training CSV.")
                return None

            print("[DataPreprocessor] Combining datasets...")
            try:
                combined = pd.concat([dsl_df, enroll_df, synth_df], ignore_index=True)
            except Exception as e:
                print(f"[ERROR] Failed to concat dataframes: {e}")
                return None

            print(f"[DataPreprocessor] Writing final training CSV: {self.output_csv}")
            try:
                combined.to_csv(self.output_csv, index=False)
            except Exception as e:
                print(f"[ERROR] Failed to save output CSV '{self.output_csv}': {e}")
                return None

            print("[DataPreprocessor] DONE.")
            return combined

        except Exception as e:
            print(f"[FATAL] Unexpected error in build_training_csv(): {e}")
            return None
