# =====================================================================
#  mouse_model_trainer.py
# =====================================================================

import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Dense, Dropout, Flatten
from keras.optimizers import Adam
import src.constants as constants

# =====================================================================
#  MouseModelTrainer Class
# =====================================================================
class MouseModelTrainer:
    def __init__(
        self,
        enrollment_csv: str,
        username: str,
        dataset_root: str = os.path.join(constants.PATH_DATASETS, "sapimouse"),
        out_dir: str = constants.PATH_MODELS,
        window_size: int = 128,
        step_size: int = 64,
        test_size: float = 0.2,
        random_state: int = 42,
        epochs: int = 15,
        batch_size: int = 32,
        lr: float = 1e-3,
    ):
        self.username = username
        self.enrollment_csv = enrollment_csv
        self.dataset_root = dataset_root
        self.window_size = window_size
        self.out_dir = out_dir
        self.window_size = window_size
        self.step_size = step_size
        self.test_size = test_size
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr

        os.makedirs(self.out_dir, exist_ok=True)
        self.model = None

    # -----------------------
    #  Load single CSV
    # -----------------------
    @staticmethod
    def load_csv(file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        df = pd.read_csv(file_path)
        return df

    # -----------------------
    #  Load all CSVs from folder
    # -----------------------
    @staticmethod
    def load_user_csvs(user_path):
        files = glob.glob(os.path.join(user_path, "*.csv"))
        dfs = []
        for f in files:
            dfs.append(pd.read_csv(f))
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    # -----------------------
    #  Compute mouse velocity
    # -----------------------
    @staticmethod
    def compute_velocity(df):
        df = df.copy()
        col = "client timestamp" if "client timestamp" in df.columns else "timestamp"
        df["timestamp"] = pd.to_numeric(df[col], errors="coerce")

        df["dx"] = df["x"].diff()
        df["dy"] = df["y"].diff()
        df["dt"] = df["timestamp"].diff()

        df = df[df["dt"] > 0]
        df["vx"] = df["dx"] / df["dt"]
        df["vy"] = df["dy"] / df["dt"]

        return df[["vx", "vy"]].dropna().values

    # -----------------------
    #  Create sliding windows
    # -----------------------
    def create_windows(self, signal):
        windows = []
        for start in range(0, len(signal) - self.window_size, self.step_size):
            windows.append(signal[start:start + self.window_size])
        return np.array(windows)

    # -----------------------
    #  Prepare dataset (positive + negative)
    # -----------------------
    def prepare_dataset(self):
        # --- Positive samples from enrollment CSV ---
        df_pos = self.load_csv(self.enrollment_csv)
        signal_pos = self.compute_velocity(df_pos)
        mu_pos = signal_pos.mean(axis=0)
        sigma_pos = signal_pos.std(axis=0) + 1e-6
        signal_pos = (signal_pos - mu_pos) / sigma_pos
        X_pos = self.create_windows(signal_pos)

        # --- Negative samples from other users in dataset ---
        X_neg = []
        for user in os.listdir(self.dataset_root):
            user_path = os.path.join(self.dataset_root, user)
            if not os.path.isdir(user_path):
                continue
            # Skip enrollment CSV (assume it belongs to target user)
            if os.path.abspath(user_path) == os.path.abspath(os.path.dirname(self.enrollment_csv)):
                continue
            df_neg = self.load_user_csvs(user_path)
            if df_neg.empty:
                continue
            signal_neg = self.compute_velocity(df_neg)

            mu_neg = signal_neg.mean(axis=0)
            sigma_neg = signal_neg.std(axis=0) + 1e-6
            signal_neg = (signal_neg - mu_neg) / sigma_neg

            windows = self.create_windows(signal_neg)
            X_neg.append(windows)

        X_neg = np.vstack(X_neg) if X_neg else np.zeros_like(X_pos)

        # Balance classes
        min_samples = min(len(X_pos), len(X_neg))
        X_pos = X_pos[:min_samples]
        X_neg = X_neg[:min_samples]

        X = np.vstack([X_pos, X_neg])
        y = np.array([1]*len(X_pos) + [0]*len(X_neg))

        # Train/test split
        return train_test_split(
            X, y,
            test_size=self.test_size,
            stratify=y,
            random_state=self.random_state
        )

    # -----------------------
    #  Build CNN model
    # -----------------------
    def build_model(self):
        self.model = Sequential([
            Conv1D(64, kernel_size=5, activation="relu", input_shape=(self.window_size, 2)),
            MaxPooling1D(2),
            Conv1D(128, kernel_size=5, activation="relu"),
            MaxPooling1D(2),
            Flatten(),
            Dense(128, activation="relu"),
            Dropout(0.5),
            Dense(1, activation="sigmoid")
        ])
        self.model.compile(
            optimizer=Adam(learning_rate=self.lr),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )

    # -----------------------
    #  Train the model
    # -----------------------
    def train(self):
        X_train, X_test, y_train, y_test = self.prepare_dataset()
        self.build_model()
        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.2,
            verbose=1
        )

        # Evaluate
        y_scores = self.model.predict(X_test).ravel()
        auc = roc_auc_score(y_test, y_scores)
        print(f"ROC AUC (authentication performance): {auc:.4f}")

        imposter_scores = y_scores[y_test == 0]
        self.calculate_threshold(imposter_scores)

        # Save model
        model_path = os.path.join(self.out_dir, f"{self.username}_cnn_model.keras")
        self.model.save(model_path)
        print(f"Model saved to {model_path}")
        return model_path, auc

    def calculate_threshold(self, imposter_scores):
        threshold = np.percentile(imposter_scores, 100 * (1 - constants.TARGET_FAR))
        #threshold = 0.9
        threshold_path = os.path.join(
            constants.PATH_METRICS,
            f"{self.username}_mouse_threshold.npy"
        )
        np.save(threshold_path, threshold)


# =====================================================================
#  Example usage
# =====================================================================
if __name__ == "__main__":
    trainer = MouseModelTrainer(
        enrollment_csv="mouse_enrollement.csv",
        dataset_root="sapimouse",
        out_dir="models"
    )
    trainer.train()
