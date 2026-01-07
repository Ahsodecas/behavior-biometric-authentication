# src/auth/authenticator.py

import os
import torch
import numpy as np
from src.ml.snn_model import TripletSNN
from src.ml.triplet_dataset import CMUDatasetTriplet
import pandas as pd

import src.constants as constants
from src.utils.user_management_utility import UserManagementUtility

class AuthenticationDecisionMaker:
    """
    Handles authentication logic (model loading, password check,
    feature normalization, embedding, distance check).
    """

    def __init__(self, username=None, threshold=0.4):
        self.username = username
        self.threshold = threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.ref_sample = None
        self.feature_cols = None
        self.user_management_utility = UserManagementUtility()

    # ---------------------------------------------------------
    # ------------ LOAD MODEL + SCALER + REF SAMPLE ----------
    # ---------------------------------------------------------
    def load_model(self, ckpt_path, username, training_csv):
        # -------------------------------
        # Checkpoint file must exist
        # -------------------------------
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # -------------------------------
        # Load TRAINING dataset to recover
        #   - scaler
        #   - feature columns
        # -------------------------------
        training_csv_path = os.path.join(constants.PATH_DATASETS, training_csv)
        if not os.path.exists(training_csv_path):
            raise FileNotFoundError(f"Training CSV not found: {training_csv}")

        tmp_dataset = CMUDatasetTriplet(training_csv_path)

        input_dim = tmp_dataset.X.shape[1]

        self.scaler = tmp_dataset.scaler
        self.feature_cols = tmp_dataset.feature_cols

        # -------------------------------
        # Load CSV again (unscaled) so we
        # can filter by subject/generated
        # -------------------------------

        df = pd.read_csv(training_csv_path)

        ref_df = df[(df["subject"] == username) & (df["generated"] == 0)]

        if ref_df.empty:
            raise ValueError(
                f"No reference samples found: need rows where subject='{username}' and generated=0."
            )

        # -------------------------------
        # Extract raw feature matrix
        # -------------------------------
        try:
            ref_matrix_raw = ref_df[self.feature_cols].to_numpy(dtype=np.float32)
        except KeyError as e:
            raise KeyError(
                f"Training CSV is missing feature column required by model: {e}"
            )

        # -------------------------------
        # Normalize using SAME scaler
        # -------------------------------
        ref_matrix_norm = self.scaler.transform(ref_matrix_raw)

        self.ref_sample = ref_matrix_norm.mean(axis=0).astype(np.float32)

        # -------------------------------
        # Load Triplet network
        # -------------------------------
        model = TripletSNN(input_dim=input_dim)
        ckpt = torch.load(ckpt_path, map_location=self.device)

        if "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            state = ckpt

        model.load_state_dict(state)
        model.to(self.device)
        model.eval()

        self.model = model
        print("Authenticator: Model loaded successfully.")

    # ---------------------------------------------------------
    # ---------------- EMBEDDING UTILITIES --------------------
    # ---------------------------------------------------------
    def embed(self, sample_vector: np.ndarray) -> np.ndarray:
        """
        sample_vector: 1D numpy array (already normalized)
        """
        arr = np.asarray(sample_vector).reshape(-1).astype(np.float32)
        x = torch.tensor(arr).float().unsqueeze(0).unsqueeze(0)
        lengths = torch.tensor([1], dtype=torch.long)

        with torch.no_grad():
            emb = self.model.subnet(x.to(self.device), lengths.to(self.device))

        return emb.cpu().numpy()[0]

    def compute_distance(self, sample_normalized):
        ref_emb = self.embed(self.ref_sample)
        sample_emb = self.embed(sample_normalized)
        return np.linalg.norm(sample_emb - ref_emb)

    # ---------------------------------------------------------
    # ----------------- AUTHENTICATION API --------------------
    # ---------------------------------------------------------
    def authenticate(self, username: str, password: str, feature_dict: dict):
        """
        GUI calls THIS function.
        Returns: (success: bool, distance: float, message: str)
        """

        if not self.user_management_utility.verify_user(username, password):
            return False, float("inf"), "Incorrect password."

        if self.model is None or self.scaler is None or self.feature_cols is None:
            return False, float("inf"), "Model not loaded."

        raw_list = []

        for col in self.feature_cols:

            if col not in feature_dict:
                val = 0
            else:
                val = feature_dict[col]
            try:
                # Case 1: numeric types -> convert directly
                num = float(val)

            except Exception:
                # Case 2: strings such as timestamps -> try to convert via datetime
                try:
                    if isinstance(val, str):
                        num = float(pd.to_datetime(val).timestamp())
                    else:
                        return False, float("inf"), f"Unsupported feature type for {col}: {val}"
                except Exception:
                    return False, float("inf"), f"Could not convert feature {col}: {val}"

            raw_list.append(num)

        raw_vec = np.array(raw_list, dtype=np.float32)

        try:
            normalized_vec = self.scaler.transform(raw_vec.reshape(1, -1))[0]
        except Exception as e:
            return False, float("inf"), f"Scaler transform failed: {e}"

        dist = self.compute_distance(normalized_vec)

        if dist < self.threshold:
            return True, dist, "Authenticated successfully."
        else:
            return False, dist, "Authentication failed."


