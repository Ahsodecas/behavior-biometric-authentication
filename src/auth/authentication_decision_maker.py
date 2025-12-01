# src/auth/authenticator.py

import os
import torch
import numpy as np
from datasets.test import TripletSNN, CMUDatasetTriplet
import pandas as pd


class AuthenticationDecisionMaker:
    """
    Handles ALL authentication logic (model loading, password check,
    feature normalization, embedding, distance check).
    The GUI should only call authenticator.authenticate().
    """

    def __init__(self, password_fixed=".tie5Roanl", threshold=0.4):
        self.password_fixed = password_fixed
        self.threshold = threshold

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.scaler = None
        self.ref_sample = None
        self.feature_cols = None

    # ---------------------------------------------------------
    # ------------ LOAD MODEL + SCALER + REF SAMPLE ----------
    # ---------------------------------------------------------
    def load_model(self, ckpt_path):
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        # Load dataset (to obtain scaler + feature columns)
        tmp_dataset = CMUDatasetTriplet("datasets/ksenia_training_2.csv")
        input_dim = tmp_dataset.X.shape[1]

        self.scaler = tmp_dataset.scaler
        self.feature_cols = tmp_dataset.feature_cols

        if tmp_dataset.X.shape[0] < 40:
            raise ValueError("Need at least 40 samples to compute reference embedding.")

        self.ref_sample = tmp_dataset.X[:40].mean(axis=0).astype(np.float32)

        # Build model
        model = TripletSNN(input_dim=input_dim)
        ckpt = torch.load(ckpt_path, map_location=self.device)
        model.load_state_dict(ckpt["model_state"])
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

        # password check
        if password != self.password_fixed:
            return False, float("inf"), "Incorrect password."

        # Ensure model loaded
        if self.model is None or self.scaler is None or self.feature_cols is None:
            return False, float("inf"), "Model not loaded."

        # Convert collected features -> ordered vector
        raw_list = []

        for col in self.feature_cols:

            if col not in feature_dict:
                return False, float("inf"), f"Missing feature in DataCollector: {col}"

            val = feature_dict[col]

            # Convert to numeric float
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

        # Now convert to float32 numpy vector
        raw_vec = np.array(raw_list, dtype=np.float32)

        # Normalize
        try:
            normalized_vec = self.scaler.transform(raw_vec.reshape(1, -1))[0]
        except Exception as e:
            return False, float("inf"), f"Scaler transform failed: {e}"

        print(normalized_vec)
        dist = self.compute_distance(normalized_vec)

        if dist < self.threshold:
            return True, dist, "Authenticated successfully."
        else:
            return False, dist, "Authentication failed."


