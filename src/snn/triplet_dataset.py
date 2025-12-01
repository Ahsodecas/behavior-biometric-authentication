import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler


class CMUDatasetTriplet(Dataset):
    """
    CMU Triplet dataset that produces (anchor, positive, negative) pairs on-the-fly.
    """

    def __init__(self, csv_path, scaler=None):
        self.df = pd.read_csv(csv_path)

        cols = list(self.df.columns)
        feat_start = cols.index("rep") + 1 if "rep" in cols else 3

        self.meta_cols = cols[:feat_start]
        self.feature_cols = cols[feat_start:]

        self.df["subject"] = self.df["subject"].astype(str)

        X = self.df[self.feature_cols].astype(float).values

        self.scaler = scaler if scaler else StandardScaler()
        X = self.scaler.fit_transform(X) if scaler is None else self.scaler.transform(X)

        self.X = X.astype(np.float32)
        self.y = self.df["subject"].values

        self.by_subject = {}
        for idx, s in enumerate(self.y):
            self.by_subject.setdefault(s, []).append(idx)

        self.subjects = list(self.by_subject.keys())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # Oversample subject "ksenia"
        if random.random() < 0.3 and "ksenia" in self.by_subject:
            idx = random.choice(self.by_subject["ksenia"])

        anchor = self.X[idx]
        subj = self.y[idx]

        # Positive
        pos_list = self.by_subject[subj]
        pos_idx = idx if len(pos_list) == 1 else random.choice(pos_list)
        positive = self.X[pos_idx]

        # Negative
        neg_candidates = (
            [s for s in self.subjects if s != subj and s != "ksenia"]
            if subj == "ksenia"
            else [s for s in self.subjects if s != subj]
        )
        neg_sub = random.choice(neg_candidates)
        neg_idx = random.choice(self.by_subject[neg_sub])
        negative = self.X[neg_idx]

        # Convert to tensors + seq dimension
        anchor = torch.tensor(anchor).unsqueeze(0)
        positive = torch.tensor(positive).unsqueeze(0)
        negative = torch.tensor(negative).unsqueeze(0)
        lengths = torch.tensor([1], dtype=torch.long)

        return anchor, positive, negative, lengths


def collate_triplet(batch):
    A = torch.cat([b[0] for b in batch], dim=0)
    P = torch.cat([b[1] for b in batch], dim=0)
    N = torch.cat([b[2] for b in batch], dim=0)
    L = torch.cat([b[3] for b in batch], dim=0)
    return A, P, N, L
