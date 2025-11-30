import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from sklearn.model_selection import train_test_split
import numpy as np
from typing import Optional
import pandas as pd
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
#  USE YOUR EXISTING DATASET
# ---------------------------------------------------------
# (pasted exactly as provided)
class CMUDatasetTriplet(Dataset):
    """Your working dataset loader — unchanged."""
    def __init__(self, csv_path: str, scaler: Optional[StandardScaler] = None, preload: bool = True):
        self.df = pd.read_csv(csv_path)
        cols = list(self.df.columns)

        if "rep" in cols:
            feat_start = cols.index("rep") + 1
        else:
            feat_start = 3

        self.meta_cols = cols[:feat_start]
        self.feature_cols = cols[feat_start:]

        self.df['subject'] = self.df['subject'].astype(str)

        X = self.df[self.feature_cols].astype(float).values

        if scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X = self.scaler.transform(X)

        self.X = X.astype(np.float32)
        self.y = self.df['subject'].values

        self.by_subject = {}
        for idx, s in enumerate(self.y):
            self.by_subject.setdefault(s, []).append(idx)

        self.subjects = list(self.by_subject.keys())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        import random
        if random.random() < 0.3:
            if "ksenia" in self.by_subject:
                idx = random.choice(self.by_subject["ksenia"])

        anchor = self.X[idx]
        subj = self.y[idx]

        pos_choices = self.by_subject[subj]

        if subj == "ksenia" and len(pos_choices) > 1:
            pos_idx = random.choice(pos_choices)
        else:
            if len(pos_choices) == 1:
                pos_idx = idx
            else:
                pos_idx = idx
                while pos_idx == idx:
                    pos_idx = random.choice(pos_choices)

        positive = self.X[pos_idx]

        possible_neg_subjs = (
            [s for s in self.subjects if s != subj and s != "ksenia"]
            if subj == "ksenia"
            else [s for s in self.subjects if s != subj]
        )

        neg_subj = random.choice(possible_neg_subjs)
        neg_idx = random.choice(self.by_subject[neg_subj])
        negative = self.X[neg_idx]

        anchor = torch.from_numpy(anchor).unsqueeze(0)
        positive = torch.from_numpy(positive).unsqueeze(0)
        negative = torch.from_numpy(negative).unsqueeze(0)
        lengths = torch.tensor([1], dtype=torch.long)

        return anchor, positive, negative, lengths


# ---------------------------------------------------------
#  LSTM EMBEDDING MODEL (compatible with sequence length = 1)
# ---------------------------------------------------------
class LSTMEmbeddingNet(nn.Module):
    def __init__(self, input_dim, embed_dim=64, hidden_dim=128, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, lengths):
        # x = (batch, seq_len=1, features)
        _, (h, _) = self.lstm(x)  # h: (num_layers, batch, hidden)
        h = h[-1]                 # last layer
        out = self.fc(h)
        return nn.functional.normalize(out, p=2, dim=1)


# ---------------------------------------------------------
#  TRAINING LOOP
# ---------------------------------------------------------
def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0

    for anchor, pos, neg, lengths in loader:
        anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
        lengths = lengths.to(device)

        optimizer.zero_grad()

        a = model(anchor, lengths)
        p = model(pos, lengths)
        n = model(neg, lengths)

        loss = criterion(a, p, n)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# ---------------------------------------------------------
#  BASIC EMBEDDING TESTING (distance-based)
# ---------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    distances_pos = []
    distances_neg = []

    with torch.no_grad():
        for anchor, pos, neg, lengths in loader:
            anchor, pos, neg = anchor.to(device), pos.to(device), neg.to(device)
            lengths = lengths.to(device)

            a = model(anchor, lengths)
            p = model(pos, lengths)
            n = model(neg, lengths)

            d_pos = torch.norm(a - p, dim=1).cpu().numpy()
            d_neg = torch.norm(a - n, dim=1).cpu().numpy()

            distances_pos.extend(d_pos.tolist())
            distances_neg.extend(d_neg.tolist())

    return np.mean(distances_pos), np.mean(distances_neg)

def compute_accuracy(model, loader, device):
    """
    Computes verification accuracy:
      accuracy = % of triplets where d(anchor,positive) < d(anchor,negative)
    """
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for anchor, pos, neg, lengths in loader:
            anchor = anchor.to(device)
            pos    = pos.to(device)
            neg    = neg.to(device)
            lengths = lengths.to(device)

            a = model(anchor, lengths)
            p = model(pos, lengths)
            n = model(neg, lengths)

            d_pos = torch.norm(a - p, dim=1)   # smaller = better
            d_neg = torch.norm(a - n, dim=1)   # larger  = better

            correct += torch.sum(d_pos < d_neg).item()
            total   += len(d_pos)

    return correct / total

# ---------------------------------------------------------
#  FULL TRAINING PIPELINE
# ---------------------------------------------------------
def run_training_pipeline(csv_path):
    print("Loading dataset...")
    full_dataset = CMUDatasetTriplet(csv_path)

    # ---------------------------------------------
    # SUBJECT-WISE SPLIT (recommended)
    # ---------------------------------------------
    subjects = np.array(full_dataset.subjects)

    train_subj, test_subj = train_test_split(subjects, test_size=0.2, random_state=42)
    train_subj, val_subj = train_test_split(train_subj, test_size=0.2, random_state=42)

    def idx_for_subjects(subj_list):
        idxs = []
        for s in subj_list:
            idxs.extend(full_dataset.by_subject[s])
        return idxs

    train_idx = idx_for_subjects(train_subj)
    val_idx = idx_for_subjects(val_subj)
    test_idx = idx_for_subjects(test_subj)

    train_data = Subset(full_dataset, train_idx)
    val_data = Subset(full_dataset, val_idx)
    test_data = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    # ---------------------------------------------
    # MODEL SETUP
    # ---------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"

    input_dim = len(full_dataset.feature_cols)
    model = LSTMEmbeddingNet(input_dim=input_dim).to(device)

    criterion = nn.TripletMarginLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # ---------------------------------------------
    # TRAINING
    # ---------------------------------------------
    for epoch in range(1):
        loss = train(model, train_loader, optimizer, criterion, device)
        dpos, dneg = evaluate(model, val_loader, device)
        acc = compute_accuracy(model, val_loader, device)

        print(f"Training | Loss={loss:.4f} | "
              f"pos={dpos:.3f} | neg={dneg:.3f} | ACC={acc * 100:.2f}%")

    # ---------------------------------------------
    # FINAL TEST METRIC
    # ---------------------------------------------
    dpos, dneg = evaluate(model, test_loader, device)
    test_acc = compute_accuracy(model, test_loader, device)

    print("\n=== TEST RESULTS ===")
    print(f"Mean positive distance: {dpos:.3f}")
    print(f"Mean negative distance: {dneg:.3f}")
    print(f"Verification Accuracy: {test_acc * 100:.2f}%")
    # print("If negative >> positive → good separation!")

    return model


def load_single_sample(csv_path, feature_cols, scaler):
    """
    Load a single-row CSV containing only one keystroke sample.
    Returns a (1, seq_len=1, feature_dim) torch tensor.
    """
    df = pd.read_csv(csv_path)

    # ensure the required columns exist
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required feature columns: {missing}")

    x = df[feature_cols].astype(float).values[0]   # shape: (feature_dim,)
    x = scaler.transform([x])[0]                   # apply training scaler

    # convert to float32 → torch → add sequence dim
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return x

def compare_two_samples(model, csv1, csv2, feature_cols, scaler, device, threshold=0.7):
    """
    Loads two files, embeds them, computes distance, and returns:
       - distance
       - same/different decision
    Threshold is adjustable (depends on your dataset).
    """
    model.eval()

    # load samples
    x1 = load_single_sample(csv1, feature_cols, scaler).to(device)
    x2 = load_single_sample(csv2, feature_cols, scaler).to(device)
    lengths = torch.tensor([1], dtype=torch.long).to(device)

    with torch.no_grad():
        e1 = model(x1, lengths)  # embedding
        e2 = model(x2, lengths)

        distance = torch.norm(e1 - e2, dim=1).item()

    same = distance < threshold
    return distance, same

import pickle

def save_model(model, scaler, feature_cols, path="trained_snn.pth"):
    """
    Save the model weights + scaler + feature column list.
    """
    save_obj = {
        "model_state": model.state_dict(),
        "scaler": scaler,
        "feature_cols": feature_cols
    }
    with open(path, "wb") as f:
        pickle.dump(save_obj, f)
    print(f"Model saved to: {path}")

def load_model(path, device="cpu"):
    """
    Loads a previously saved SNN model with scaler + feature columns.
    Returns: (model, scaler, feature_cols)
    """
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)

    scaler = checkpoint["scaler"]
    feature_cols = checkpoint["feature_cols"]

    input_dim = len(feature_cols)

    model = LSTMEmbeddingNet(input_dim=input_dim)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    print(f"Model loaded from: {path}")
    return model, scaler, feature_cols
# ---------------------------------------------------------
#  ENTRY POINT
# ---------------------------------------------------------
if __name__ == "__main__":
    # ----- 1) Train the network -----
    model = run_training_pipeline("ksenia_training_2.csv")

    # Reload dataset to recover scaler + feature cols
    dataset = CMUDatasetTriplet("ksenia_training_2.csv")
    scaler = dataset.scaler
    feature_cols = dataset.feature_cols

    # ----- 2) SAVE TRAINED MODEL -----
    save_model(model, scaler, feature_cols, path="../models/ksenia_snn_model.pth")
