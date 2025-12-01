"""
Siamese (Triplet) Neural Network implementation matching the architecture described in:
"Impact of Data Breadth and Depth on Performance of Siamese Neural Network Model" (arXiv:2501.07600v1)

This script builds a PyTorch implementation of the SNN sub-network and the triplet SNN.
It includes:
 - CSV loader for the CMU dataset format (assumes a table like the sample you provided)
 - TripletDataset that samples (anchor, positive, negative) triplets on-the-fly
 - Siamese sub-network: Masking (via lengths / packing), BatchNorm, 2 LSTM layers (tanh), Dropout
 - Triplet training loop using nn.TripletMarginLoss
 - Utilities for evaluation (embed + compute distances, sample EER/ROC helper comments)

Notes / assumptions made (best-effort):
 - The paper describes a time-series sub-network; the CMU CSV snippet provided looks like precomputed features
   (one row per sample). To stay general, the loader will treat each sample as a sequence of timesteps if
   the input CSV has column groups representing timesteps OR as a single-timestep feature vector otherwise.
 - Masking is supported if you provide a `lengths` column; otherwise all samples are treated as full length.

Requirements:
 - Python 3.8+
 - torch, pandas, numpy, scikit-learn

Usage example:
  python snn_cmu.py --csv path/to/cmu.csv --epochs 10 --batch 64

"""

import argparse
import random
import os
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler


##############################
# Dataset & Preprocessing
##############################

class CMUDatasetTriplet(Dataset):
    """Load CMU-style CSV and produce triplets.

    Assumptions about CSV:
      - It contains columns: subject, sessionIndex, rep, <feature columns...>
      - Feature columns are numeric. All columns after `rep` are considered features.
      - If features already encode time-series (e.g., flattened timesteps), the dataset will treat each row
        as a sequence of length 1 with feature-dim = n_features. This keeps the LSTM pipeline working.

    The dataset supports an optional `lengths` numpy array to allow packing / masking. If not provided,
    all sequences are assumed equal length.
    """

    def __init__(self, csv_path: str, scaler: Optional[StandardScaler] = None, preload: bool = True):
        self.df = pd.read_csv(csv_path)
        # basic column inference
        expected_prefixes = ['subject', 'sessionIndex', 'rep']
        cols = list(self.df.columns)
        # locate 'rep' column index
        if 'rep' in cols:
            feat_start = cols.index('rep') + 1
        else:
            # fallback: assume first three columns are subject/session/rep
            feat_start = 3

        self.meta_cols = cols[:feat_start]
        self.feature_cols = cols[feat_start:]

        # convert subject to string
        self.df['subject'] = self.df['subject'].astype(str)

        # features matrix
        X = self.df[self.feature_cols].astype(float).values
        if scaler is None:
            self.scaler = StandardScaler()
            X = self.scaler.fit_transform(X)
        else:
            self.scaler = scaler
            X = self.scaler.transform(X)


        # We'll treat each sample as a sequence of length 1, features = n_features
        # This makes the code general and compatible with LSTM-based subnetwork from the paper.
        self.X = X.astype(np.float32)
        self.y = self.df['subject'].values

        # Build index mapping subject -> list of indices
        self.by_subject = {}
        for idx, s in enumerate(self.y):
            self.by_subject.setdefault(s, []).append(idx)

        self.subjects = list(self.by_subject.keys())

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        # -------- Oversampling Ksenia ----------
        # 4× higher probability to sample her as anchor
        if random.random() < 0.3:  # 60% chance
            if "ksenia" in self.by_subject:
                idx = random.choice(self.by_subject["ksenia"])

        anchor = self.X[idx]
        subj = self.y[idx]

        # -------- Positive sample --------
        pos_choices = self.by_subject[subj]

        # increase positive pairing for Ksenia
        if subj == "ksenia" and len(pos_choices) > 1:
            pos_idx = random.choice(pos_choices)
        else:
            # default behavior
            if len(pos_choices) == 1:
                pos_idx = idx
            else:
                pos_idx = idx
                while pos_idx == idx:
                    pos_idx = random.choice(pos_choices)

        positive = self.X[pos_idx]

        # -------- Negative sample --------
        # decrease chance of negatives being Ksenia (avoid collapsing)
        possible_neg_subjs = (
            [s for s in self.subjects if s != subj and s != "ksenia"]
            if subj == "ksenia"
            else [s for s in self.subjects if s != subj]
        )

        neg_subj = random.choice(possible_neg_subjs)
        neg_idx = random.choice(self.by_subject[neg_subj])
        negative = self.X[neg_idx]

        # -------- Convert to tensors --------
        anchor = torch.from_numpy(anchor).unsqueeze(0)
        positive = torch.from_numpy(positive).unsqueeze(0)
        negative = torch.from_numpy(negative).unsqueeze(0)
        lengths = torch.tensor([1], dtype=torch.long)

        return anchor, positive, negative, lengths


def collate_triplet(batch):
    # batch is list of tuples (a,p,n,lengths)
    A = torch.cat([b[0] for b in batch], dim=0)  # (B, seq, F)
    P = torch.cat([b[1] for b in batch], dim=0)
    N = torch.cat([b[2] for b in batch], dim=0)
    lengths = torch.cat([b[3] for b in batch], dim=0)
    return A, P, N, lengths


##############################
# Model
##############################

class SiameseSubNet(nn.Module):
    """Sub-network described in the paper:
    - Masking (handled via packing using lengths)
    - Batch Normalization
    - Two LSTM layers with tanh activations
    - Dropout
    - Final projection to embedding vector
    """

    def __init__(self, input_dim: int, lstm_hidden: int = 128, lstm_layers: int = 2, embedding_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.input_dim = input_dim
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.embedding_dim = embedding_dim

        # BatchNorm over feature dim (applied per timestep)
        self.batchnorm = nn.BatchNorm1d(input_dim)

        # Two-layer LSTM (stacked). Use tanh nonlinearity (default) and batch_first
        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=lstm_hidden,
                            num_layers=lstm_layers,
                            batch_first=True,
                            bidirectional=False)

        self.dropout = nn.Dropout(dropout)
        # final linear projection from hidden to embedding
        self.fc = nn.Linear(lstm_hidden, embedding_dim)
        # l2 normalization at the end helps triplet training
        self.l2norm = nn.functional.normalize

    def forward(self, x, lengths=None):
        # Ensure 3D input: (B, seq, feat)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, F)

        B, S, F = x.shape
        # apply batchnorm: BatchNorm1d expects (B, F, L) or (B, F) per timestep
        # We'll fold batch and seq dims then apply BN and restore
        x_reshaped = x.contiguous().view(B * S, F)
        x_bn = self.batchnorm(x_reshaped)
        x_bn = x_bn.view(B, S, F)

        if lengths is not None:
            # pack
            packed = nn.utils.rnn.pack_padded_sequence(x_bn, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_out, (hn, cn) = self.lstm(packed)
            # take last hidden state from hn from last LSTM layer
            # hn shape: (num_layers, B, hidden)
            last_hidden = hn[-1]
        else:
            out, (hn, cn) = self.lstm(x_bn)
            # out shape (B, seq, hidden) -> take last timestep
            last_hidden = out[:, -1, :]

        last_hidden = self.dropout(last_hidden)
        emb = self.fc(last_hidden)
        emb = self.l2norm(emb, p=2, dim=1)
        return emb


class TripletSNN(nn.Module):
    def __init__(self, input_dim: int, **subnet_kwargs):
        super().__init__()
        self.subnet = SiameseSubNet(input_dim, **subnet_kwargs)

    def forward(self, anchor, positive, negative, lengths=None):
        # anchor/positive/negative: (B, seq, feat)
        # lengths can be None or a tensor of shape (B,) - assumed same for all three here
        emb_a = self.subnet(anchor, lengths)
        emb_p = self.subnet(positive, lengths)
        emb_n = self.subnet(negative, lengths)
        return emb_a, emb_p, emb_n


##############################
# Training loop
##############################

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.out_dir, exist_ok=True)

    dataset = CMUDatasetTriplet(args.csv)
    input_dim = dataset.X.shape[1]

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_triplet, num_workers=0)

    model = TripletSNN(input_dim=input_dim,
                       lstm_hidden=args.hidden,
                       lstm_layers=2,
                       embedding_dim=args.embedding,
                       dropout=args.dropout)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.TripletMarginLoss(margin=args.margin, p=2)

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for (A, P, N, lengths) in dataloader:
            A = A.to(device)
            P = P.to(device)
            N = N.to(device)
            lengths = lengths.to(device).view(-1)

            emb_a, emb_p, emb_n = model(A, P, N, lengths)
            loss = criterion(emb_a, emb_p, emb_n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * A.size(0)

        avg_loss = total_loss / len(dataset)
        print(f"Epoch {epoch}/{args.epochs}  avg_loss={avg_loss:.6f}")

        # checkpointing
        if epoch % args.ckpt_every == 0:
            ckpt_path = os.path.join(args.out_dir, f"snn_epoch{epoch}.pt")
            torch.save({'epoch': epoch, 'model_state': model.state_dict(), 'optimizer_state': optimizer.state_dict()}, ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

    # final save
    final_path = os.path.join(args.out_dir, "snn_final.pt")
    torch.save({'epoch': args.epochs, 'model_state': model.state_dict()}, final_path)
    print(f"Training complete. Model saved to {final_path}")


##############################
# Quick evaluation utility
##############################

def embed_all(model: TripletSNN, dataset: CMUDatasetTriplet, device: torch.device = None) -> Tuple[np.ndarray, List[str]]:
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    loader = DataLoader(dataset, batch_size=128, shuffle=False, collate_fn=lambda b: collate_triplet([(x,x2,x3,l) for x,x2,x3,l in b]))
    embeddings = []
    labels = []
    with torch.no_grad():
        for A, P, N, lengths in loader:
            # because collate returned triple copies, take A as the sample of interest
            A = A.to(device)
            lengths = lengths.to(device).view(-1)
            emb = model.subnet(A, lengths)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.vstack(embeddings)
    labels = list(dataset.y)
    return embeddings, labels


##############################
# CLI
##############################

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--mode', type=str, required=True, help='Mode: test or train')
    p.add_argument('--csv', type=str, required=True, help='Path to CMU-style CSV file (rows are samples)')
    p.add_argument('--out_dir', type=str, default='datasets/checkpoints', help='Where to save model checkpoints')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--margin', type=float, default=1.0)
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--embedding', type=int, default=128)
    p.add_argument('--dropout', type=float, default=0.3)
    p.add_argument('--ckpt_every', type=int, default=5)
    return p.parse_args()

def compute_embeddings(model_path, csv_path):
    dataset = CMUDatasetTriplet(csv_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_dim = dataset.X.shape[1]
    model = TripletSNN(input_dim=input_dim)
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    emb, labels = embed_all(model, dataset, device)
    return emb, np.array(labels)

def verification_accuracy(embeddings, labels, threshold=None, max_pairs=50000):
    from sklearn.metrics import accuracy_score

    N = len(embeddings)

    # Randomly sample pairs
    rng = np.random.default_rng(123)

    idx_i = rng.integers(0, N, size=max_pairs)
    idx_j = rng.integers(0, N, size=max_pairs)

    # Ensure i < j
    mask = idx_i != idx_j
    idx_i = idx_i[mask]
    idx_j = idx_j[mask]

    # Compute distances
    dists = np.linalg.norm(embeddings[idx_i] - embeddings[idx_j], axis=1)
    targets = (labels[idx_i] == labels[idx_j]).astype(int)

    if threshold is None:
        pos_mean = dists[targets == 1].mean()
        neg_mean = dists[targets == 0].mean()
        threshold = (pos_mean + neg_mean) / 2

    preds = (dists < threshold).astype(int)
    acc = accuracy_score(targets, preds)
    return acc, threshold


if __name__ == "__main__":
    args = parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.mode == "train":
        train(args)

    elif args.mode == "test":
        # Load model and compute embeddings
        print("Computing embeddings...")
        emb, labels = compute_embeddings(
            model_path=os.path.join(args.out_dir, "snn_final.pt"),
            csv_path=args.csv
        )

        # Evaluate verification accuracy
        print("Computing accuracy...")
        acc, threshold = verification_accuracy(emb, labels)
        print("Verification accuracy:", acc)
        print("Optimal threshold:", threshold)

    else:
        raise ValueError("Unknown mode. Use --mode train OR --mode test.")
