import os
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from src.ml.triplet_dataset import CMUDatasetTriplet, collate_triplet
from src.ml.snn_model import TripletSNN


class ModelTrainer:
    """Handles training, checkpointing, and saving the SNN model."""

    def __init__(self, csv_path, out_dir, batch_size=64, lr=1e-3,
                 hidden=128, embedding=128, dropout=0.3, margin=1.0, val_split=0.1):

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        # Load full dataset
        self.dataset = CMUDatasetTriplet(csv_path)
        self.input_dim = self.dataset.X.shape[1]
        print(f"X:{self.dataset.X.shape}")
        print(f"feature_cols: {self.dataset.feature_cols}")

        # Split into train and validation
        val_size = int(len(self.dataset) * val_split)
        train_size = len(self.dataset) - val_size
        self.train_dataset, self.val_dataset = random_split(self.dataset, [train_size, val_size])

        # DataLoaders
        self.train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_triplet)
        self.val_loader = DataLoader(self.val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_triplet)

        # Model, device, optimizer, criterion
        self.model = TripletSNN(
            input_dim=self.input_dim,
            lstm_hidden=hidden,
            embedding_dim=embedding,
            dropout=dropout
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.TripletMarginLoss(margin=margin, p=2)

    def train(self, epochs=1):
        print("Training on", self.device)

        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0

            for A, P, N, L in self.train_loader:
                A, P, N, L = A.to(self.device), P.to(self.device), N.to(self.device), L.to(self.device)

                embA, embP, embN = self.model(A, P, N, L)
                loss = self.criterion(embA, embP, embN)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * A.size(0)

            avg_loss = total_loss / len(self.train_dataset)
            print(f"Epoch {epoch}/{epochs}  train_loss={avg_loss:.5f}")

        # Save model
        final_path = os.path.join(self.out_dir, "snn_final.pt")
        torch.save({'epoch': epochs, 'model_state': self.model.state_dict()}, final_path)
        print("Final model saved:", final_path)

        # Compute accuracy on training and validation
        train_acc = self.evaluate_accuracy(self.train_loader)
        val_acc = self.evaluate_accuracy(self.val_loader)

        print(f"Train Triplet Accuracy: {train_acc:.4f}")
        print(f"Validation Triplet Accuracy: {val_acc:.4f}")

    def evaluate_accuracy(self, dataloader):
        """Compute fraction of triplets where anchor is closer to positive than negative."""
        self.model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for A, P, N, L in dataloader:
                A, P, N, L = A.to(self.device), P.to(self.device), N.to(self.device), L.to(self.device)
                embA, embP, embN = self.model(A, P, N, L)

                d_ap = torch.norm(embA - embP, p=2, dim=1)
                d_an = torch.norm(embA - embN, p=2, dim=1)

                correct += (d_ap < d_an).sum().item()
                total += A.size(0)

        return correct / total if total > 0 else 0.0
