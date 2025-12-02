import os
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.snn.triplet_dataset import CMUDatasetTriplet, collate_triplet
from src.snn.snn_model import TripletSNN


class ModelTrainer:
    """Handles training, checkpointing, and saving the SNN model."""

    def __init__(self, csv_path, out_dir, batch_size=64, lr=1e-3,
                 hidden=128, embedding=128, dropout=0.3, margin=1.0):

        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

        self.dataset = CMUDatasetTriplet(csv_path)
        self.input_dim = self.dataset.X.shape[1]

        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_triplet
        )

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

            for A, P, N, L in self.dataloader:
                A, P, N, L = A.to(self.device), P.to(self.device), N.to(self.device), L.to(self.device)

                embA, embP, embN = self.model(A, P, N, L)
                loss = self.criterion(embA, embP, embN)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * A.size(0)

            avg = total_loss / len(self.dataset)
            print(f"Epoch {epoch}/{epochs}  loss={avg:.5f}")


        final_path = f"{self.out_dir}/snn_final.pt"
        torch.save({'epoch': 1, 'model_state': self.model.state_dict()}, final_path)
        print("Final model saved:", final_path)
