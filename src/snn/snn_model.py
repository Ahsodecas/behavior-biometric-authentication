import torch
from torch import nn


class SiameseSubNet(nn.Module):
    """Two-layer LSTM + BN + Dropout embedding network."""

    def __init__(self, input_dim, lstm_hidden=128, lstm_layers=2,
                 embedding_dim=128, dropout=0.3):
        super().__init__()

        self.batchnorm = nn.BatchNorm1d(input_dim)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_hidden, embedding_dim)
        self.l2norm = nn.functional.normalize

    def forward(self, x, lengths=None):
        if x.dim() == 2:
            x = x.unsqueeze(1)

        B, S, F = x.shape
        x = x.reshape(B * S, F)
        x = self.batchnorm(x)
        x = x.view(B, S, F)

        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, (hn, _) = self.lstm(packed)
            h = hn[-1]
        else:
            out, _ = self.lstm(x)
            h = out[:, -1, :]

        h = self.dropout(h)
        emb = self.fc(h)
        return self.l2norm(emb, p=2, dim=1)


class TripletSNN(nn.Module):
    """Anchor, Positive, Negative → Embeddings."""

    def __init__(self, input_dim, **kwargs):
        super().__init__()
        self.subnet = SiameseSubNet(input_dim, **kwargs)

    def forward(self, A, P, N, lengths=None):
        return (
            self.subnet(A, lengths),
            self.subnet(P, lengths),
            self.subnet(N, lengths)
        )
