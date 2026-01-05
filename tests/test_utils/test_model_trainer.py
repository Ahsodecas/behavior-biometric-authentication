# test_model_trainer.py
import os
import torch
import pytest
from unittest.mock import MagicMock, patch
from src.ml.model_trainer import ModelTrainer
import torch
import torch.nn as nn


# ------------------------
# Fixtures
# ------------------------
@pytest.fixture
def trainer(tmp_path):
    out_dir = tmp_path / "models"
    return ModelTrainer(
        csv_path="dummy.csv",
        out_dir=str(out_dir),
        username="test_user",
        batch_size=2,
        lr=1e-3,
        hidden=8,
        embedding=8,
        dropout=0.1,
        val_split=0.2
    )

class DummyModel(nn.Module):
    def __init__(self, input_dim=5, embedding_dim=8):
        super().__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)

    def forward(self, A, P, N, L):
        # Return embeddings for triplets
        return self.fc(A), self.fc(P), self.fc(N)

# ------------------------
# Test initialize
# ------------------------
@patch("src.ml.model_trainer.CMUDatasetTriplet")
@patch("torch.utils.data.DataLoader")
@patch("src.ml.model_trainer.TripletSNN", return_value=DummyModel())
def test_initialize(mock_model_class, mock_dataloader, mock_dataset, trainer):
    # Mock dataset
    dataset_instance = MagicMock()
    dataset_instance.X.shape = (10, 5)
    dataset_instance.feature_cols = ["f1","f2","f3","f4","f5"]
    dataset_instance.__len__.return_value = 10
    dataset_instance.__getitem__.side_effect = lambda idx: (
        torch.randn(5), torch.randn(5), torch.randn(5), torch.tensor(0)
    )

    mock_dataset.return_value = dataset_instance

    trainer.initialize()

    # Optimizer now has real parameters
    assert trainer.model is not None
    assert trainer.optimizer is not None
    assert len(list(trainer.model.parameters())) > 0

# ------------------------
# Test train raises if not initialized
# ------------------------
def test_train_not_initialized(trainer):
    with pytest.raises(RuntimeError):
        trainer.train()

# ------------------------
# Test evaluate_accuracy with mock
# ------------------------
@patch("src.ml.model_trainer.TripletSNN")
def test_evaluate_accuracy(mock_model_class, trainer):
    # Mock model to return predictable embeddings
    mock_model = MagicMock()
    trainer.model = mock_model
    trainer.device = torch.device("cpu")

    # Create dummy dataloader with 2 samples
    A = torch.tensor([[0.,0.],[1.,1.]])
    P = torch.tensor([[0.,0.],[1.,1.]])
    N = torch.tensor([[1.,1.],[0.,0.]])
    L = torch.tensor([0,1])
    dataloader = [(A,P,N,L)]

    mock_model.side_effect = lambda a,p,n,l: (a,p,n)
    acc = trainer.evaluate_accuracy(dataloader)
    assert 0 <= acc <= 1

# ------------------------
# Test full train with mocks (fast)
# ------------------------
@patch("src.ml.model_trainer.CMUDatasetTriplet")
@patch("src.ml.model_trainer.DataLoader")
@patch("src.ml.model_trainer.TripletSNN", return_value=DummyModel())
@patch("torch.save")
def test_train_full(mock_save, mock_model_class, mock_dataloader, mock_dataset, trainer):
    # Mock dataset
    dataset_instance = MagicMock()
    dataset_instance.X.shape = (4, 5)
    dataset_instance.feature_cols = ["f1","f2","f3","f4","f5"]
    dataset_instance.__len__.return_value = 4
    dataset_instance.__getitem__.side_effect = lambda idx: (
        torch.randn(4,5), torch.randn(4,5), torch.randn(4,5), torch.tensor([0,1])
    )
    mock_dataset.return_value = dataset_instance

    # Mock DataLoader to return iterable
    dummy_batch = [(torch.randn(4,5), torch.randn(4,5), torch.randn(4,5), torch.tensor([0,1]))]
    mock_loader_instance = MagicMock()
    mock_loader_instance.__iter__.return_value = iter(dummy_batch)
    mock_dataloader.return_value = mock_loader_instance

    # Initialize and train
    trainer.initialize()
    trainer.train(epochs=1)

    # Assertions
    mock_save.assert_called_once()
    assert trainer.model is not None
    assert len(list(trainer.model.parameters())) > 0
