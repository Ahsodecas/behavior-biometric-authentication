import pandas as pd
import numpy as np
import torch
import pytest

from sklearn.preprocessing import StandardScaler
from src.ml.triplet_dataset import CMUDatasetTriplet, collate_triplet


def create_dummy_csv(path):
    """
    Creates a small dataset with two subjects.
    rep column exists -> feature start = index(rep) + 1
    """
    df = pd.DataFrame({
        "subject": ["s1", "s1", "ksenia", "s2"],
        "session": [1, 1, 1, 1],
        "rep": [0, 0, 0, 0],
        "f1": [1.0, 2.0, 3.0, 4.0],
        "f2": [5.0, 6.0, 7.0, 8.0],
    })
    df.to_csv(path, index=False)
    return df


def test_getitem_oversampling(tmp_path, mocker):
    """
    When random.random() < 0.3, index must be replaced with some "ksenia" index.
    """
    csv_path = tmp_path / "data.csv"
    df = create_dummy_csv(csv_path)
    ds = CMUDatasetTriplet(csv_path)

    # Force oversampling every time
    mocker.patch("random.random", return_value=0.0)

    # Mock random.choice to return the SECOND ksenia index if multiple
    mocker_choice = mocker.patch("random.choice", side_effect=lambda lst: lst[-1])

    a, p, n, l = ds[0]

    # The anchor must come from ksenia row. ksenia is row index 2.
    assert torch.allclose(a.squeeze(0), torch.tensor(ds.X[2]))

    # Ensure random.choice got called
    assert mocker_choice.called


def test_collate_triplet():
    b1 = (
        torch.tensor([[1., 2.]]),
        torch.tensor([[3., 4.]]),
        torch.tensor([[5., 6.]]),
        torch.tensor([1])
    )
    b2 = (
        torch.tensor([[7., 8.]]),
        torch.tensor([[9., 10.]]),
        torch.tensor([[11., 12.]]),
        torch.tensor([1])
    )

    A, P, N, L = collate_triplet([b1, b2])

    assert A.shape == (2, 2)
    assert P.shape == (2, 2)
    assert N.shape == (2, 2)
    assert L.shape == (2,)

    assert torch.allclose(A, torch.tensor([[1., 2.], [7., 8.]]))
    assert torch.allclose(P, torch.tensor([[3., 4.], [9., 10.]]))
    assert torch.allclose(N, torch.tensor([[5., 6.], [11., 12.]]))


