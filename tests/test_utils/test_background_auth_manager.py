import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock, patch

from src.auth.background_auth_manager import BackgroundAuthManager

@pytest.fixture
def mock_model():
    model = MagicMock()
    model.predict.return_value = np.array([[0.1], [0.2], [0.3]])
    return model


@pytest.fixture
def manager(mock_model, tmp_path):
    with patch("tensorflow.keras.models.load_model", return_value=mock_model), \
         patch("src.auth.background_auth_manager.DataUtility") as mock_du, \
         patch("numpy.load", return_value=np.array(-0.1)):

        du = mock_du.return_value
        du.mouse_data_collector.get_data.return_value = pd.DataFrame()

        return BackgroundAuthManager(
            username="test_user",
            data_utility=du,
            authenticator_model_path="dummy_path"
        )

def test_compute_velocity_basic():
    df = pd.DataFrame({
        "x": [0, 1, 3],
        "y": [0, 2, 2],
        "timestamp": [0, 1, 2]
    })

    v = BackgroundAuthManager._compute_velocity(df)

    assert v.shape == (2, 2)
    assert np.allclose(v[0], [1.0, 2.0])

def test_create_windows():
    signal = np.arange(20).reshape(10, 2)

    windows = BackgroundAuthManager._create_windows(
        signal, window_size=4, step_size=2
    )

    assert windows.shape == (3, 4, 2)

def test_authenticate_empty_df(manager):
    df = pd.DataFrame()

    accepted, score = manager._authenticate(df)

    assert accepted is False
    assert score == 0.0

def test_authenticate_no_windows(manager):
    df = pd.DataFrame({
        "x": [0, 1],
        "y": [0, 1],
        "timestamp": [0, 1]
    })

    accepted, score = manager._authenticate(df)

    assert accepted is False
    assert score == 0.0

@patch("src.auth.background_auth_manager.np.load", return_value=np.array(0.4))
def test_authenticate_accepted(mock_load, manager, mock_model):
    df = pd.DataFrame({
        "x": np.arange(200),
        "y": np.arange(200),
        "timestamp": np.arange(200)
    })

    mock_model.predict.return_value = np.array([
        [0.5], [0.6], [0.7]
    ])

    accepted, score = manager._authenticate(df)

    assert bool(accepted) is True
    assert isinstance(score, float)


@patch("src.auth.background_auth_manager.np.load", return_value=np.array(0.0))
def test_authenticate_rejected(mock_load, manager, mock_model):
    df = pd.DataFrame({
        "x": np.arange(200),
        "y": np.arange(200),
        "timestamp": np.arange(200)
    })

    mock_model.predict.return_value = np.array([
        [-1.0], [-0.8], [-0.9]
    ])

    accepted, score = manager._authenticate(df)

    assert bool(accepted) is False
    assert isinstance(score, float)

def test_start_emits_status(manager):
    manager.status_update = MagicMock()

    with patch("threading.Thread.start"):
        manager.start()

    manager.status_update.emit.assert_called_with("● Collecting data")


def test_stop_stops_collection(manager):
    manager.status_update = MagicMock()
    manager.data_utility.stop_background_collection = MagicMock()

    manager.stop()

    manager.data_utility.stop_background_collection.assert_called_once()
    manager.status_update.emit.assert_called_with("● Idle")




