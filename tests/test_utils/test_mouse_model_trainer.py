# test_mouse_model_trainer.py
import os
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch, MagicMock
from src.ml.mouse_model_trainer import MouseModelTrainer
import src.constants as constants


# ------------------------
# Fixtures
# ------------------------
@pytest.fixture
def sample_csv(tmp_path):
    path = tmp_path / "user.csv"
    df = pd.DataFrame({
        "x": [0, 1, 2, 3, 4],
        "y": [0, 1, 2, 3, 4],
        "client timestamp": [0, 1, 2, 3, 4]
    })
    df.to_csv(path, index=False)
    return str(path)


@pytest.fixture
def trainer(sample_csv, tmp_path):
    return MouseModelTrainer(
        enrollment_csv=sample_csv,
        username="test_user",
        dataset_root=str(tmp_path),
        out_dir=str(tmp_path),
        window_size=2,
        step_size=1,
        epochs=1
    )


# ------------------------
# Test CSV loading
# ------------------------
def test_load_csv(sample_csv):
    df = MouseModelTrainer.load_csv(sample_csv)
    assert isinstance(df, pd.DataFrame)
    assert all(col in df.columns for col in ["x", "y", "client timestamp"])


def test_load_csv_not_found():
    with pytest.raises(FileNotFoundError):
        MouseModelTrainer.load_csv("non_existent.csv")


# ------------------------
# Test loading all CSVs in folder
# ------------------------
def test_load_user_csvs(tmp_path):
    # Create multiple CSVs
    for i in range(2):
        pd.DataFrame({"x": [i], "y": [i], "timestamp": [i]}).to_csv(tmp_path / f"{i}.csv", index=False)
    df = MouseModelTrainer.load_user_csvs(str(tmp_path))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


# ------------------------
# Test compute_velocity
# ------------------------
def test_compute_velocity():
    df = pd.DataFrame({
        "x": [0, 3, 6],
        "y": [0, 4, 8],
        "timestamp": [0, 1, 2]
    })
    velocities = MouseModelTrainer.compute_velocity(df)
    assert velocities.shape == (2, 2)
    assert np.allclose(velocities, [[3, 4], [3, 4]])


# ------------------------
# Test create_windows
# ------------------------
def test_create_windows(trainer):
    signal = np.arange(10).reshape(-1, 1)
    windows = trainer.create_windows(signal)
    # window_size=2, step_size=1 => 8 windows
    assert windows.shape[0] == 8
    assert windows.shape[1] == 2


# ------------------------
# Test prepare_dataset with mocks
# ------------------------
@patch.object(MouseModelTrainer, "load_csv")
@patch.object(MouseModelTrainer, "load_user_csvs")
@patch.object(MouseModelTrainer, "compute_velocity")
@patch.object(MouseModelTrainer, "create_windows")
def test_prepare_dataset(
    mock_windows, mock_velocity, mock_load_user_csvs, mock_load_csv, trainer
):
    mock_load_csv.return_value = pd.DataFrame()
    mock_load_user_csvs.return_value = pd.DataFrame()

    # velocity: (N, 2)
    mock_velocity.return_value = np.arange(40).reshape(-1, 2)

    # windows: (num_windows, window_size, 2)
    mock_windows.return_value = np.arange(
        trainer.window_size * 2 * 4
    ).reshape(4, trainer.window_size, 2)

    X_train, X_test, y_train, y_test = trainer.prepare_dataset()

    assert X_train.ndim == 3
    assert X_train.shape[1:] == (trainer.window_size, 2)
    assert set(y_train).issubset({0, 1})



# ------------------------
# Test build_model
# ------------------------
def test_build_model(trainer):
    trainer.window_size = 32

    trainer.build_model()

    assert trainer.model is not None

    last_layer = trainer.model.layers[-1]
    assert last_layer.activation.__name__ == "sigmoid"


# ------------------------
# Test calculate_threshold
# ------------------------
def test_calculate_threshold(trainer, tmp_path):
    imposter_scores = np.array([0.1, 0.2, 0.3])
    trainer.username = "test_user"
    constants.TARGET_FAR = 0.2  # 20% FAR for test
    trainer.out_dir = str(tmp_path)

    # Patch np.save to avoid actual file creation
    with patch("numpy.save") as mock_save:
        trainer.calculate_threshold(imposter_scores)
        threshold_file = os.path.join(constants.PATH_MODELS, "test_user_mouse_threshold.npy")
        expected_threshold = np.percentile(imposter_scores, 100 * (1 - constants.TARGET_FAR))
        mock_save.assert_called_once()
        np.testing.assert_equal(mock_save.call_args[0][1], expected_threshold)
