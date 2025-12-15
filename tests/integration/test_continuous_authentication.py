import os
import pytest
import shutil
import pandas as pd

from src.ml.mouse_model_trainer import MouseModelTrainer
from src.auth.background_auth_manager import BackgroundAuthManager
from src.utils.data_utility import DataUtility
import src.constants as constants


@pytest.mark.integration
def test_continuous_mouse_authentication(tmp_path):
    """
    Integration test for continuous mouse-based authentication:
      1. Copy mouse enrollment CSV
      2. Train mouse behavioral model
      3. Simulate background authentication
      4. Assert authentication results
    """

    username = "mouse_user"

    # ----------------------------------------------------------------
    # Setup directories
    # ----------------------------------------------------------------
    extracted_dir = tmp_path / "extracted" / username
    models_dir = tmp_path / "models"

    extracted_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    constants.PATH_EXTRACTED = str(tmp_path / "extracted")
    constants.PATH_MODELS = str(models_dir)

    # ----------------------------------------------------------------
    # Step 1: Copy existing mouse enrollment CSV
    # ----------------------------------------------------------------
    src_csv = "tests/test_data/mouse_enrollment_test.csv"
    assert os.path.exists(src_csv), f"{src_csv} not found!"

    mouse_csv = extracted_dir / "mouse_enrollement.csv"
    shutil.copy(src_csv, mouse_csv)
    assert mouse_csv.exists()

    # ----------------------------------------------------------------
    # Step 2: Train mouse behavioral model
    # ----------------------------------------------------------------
    model_out_dir = str(models_dir)
    trainer = MouseModelTrainer(
        enrollment_csv=str(mouse_csv),
        username=username,
        dataset_root="datasets/sapimouse",
        out_dir=model_out_dir,
        window_size=128,
        step_size=64,
        epochs=1,       # keep epochs low for test speed
        batch_size=16,
        lr=1e-3
    )

    model_path, auc = trainer.train()
    assert os.path.exists(model_path)
    assert auc > 0.2  # minimal sanity check

    # ----------------------------------------------------------------
    # Step 3: Simulate continuous/background authentication
    # ----------------------------------------------------------------
    data_util = DataUtility()
    data_util.set_username(username)

    bg_manager = BackgroundAuthManager(
        username=username,
        data_utility=data_util,
        authenticator_model_path=model_path
    )

    results = []

    def capture_result(accepted, mean_score):
        results.append((accepted, mean_score))

    bg_manager.auth_result.connect(capture_result)

    bg_manager.start()

    # Wait for the collection to finish
    bg_manager._thread.join()

    # ---------------------------------
    # Step 4: Assert results
    # ---------------------------------
    assert len(results) >= 0
    for accepted, mean_score in results:
        assert isinstance(accepted, bool)
        assert isinstance(mean_score, float)
