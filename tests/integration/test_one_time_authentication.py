import os
import pytest
import pandas as pd

from src.auth.authentication_decision_maker import AuthenticationDecisionMaker
from src.ml.data_preprocessor import DataPreprocessor
from src.ml.model_trainer import ModelTrainer
import src.constants as constants


@pytest.mark.integration
def test_one_time_authentication_pipeline(tmp_path):
    """
    Full integration test of ONE TIME authentication
    using an existing enrollment_features_test.csv:
      1. Copy enrollment CSV into temp extracted folder
      2. Generate synthetic and imposter features
      3. Build training CSV
      4. Train model
      5. Perform one-time authentication
      6. Assert outputs
    """

    username = "s002"
    password = ".tie5Roanl"

    # ----------------------------------------------------------------
    # Setup directories
    # ----------------------------------------------------------------
    extracted_dir = tmp_path / "extracted" / username
    datasets_dir = tmp_path / "datasets"
    models_dir = tmp_path / "models"

    extracted_dir.mkdir(parents=True)
    datasets_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)

    # Patch constants paths
    constants.PATH_EXTRACTED = str(tmp_path / "extracted")
    constants.PATH_DATASETS = str(datasets_dir)
    constants.PATH_MODELS = str(models_dir)

    # ----------------------------------------------------------------
    # Step 1: Copy existing enrollment CSV
    # ----------------------------------------------------------------
    src_csv = "tests/test_data/enrollment_features_test.csv"
    assert os.path.exists(src_csv), f"{src_csv} not found!"

    enrollment_csv = extracted_dir / "enrollment_features.csv"
    training_csv = datasets_dir / f"{username}_training.csv"
    model_path = models_dir / f"{username}_snn.pt"

    # Copy test CSV to extracted folder
    import shutil
    shutil.copy(src_csv, enrollment_csv)
    assert enrollment_csv.exists()

    # ----------------------------------------------------------------
    # Step 2: Build training CSV (enrollment + synthetic + imposter)
    # ----------------------------------------------------------------
    preprocessor = DataPreprocessor(
        enrollment_csv=str(enrollment_csv),
        username=username,
        output_csv=str(training_csv),
        synth_reps=10
    )
    combined_df = preprocessor.build_training_csv()
    assert combined_df is not None
    assert training_csv.exists()
    assert len(combined_df) >= 1

    # ----------------------------------------------------------------
    # Step 3: Train model (headless)
    # ----------------------------------------------------------------
    trainer = ModelTrainer(
        csv_path=str(training_csv),
        username=username,
        out_dir=str(models_dir),
        batch_size=16,
        lr=1e-3
    )
    trainer.initialize()
    trainer.train()

    # Step 4: Assert model file exists
    assert model_path.exists()
    assert os.path.getsize(model_path) > 0

    # ----------------------------------------------------------------
    # Step 5: One-time authentication
    # ----------------------------------------------------------------
    df = pd.read_csv(training_csv)
    assert not df.empty

    authenticator = AuthenticationDecisionMaker(threshold=0.3)
    authenticator.load_model(model_path, username=username, training_csv=training_csv)

    sample_features = df.iloc[0].to_dict()

    success, dist, message = authenticator.authenticate(username, password, sample_features)
    dist = float(dist)

    # ----------------------------------------------------------------
    # Step 6: Assert authentication outputs
    # ----------------------------------------------------------------
    assert isinstance(success, bool)
    assert isinstance(dist, float)
    assert isinstance(message, str)
