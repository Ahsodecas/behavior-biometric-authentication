import os
import pytest
import pandas as pd
import random

from src.auth.authentication_decision_maker import AuthenticationDecisionMaker
from src.ml.data_preprocessor import DataPreprocessor
from src.ml.model_trainer import ModelTrainer
import src.gui.constants as constants


@pytest.mark.integration
def test_registration_and_authentication_pipeline(tmp_path):
    """
    Full integration test:
      1. Create fake enrollment data
      2. Generate synthetic and imposter features
      3. Build training CSV
      4. Train model
      5. Perform one-time authentication
      6. Assert outputs
    """

    username = "test_user"
    password = "password123"  # Replace with constants.PASSWORD if needed

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

    enrollment_csv = extracted_dir / "enrollment_features.csv"
    training_csv = datasets_dir / f"{username}_training.csv"
    model_path = models_dir / f"{username}_snn.pt"

    # ----------------------------------------------------------------
    # Step 1: Create fake enrollment CSV with keystroke dynamics
    # ----------------------------------------------------------------

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
    assert len(combined_df) >= num_samples

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

    print(df)

    authenticator = AuthenticationDecisionMaker(threshold=0.3)
    authenticator.load_model(model_path, username=username, training_csv=training_csv)

    sample_features = df.drop(columns=["label"]).iloc[0].to_dict()

    success, dist, message = authenticator.authenticate(username, password, sample_features)

    # ----------------------------------------------------------------
    # Step 6: Assert authentication outputs
    # ----------------------------------------------------------------
    assert isinstance(success, bool)
    assert isinstance(dist, float)
    assert isinstance(message, str)
