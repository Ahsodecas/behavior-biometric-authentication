import os
import pandas as pd
import pytest

from src.ml.data_preprocessor import DataPreprocessor

def write_csv(path, rows):
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def test_load_csv_success(tmp_path):
    csv_path = tmp_path / "enroll.csv"
    df_orig = write_csv(csv_path, [{"a": 1}, {"a": 2}])

    dp = DataPreprocessor(str(csv_path), "", "", "")
    df = dp.load_csv(str(csv_path))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(df_orig)


def test_load_csv_missing_file(tmp_path):
    csv_path = tmp_path / "missing.csv"

    dp = DataPreprocessor(str(csv_path), "", "", "")
    df = dp.load_csv(str(csv_path))

    assert df is None


def test_generate_synthetic_happy_path(tmp_path, mocker):
    enrollment_path = tmp_path / "enroll.csv"
    write_csv(enrollment_path, [{"f1": 1, "f2": 2}])

    synth_dir = tmp_path / "extracted_features" / "user1"
    synth_dir.mkdir(parents=True, exist_ok=True)

    synthetic_file_path = synth_dir / "synthetic_temp.csv"
    write_csv(synthetic_file_path, [{"f1": 10, "f2": 20}])

    mocker.patch("os.path.exists", return_value=True)

    mock_du = mocker.MagicMock()
    mock_du.feature_extractor.load_row_into_feature_extractor.return_value = None
    mock_du.generate_synthetic_features.return_value = None

    dp = DataPreprocessor(
        enrollment_csv=str(enrollment_path),
        dsl_dataset_csv="",
        username="user1",
        output_csv="out.csv"
    )
    dp.data_utility = mock_du

    def fake_load_csv(path):
        if "synthetic_temp.csv" in path:
            return pd.read_csv(synthetic_file_path)
        return pd.read_csv(enrollment_path)

    mocker.patch.object(dp, "load_csv", side_effect=fake_load_csv)

    synth_df = dp.generate_synthetic(str(enrollment_path), "user1")

    assert isinstance(synth_df, pd.DataFrame)
    assert len(synth_df) == 1
    assert synth_df.iloc[0]["f1"] == 10


def test_generate_synthetic_empty_input(tmp_path, mocker):
    enrollment_path = tmp_path / "enroll.csv"
    write_csv(enrollment_path, [])

    dp = DataPreprocessor(
        enrollment_csv=str(enrollment_path),
        dsl_dataset_csv="",
        username="userX",
        output_csv="out.csv"
    )

    mocker.patch.object(dp, "load_csv", return_value=pd.DataFrame())

    df = dp.generate_synthetic(str(enrollment_path), "userX")

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_build_training_csv_success(tmp_path, mocker):
    enroll_path = tmp_path / "enroll.csv"
    dsl_path = tmp_path / "dsl.csv"
    output_path = tmp_path / "out.csv"

    enroll_df = write_csv(enroll_path, [{"f": 1}, {"f": 2}])
    dsl_df = write_csv(dsl_path, [{"f": 100}])

    synth_df = pd.DataFrame({"f": [999]})

    dp = DataPreprocessor(
        enrollment_csv=str(enroll_path),
        dsl_dataset_csv=str(dsl_path),
        username="user1",
        output_csv=str(output_path)
    )

    mocker.patch.object(
        dp,
        "load_csv",
        side_effect=[enroll_df, synth_df, dsl_df]
    )

    mocker.patch.object(dp, "generate_synthetic", return_value=synth_df)

    combined = dp.build_training_csv()

    assert isinstance(combined, pd.DataFrame)
    assert len(combined) == 4
    assert os.path.exists(output_path)


def test_build_training_csv_missing_dsl(tmp_path, mocker):
    enroll_path = tmp_path / "enroll.csv"
    dsl_path = tmp_path / "missing.csv"
    output_path = tmp_path / "out.csv"

    write_csv(enroll_path, [{"f": 1}])

    dp = DataPreprocessor(
        enrollment_csv=str(enroll_path),
        dsl_dataset_csv=str(dsl_path),
        username="user1",
        output_csv=str(output_path)
    )

    # First call: enrollment CSV
    # Second call: synthetic
    # Third call: DSL -> return None to simulate missing file
    mocker.patch.object(
        dp,
        "load_csv",
        side_effect=[pd.DataFrame({"f":[1]}), pd.DataFrame({"f":[9]}), None]
    )

    combined = dp.build_training_csv()
    assert combined is None
