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
    write_csv(csv_path, [{"a": 1}, {"a": 2}])

    dp = DataPreprocessor(str(csv_path), "user", "out.csv")
    df = dp.load_csv(str(csv_path))

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2


def test_load_csv_missing_file(tmp_path):
    dp = DataPreprocessor(str(tmp_path / "missing.csv"), "user", "out.csv")
    assert dp.load_csv("does_not_exist.csv") is None


def test_generate_synthetic_empty_enrollment(tmp_path, mocker):
    enroll_path = tmp_path / "enroll.csv"
    write_csv(enroll_path, [])

    dp = DataPreprocessor(str(enroll_path), "user", "out.csv")

    mocker.patch.object(dp, "load_csv", return_value=pd.DataFrame())

    df = dp.generate_synthetic()
    assert df.empty



def test_build_training_csv_success(tmp_path, mocker):
    enroll_path = tmp_path / "enroll.csv"
    out_path = tmp_path / "out.csv"

    enroll_df = write_csv(enroll_path, [{"f": 1}, {"f": 2}])
    synth_df = pd.DataFrame({"f": [10]})
    dsl_df = pd.DataFrame({"f": [100]})

    dp = DataPreprocessor(str(enroll_path), "user", str(out_path))

    mocker.patch.object(dp, "load_csv", side_effect=[
        enroll_df,     # enrollment
        synth_df,      # synthetic
        dsl_df         # imposter
    ])

    mocker.patch.object(dp, "generate_synthetic", return_value=synth_df)
    mocker.patch.object(dp, "generate_imposter_synthetic", return_value=dsl_df)

    combined = dp.build_training_csv()

    assert isinstance(combined, pd.DataFrame)
    assert len(combined) == 4
    assert out_path.exists()

def test_build_training_csv_missing_imposter(tmp_path, mocker):
    enroll_path = tmp_path / "enroll.csv"
    write_csv(enroll_path, [{"f": 1}])

    dp = DataPreprocessor(str(enroll_path), "user", str(tmp_path / "out.csv"))

    mocker.patch.object(dp, "load_csv", return_value=pd.DataFrame({"f": [1]}))
    mocker.patch.object(dp, "generate_synthetic", return_value=pd.DataFrame({"f": [9]}))
    mocker.patch.object(dp, "generate_imposter_synthetic", return_value=pd.DataFrame())

    combined = dp.build_training_csv()
    assert combined is not None
