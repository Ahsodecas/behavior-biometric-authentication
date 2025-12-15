import csv
import pandas as pd
import numpy as np
import pytest

from src.utils.extracted_features import ExtractedFeatures

def test_update_and_getters():
    ef = ExtractedFeatures()

    metadata = {
        "subject": "user1",
        "sessionIndex": 1,
        "rep": 0,
        "generated": 0,
    }
    features = {
        "H.a": 0.2,
        "DD.a.b": 0.4,
    }

    ef.update(metadata, features)

    # metadata + features merged
    assert ef.get_key("subject") == "user1"
    assert ef.get_key("H.a") == 0.2

    # raw feature dict preserved
    assert ef.get_features() == features

    # all_features appended as list of tuples
    assert ("H.a", 0.2) in ef.all_features


def test_clear_resets_all_state():
    ef = ExtractedFeatures()
    ef.update({"subject": "u"}, {"H.a": 1.0})

    ef.clear()

    assert ef.features == {}
    assert ef.metadata == {}
    assert ef.data == {}
    assert ef.all_features == []

def test_load_csv_features_single_row(tmp_path):
    csv_path = tmp_path / "features.csv"

    rows = [
        {
            "subject": "userX",
            "sessionIndex": 1,
            "rep": 0,
            "generated": 0,
            "H.a": 0.25,
            "DD.a.b": 0.5,
        }
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    ef = ExtractedFeatures()
    username = ef.load_csv_features(csv_path)

    assert username == "userX"
    assert ef.get_key("subject") == "userX"
    assert ef.get_key("sessionIndex") == 1
    assert ef.get_key("H.a") == pytest.approx(0.25)
    assert ef.get_key("DD.a.b") == pytest.approx(0.5)

def test_load_csv_features_non_numeric_feature(tmp_path):
    csv_path = tmp_path / "features.csv"

    rows = [
        {
            "subject": "user1",
            "sessionIndex": 1,
            "rep": 0,
            "generated": 0,
            "H.a": "not_a_number",
        }
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    ef = ExtractedFeatures()
    ef.load_csv_features(csv_path)

    assert np.isnan(ef.get_key("H.a"))

def test_load_csv_features_all_rows(tmp_path):
    csv_path = tmp_path / "features_all.csv"

    rows = [
        {
            "subject": "user1",
            "sessionIndex": 1,
            "rep": 0,
            "generated": 0,
            "H.a": 0.1,
        },
        {
            "subject": "user1",
            "sessionIndex": 1,
            "rep": 1,
            "generated": 0,
            "H.a": 0.2,
        },
    ]
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    ef = ExtractedFeatures()
    username = ef.load_csv_features_all_rows(csv_path)

    # last username returned
    assert username == "user1"

    # internal state reflects LAST row
    assert ef.get_key("rep") == 1
    assert ef.get_key("H.a") == pytest.approx(0.2)

    # all_features accumulated
    assert len(ef.all_features) >= 2


def test_load_csv_missing_file():
    ef = ExtractedFeatures()
    result = ef.load_csv_features("does_not_exist.csv")
    assert result is None

    result_all = ef.load_csv_features_all_rows("does_not_exist.csv")
    assert result_all is None
