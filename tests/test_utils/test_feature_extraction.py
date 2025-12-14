# tests/test_utils/test_feature_extractor.py

import os
import time
import csv
import json
from unittest import mock
import pytest
import pandas as pd

from src.utils.extracted_features import ExtractedFeatures
from src.utils.feature_extractor import FeatureExtractor

# Dummy raw key events for testing
class DummyKeyEvent:
    def __init__(self, key, event_type, timestamp, key_char=None):
        self.keysym = str(key)
        self.event_type = event_type
        self.timestamp = timestamp
        self.key = key_char or str(key)

    def to_dict(self):
        return {
            "key": self.key,
            "keysym": self.keysym,
            "event_type": self.event_type,
            "timestamp": self.timestamp,
            "session_elapsed_time": 0
        }

@pytest.fixture
def extractor(tmp_path):
    """Return a FeatureExtractor with a temporary features directory."""
    fe = FeatureExtractor(username="test_user", raw_key_data=[])
    fe.features_dir = tmp_path  # override to avoid writing to real disk
    return fe

def test_initialization(extractor):
    assert extractor.username == "test_user"
    assert isinstance(extractor.key_features, ExtractedFeatures)
    assert os.path.exists(extractor.features_dir)

def test_extract_key_features_simple(extractor):
    t0 = time.time()
    extractor.raw_key_data = [
        DummyKeyEvent(65, "press", t0, "A"),
        DummyKeyEvent(65, "release", t0 + 0.1, "A"),
        DummyKeyEvent(66, "press", t0 + 0.2, "B"),
        DummyKeyEvent(66, "release", t0 + 0.25, "B"),
    ]

    extractor.extract_key_features()

    # all_features is a list of feature dicts
    assert isinstance(extractor.key_features.all_features, list)
    assert len(extractor.key_features.all_features) > 0

    features = extractor.key_features.data  # get first feature dict
    assert features["subject"] == "test_user"

    # Check that hold times are created
    assert any(k.startswith("H.") for k in features if k.startswith("H."))
    # Check that DD and UD features exist
    assert any(k.startswith("DD.") or k.startswith("UD.") for k in features)


def test_save_key_features_csv_creates_file(extractor, tmp_path):
    t0 = time.time()
    extractor.raw_key_data = [
        DummyKeyEvent(65, "press", t0, "A"),
        DummyKeyEvent(65, "release", t0 + 0.1, "A"),
    ]
    extractor.extract_key_features()
    extractor.features_dir = tmp_path

    # Mock open to avoid writing real file
    with mock.patch("builtins.open", mock.mock_open()) as m:
        success = extractor.save_key_features_csv("features.csv")
        assert success is True
        m.assert_called_once()  # open called

def test_clear_data_resets_features(extractor):
    # Add dummy feature
    extractor.key_features.update({"subject": "test"}, {"H.A": 0.1})
    assert extractor.key_features.all_features
    extractor.clear_data()
    assert extractor.key_features.all_features == []

def test_clear_for_next_rep_increments_counter(extractor):
    initial_counter = extractor.rep_counter
    extractor.clear_for_next_rep()
    assert extractor.rep_counter == initial_counter + 1
    assert extractor.key_features.all_features == []
