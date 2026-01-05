import csv
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import pandas as pd

from src.utils.feature_extractor import FeatureExtractor
from src.utils.extracted_features import ExtractedFeatures
from src.utils.key_stroke_event import KeyStrokeEvent


KeyStrokeEvent(
    key_code=65,          # virtual key code
    key_char="a",         # character (can be empty for Shift)
    event_type="press",
    timestamp=1.0,
    session_elapsed=1.0
)


@pytest.fixture
def extractor(tmp_path):
    with patch("src.utils.feature_extractor.FeatureExtractor.FEATURES_DIR", tmp_path):
        fe = FeatureExtractor(
            username="test_user",
            raw_key_data=[],
            rep_counter=0
        )
        yield fe


def test_generate_required_features_order_and_content(extractor):
    import src.constants as constants
    constants.PASSWORD = ".tie5Roanl"

    feats = extractor.generate_required_features()

    assert feats == [
        'H..',
        'DD...t',
        'UD...t',
        'H.t',
        'DD.t.i',
        'UD.t.i',
        'H.i',
        'DD.i.e',
        'UD.i.e',
        'H.e',
        'DD.e.5',
        'UD.e.5',
        'H.5',
        'DD.5.Shift.r',
        'UD.5.Shift.r',
        'H.Shift.r',
        'DD.Shift.r.o',
        'UD.Shift.r.o',
        'H.o',
        'DD.o.a',
        'UD.o.a',
        'H.a',
        'DD.a.n',
        'UD.a.n',
        'H.n',
        'DD.n.l',
        'UD.n.l',
        'H.l'
    ]



def test_create_features_directory_called(tmp_path):
    with patch("src.utils.feature_extractor.FeatureExtractor.FEATURES_DIR", tmp_path):
        FeatureExtractor(username="x")
        assert tmp_path.exists()

def test_extract_key_features_simple_sequence(extractor):
    import src.constants as constants
    constants.PASSWORD = ".ti"

    extractor.raw_key_data = [
        KeyStrokeEvent(190, "press", 0.0, 0.0, "."),
        KeyStrokeEvent(190, "release", 0.1, 0.1, "."),
        KeyStrokeEvent(84, "press", 0.2, 0.2, "t"),
        KeyStrokeEvent(84, "release", 0.3, 0.3, "t"),
        KeyStrokeEvent(73, "press", 0.4, 0.4, "i"),
        KeyStrokeEvent(73, "release", 0.5, 0.5, "i"),
    ]

    extractor.extract_key_features()

    assert extractor.key_features.get_key("H..") == pytest.approx(0.1)
    assert extractor.key_features.get_key("H.t") == pytest.approx(0.1)
    assert extractor.key_features.get_key("H.i") == pytest.approx(0.1)

def test_metadata_written(extractor):
    extractor.raw_key_data = [
        KeyStrokeEvent(65, "press", 1.0, 1.0, "a"),
        KeyStrokeEvent(65, "release", 1.1, 1.1, "a"),
    ]

    extractor.extract_key_features()
    meta = extractor.key_features.metadata

    assert meta["subject"] == "test_user"
    assert meta["rep"] == 0

def test_save_key_features_csv(tmp_path, extractor):
    import src.constants as constants
    constants.PASSWORD = "a"
    extractor.feature_cols = extractor.generate_required_features()

    extractor.raw_key_data = [
        KeyStrokeEvent(65, "press", 1.0, 1.0, "a"),
        KeyStrokeEvent(65, "release", 1.2, 1.2, "a"),
    ]
    extractor.extract_key_features()

    out = tmp_path / "out.csv"

    ok = extractor.save_key_features_csv(filename=str(out))

    assert ok is True
    assert out.exists()

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 2
    assert "H.a" in rows[0]


def test_missing_features_written_as_zero(tmp_path, extractor):
    extractor.raw_key_data = [
        KeyStrokeEvent(65, "press", 1.0, 1.0, "a"),
        KeyStrokeEvent(65, "release", 1.2, 1.2, "a"),
    ]
    extractor.extract_key_features()

    out = tmp_path / "out.csv"
    extractor.save_key_features_csv(filename=out)

    with open(out, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        row = next(reader)

    # All required features must exist (missing ones filled with 0)
    for feat in extractor.feature_cols:
        assert feat in row
def test_append_mode(tmp_path, extractor):
    extractor.raw_key_data = [
        KeyStrokeEvent(65, "press", 1.0, 1.0, "a"),
        KeyStrokeEvent(65, "release", 1.1, 1.1, "a"),
    ]
    extractor.extract_key_features()

    out = tmp_path / "append.csv"

    extractor.save_key_features_csv(filename=out)
    extractor.save_key_features_csv(filename=out, append=True)

    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))

    assert len(rows) == 3  # header + 2 data rows


def test_clear_for_next_rep(extractor):
    extractor.key_features.data["x"] = 1
    extractor.rep_counter = 0

    extractor.clear_for_next_rep()

    assert extractor.key_features.data == {}
    assert extractor.rep_counter == 1

def test_load_csv_key_features_calls_extracted_features(extractor):
    extractor.key_features.load_csv_features_all_rows = MagicMock(
        return_value="loaded_user"
    )

    user = extractor.load_csv_key_features("file.csv")

    assert user == "loaded_user"
    extractor.key_features.load_csv_features_all_rows.assert_called_once()



