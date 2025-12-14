import csv
from pathlib import Path
from unittest.mock import patch

import pytest

from src.ml.keystroke_dataset_reader import KeystrokeDatasetReader

@pytest.fixture
def reader(tmp_path):
    """
    Dataset reader with isolated dataset directory.
    """
    r = KeystrokeDatasetReader()
    r.dataset_dir = tmp_path
    return r

@pytest.mark.parametrize(
    "vk,expected",
    [
        ("65", "A"),
        ("90", "Z"),
        ("48", "0"),
        ("57", "9"),
        ("190", "."),
        ("32", "SPACE"),
        ("8", "BACKSPACE"),
        ("999", "UNK"),
        ("abc", "UNK"),
    ],
)
def test_vk_to_token(reader, vk, expected):
    assert reader.vk_to_token(vk) == expected

def create_human_csv(path: Path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vk", "hold", "dd"])
        for r in rows:
            writer.writerow(r)

def test_ignores_non_human_files(reader, tmp_path):
    create_human_csv(tmp_path / "sample_AI.csv", [
        ("65", 0.2, 0.1),
    ])

    hold, dd, ud = reader.load_key_dataset()

    assert hold == []
    assert dd == []
    assert ud == []

def test_load_single_human_file(reader, tmp_path):
    file_path = tmp_path / "user1_HUMAN.csv"
    create_human_csv(file_path, [
        ("65", 0.2, 0.1),  # A
        ("66", 0.4, 0.2),  # B
    ])

    hold, dd, ud = reader.load_key_dataset()

    assert len(hold) == 2
    assert len(dd) == 2
    assert len(ud) == 2

    keys = [k for k, _ in hold]
    assert keys == ["a", "b"]

    # Normalization correctness
    # hold: min=0.2 max=0.4 → denom=0.2
    assert hold[0][1] == 0.0
    assert hold[1][1] == 1.0

    # dd: min=0.1 max=0.2 → denom=0.1
    assert dd[0][1] == 0.0
    assert dd[1][1] == 1.0

    # ud = hold - dd → [0.1, 0.2] → normalized
    assert ud[0][1] == 0.0
    assert ud[1][1] == 1.0

def test_zero_variance_normalization(reader, tmp_path):
    file_path = tmp_path / "user2_HUMAN.csv"
    create_human_csv(file_path, [
        ("65", 0.3, 0.1),
        ("65", 0.3, 0.1),
    ])

    hold, dd, ud = reader.load_key_dataset()

    assert all(v == 0.0 for _, v in hold)
    assert all(v == 0.0 for _, v in dd)
    assert all(v == 0.0 for _, v in ud)


def test_zero_variance_normalization(reader, tmp_path):
    file_path = tmp_path / "user2_HUMAN.csv"
    create_human_csv(file_path, [
        ("65", 0.3, 0.1),
        ("65", 0.3, 0.1),
    ])

    hold, dd, ud = reader.load_key_dataset()

    assert all(v == 0.0 for _, v in hold)
    assert all(v == 0.0 for _, v in dd)
    assert all(v == 0.0 for _, v in ud)


def test_multiple_files_aggregated(reader, tmp_path):
    create_human_csv(tmp_path / "u1_HUMAN.csv", [
        ("65", 0.2, 0.1),
    ])
    create_human_csv(tmp_path / "u2_HUMAN.csv", [
        ("66", 0.4, 0.2),
    ])

    hold, dd, ud = reader.load_key_dataset()

    assert len(hold) == 2
    assert {k for k, _ in hold} == {"a", "b"}


def test_skips_short_rows(reader, tmp_path):
    path = tmp_path / "bad_HUMAN.csv"
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["vk", "hold", "dd"])
        writer.writerow(["65"])
        writer.writerow(["65", "0.2", "0.1"])

    hold, dd, ud = reader.load_key_dataset()

    assert len(hold) == 1
    assert hold[0][0] == "a"