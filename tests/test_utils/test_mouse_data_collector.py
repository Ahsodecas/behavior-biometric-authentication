import csv
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from src.utils.mouse_data_collector import MouseDataCollector

@pytest.fixture
def tmp_collector(tmp_path):
    """
    Collector with isolated filesystem and deterministic state.
    """
    with patch("src.utils.mouse_data_collector.PATH_COLLECTED", tmp_path):
        collector = MouseDataCollector(username="test_user")
        yield collector


def test_initial_state(tmp_collector):
    c = tmp_collector

    assert c.positions == []
    assert c.clicks == []
    assert c.running is False
    assert c.thread is None
    assert c.rep_counter == 0


def test_start_sets_running_and_thread(tmp_collector):
    c = tmp_collector

    def fake_collect_loop():
        while c.running:
            time.sleep(0.01)

    with patch.object(c, "_collect_loop", side_effect=fake_collect_loop):
        c.start()

        assert c.running is True
        assert c.thread is not None
        assert isinstance(c.thread, threading.Thread)
        assert c.thread.daemon is True

        c.stop()


def test_stop_stops_thread(tmp_collector):
    c = tmp_collector

    def fake_collect_loop():
        while c.running:
            time.sleep(0.01)

    with patch.object(c, "_collect_loop", side_effect=fake_collect_loop):
        c.start()
        c.stop()

        assert c.running is False
        assert not c.thread.is_alive()


def test_save_to_csv_creates_user_directory(tmp_collector, tmp_path):
    c = tmp_collector

    c.positions.append((10, 20, 1.0))
    c.clicks.append((10, 20, "Button.left", True, 2.0))

    c.save_to_csv()

    user_dir = tmp_path / "test_user"
    assert user_dir.exists()
    assert user_dir.is_dir()

    files = list(user_dir.iterdir())
    assert len(files) == 1
    assert files[0].name == "rep_0.csv"

def test_rep_counter_increments(tmp_collector):
    c = tmp_collector

    c.save_to_csv()
    c.save_to_csv()

    assert c.rep_counter == 2

def test_custom_filename_does_not_increment_rep_counter(tmp_collector):
    c = tmp_collector

    c.save_to_csv("custom.csv")

    assert c.rep_counter == 0

def test_save_to_csv_content(tmp_collector, tmp_path):
    c = tmp_collector

    c.positions = [
        (100, 200, 1.1),
        (110, 210, 1.2),
    ]
    c.clicks = [
        (120, 220, "Button.left", True, 1.3),
        (120, 220, "Button.left", False, 1.4),
    ]

    c.save_to_csv("test.csv")

    csv_path = tmp_path / "test_user" / "test.csv"
    assert csv_path.exists()

    with open(csv_path, newline="") as f:
        rows = list(csv.reader(f))

    assert rows[0] == [
        "event_type", "x", "y", "button", "pressed", "timestamp"
    ]

    assert rows[1] == ["move", "100", "200", "", "", "1.1"]
    assert rows[2] == ["move", "110", "210", "", "", "1.2"]
    assert rows[3] == ["click", "120", "220", "Button.left", "True", "1.3"]
    assert rows[4] == ["click", "120", "220", "Button.left", "False", "1.4"]





