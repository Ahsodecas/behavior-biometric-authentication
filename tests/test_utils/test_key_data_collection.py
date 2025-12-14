# tests/test_data_collector.py

import os
import time
import csv
from unittest import mock
import pytest
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
from src.utils.data_collector import DataCollector
from src.utils.key_stroke_event import KeyStrokeEvent

class DummyQtEvent:
    """A dummy Qt event to simulate key presses/releases."""
    def __init__(self, key, text=None):
        self._key = key
        self._text = text or ""
    def key(self):
        return self._key
    def text(self):
        return self._text

@pytest.fixture
def collector(tmp_path):
    dc = DataCollector(username="test_user")
    dc.data_dir = tmp_path  # Override to avoid writing to real disk
    return dc

def test_initialization(collector):
    assert collector.data == []
    assert collector.rep_counter == 0
    assert collector.username == "test_user"
    assert os.path.exists(collector.data_dir)

def test_start_session(collector):
    collector.start_session()
    assert collector.data == []
    assert collector.session_start_time is not None
    assert collector.last_event_time == collector.session_start_time

def test_collect_key_event(collector):
    collector.start_session()
    event = DummyQtEvent(key=65, text="A")  # Qt Key_A
    collector.collect_key_event(event, "press")

    assert len(collector.data) == 1
    evt = collector.data[0]
    assert isinstance(evt, KeyStrokeEvent)
    assert evt.keysym == '65'
    assert evt.event_type == "press"
    assert evt.key == "A"
    assert evt.timestamp >= collector.session_start_time
    assert evt.session_elapsed_time >= 0

def test_clear_for_next_rep_increments_counter(collector):
    collector.start_session()
    collector.rep_counter = 0

    collector.clear_for_next_rep(failed=False)
    assert collector.data == []
    assert collector.rep_counter == 1

def test_clear_for_next_rep_failed_does_not_increment(collector):
    collector.start_session()
    collector.rep_counter = 2

    collector.clear_for_next_rep(failed=True)
    assert collector.data == []
    assert collector.rep_counter == 2

def test_save_key_raw_csv(collector, tmp_path):
    collector.start_session()
    collector.data.append(KeyStrokeEvent(65, "press", time.time(), 0.1, "A"))
    collector.username = "test_user"

    # Override data_dir to tmp_path
    collector.data_dir = tmp_path

    filename = collector.save_key_raw_csv()

    assert filename is not None
    assert os.path.exists(filename)

    # Check CSV contents
    with open(filename, newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    # Header + 1 event
    assert rows[0] == ["key", "keysym", "event_type", "timestamp", "session_elapsed_time"]
    assert rows[1][0] == "A"
    assert rows[1][1] == "65"
    assert rows[1][2] == "press"
