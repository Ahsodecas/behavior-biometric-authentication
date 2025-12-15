# src/utils/mouse_collector.py
import pandas as pd
from pynput import mouse
from threading import Thread
import time
import csv
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(CURRENT_DIR))
PATH_COLLECTED = os.path.join(PROJECT_ROOT, "collected_data")


class MouseDataCollector:
    def __init__(self, username=None):
        self.events = []
        self.running = False
        self.thread = None
        self.username = username or "unknown_user"
        self.rep_counter = 0
        self.data_dir = PATH_COLLECTED
        self.start_time = None  # store experiment start time

    def _collect_loop(self):
        def on_move(x, y):
            timestamp = (time.time() - self.start_time) * 1000  # relative timestamp
            # Check if last event was a press to consider drag
            if self.events and self.events[-1][2] in ("Pressed", "Drag"):
                state = "Drag"
            else:
                state = "Move"
            self.events.append((timestamp, "NoButton", state, x, y))

        def on_click(x, y, button, pressed):
            timestamp = time.time() - self.start_time  # relative timestamp
            state = "Pressed" if pressed else "Released"
            self.events.append((timestamp, str(button), state, x, y))

        with mouse.Listener(on_move=on_move, on_click=on_click) as listener:
            while self.running:
                time.sleep(0.01)
            listener.stop()

    def start(self):
        if not self.running:
            self.start_time = time.time()  # record start of experiment
            self.running = True
            self.thread = Thread(target=self._collect_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def save_to_csv(self, filename=None):
        user_dir = os.path.join(self.data_dir, self.username)
        os.makedirs(user_dir, exist_ok=True)

        if filename is None:
            filename = os.path.join(user_dir, f"rep_{self.rep_counter}.csv")
            self.rep_counter += 1
        else:
            filename = os.path.join(user_dir, filename)

        with open(filename, 'w', newline='') as f:
            writer = csv.writer(f)
            # Header
            writer.writerow(['timestamp', 'button', 'state', 'x', 'y'])
            for event in self.events:
                writer.writerow(event)
        print(f"Saved Mouse Data to {filename}")

    def get_data(self):
        return pd.DataFrame(
            self.events,
            columns=["timestamp", "button", "state", "x", "y"]
        )

    def clear(self):
        self.events = []