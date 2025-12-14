# src/utils/mouse_collector.py
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
        self.positions = []
        self.clicks = []
        self.running = False
        self.thread = None
        self.username = username or "unknown_user"
        self.rep_counter = 0
        self.data_dir = PATH_COLLECTED

    def _collect_loop(self):
        def on_move(x, y):
            self.positions.append((x, y, time.time()))

        def on_click(x, y, button, pressed):
            self.clicks.append((x, y, str(button), pressed, time.time()))

        with mouse.Listener(on_move=on_move, on_click=on_click) as listener:
            while self.running:
                time.sleep(0.1)
            listener.stop()

    def start(self):
        if not self.running:
            self.running = True
            self.thread = Thread(target=self._collect_loop, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()

    def save_to_csv(self, filename=None):
        """
        Save both positions and clicks into a single CSV file.
        Each row is either a move or a click, with an event_type column.
        """
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
            writer.writerow(['event_type', 'x', 'y', 'button', 'pressed', 'timestamp'])
            # Write positions
            for x, y, ts in self.positions:
                writer.writerow(['move', x, y, '', '', ts])
            # Write clicks
            for x, y, button, pressed, ts in self.clicks:
                writer.writerow(['click', x, y, button, pressed, ts])
