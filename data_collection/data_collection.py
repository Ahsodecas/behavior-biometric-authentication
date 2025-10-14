import time
import pandas as pd
import os
import datetime
from pynput import keyboard, mouse

class DataCollector:
    def __init__(self):
        self.events = []

    def record(self, duration_seconds=10, time_sleep=0.005):
        keyboard_listener = keyboard.Listener(on_press=self.on_key_press, on_release=self.on_key_release)
        mouse_listener = mouse.Listener(on_move=self.on_mouse_move, on_click=self.on_mouse_click, on_scroll=self.on_mouse_scroll)

        keyboard_listener.start()
        mouse_listener.start()
        print(f"Recording started for {duration_seconds}s.")
        start = time.time()
        while time.time() - start < duration_seconds:
            time.sleep(time_sleep)

        keyboard_listener.stop()
        mouse_listener.stop()

        df = pd.DataFrame(self.events)
        return df

    def save(self, df, directory="records", filename=None):
        os.makedirs(directory, exist_ok=True)
        if filename is None:
            filename = f"recording_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = os.path.join(directory, filename)
        df.to_csv(path, index=False)
        print(f"Saved to {path}")
        return path

    def on_key_press(self, key):
        try:
            key_str = key.char
        except AttributeError:
            key_str = str(key)

        self.events.append({
            "timestamp": time.time(),
            "device": "keyboard",
            "event": "press",
            "detail": key_str,
            "x": None,
            "y": None
        })

    def on_key_release(self, key):
        try:
            key_str = key.char
        except AttributeError:
            key_str = str(key)

        self.events.append({
            "timestamp": time.time(),
            "device": "keyboard",
            "event": "release",
            "detail": key_str,
            "x": None,
            "y": None
        })

    def on_mouse_move(self, x, y):
        self.events.append({
            "timestamp": time.time(),
            "device": "mouse",
            "event": "move",
            "detail": "",
            "x": x,
            "y": y
        })

    def on_mouse_click(self, x, y, button, pressed):
        btn = str(button).split(".")[-1]  # e.g. "Button.left" → "left"
        event_type = f"{btn}_{'press' if pressed else 'release'}"
        self.events.append({
            "timestamp": time.time(),
            "device": "mouse",
            "event": event_type,
            "detail": "",
            "x": x,
            "y": y
        })

    def on_mouse_scroll(self, x, y, dx, dy):
        self.events.append({
            "timestamp": time.time(),
            "device": "mouse",
            "event": "scroll",
            "detail": f"dx={dx},dy={dy}",
            "x": x,
            "y": y
        })


if __name__ == "__main__":
    data_collector = DataCollector()
    df = data_collector.record(duration_seconds=10)
    data_collector.save(df)
