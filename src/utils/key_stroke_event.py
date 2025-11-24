import time
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime

class KeyStrokeEvent:
    def __init__(self, key_code, event_type, timestamp, session_elapsed, key_char = None):
        self.key = key_char or ''
        self.keysym = str(key_code)
        self.event_type = event_type
        self.timestamp = timestamp
        self.session_elapsed_time = round(session_elapsed, 6)
