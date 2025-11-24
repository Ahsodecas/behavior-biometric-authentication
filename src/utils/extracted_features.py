import time
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime

class ExtractedFeatures:
    def __init__(self):
        self.features = {}
        self.metadata = {}

    def update(self, metadata, features):
        self.metadata = metadata
        self.features = features

    def clear(self):
        self.features = {}
        self.metadata = {}

