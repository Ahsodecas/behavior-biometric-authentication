import time
import csv
import json
import pandas as pd
import numpy as np
from datetime import datetime

class ExtractedFeatures:
    def __init__(self):
        self.features = {}
        self.all_features = []
        self.metadata = {}
        self.data ={}

    def update(self, metadata, features):
        self.metadata = metadata
        self.features = features
        self.all_features.extend(list(features.items()))
        self.data = {**self.metadata, **self.features}

    def get_keys(self):
        return self.data.keys()

    def get_key(self, key):
        return self.data.get(key)

    def get_features(self):
        return self.features

    def clear(self):
        self.features = {}
        self.metadata = {}
        self.data = {}

