import os
import torch
import numpy as np

one_time_model = None
one_time_scaler = None
one_time_ref_sample = None

# FIX add user id
class SecurityController:
    def __init__(self, threshold: float):
        # self.user_id = user_id
        self.threshold = threshold

    def update_authentication_threshold(self, value: float):
        self.threshold = value

    def login_user(self, username, password, sample):
        pass

    def logout_user(self):
        pass

    # Will be implemented later as for now we use modes for debugging purposes
    def issue_continuous_authentication(self):
        ...

    def issue_one_time_authentication(self):
        ...
