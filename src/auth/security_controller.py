import os
import torch
import numpy as np
from src.auth.authentication_decision_maker import AuthenticationDecisionMaker

one_time_model = None
one_time_scaler = None
one_time_ref_sample = None

# FIX add user id
class SecurityController:
    def __init__(self, threshold: float):
        # self.user_id = user_id
        self.threshold = threshold
        self.one_time_decision_maker = AuthenticationDecisionMaker(one_time_model, one_time_scaler, one_time_ref_sample)

    def update_authentication_threshold(self, value: float):
        self.threshold = value

    def login_user(self, username, password, sample):
        return self.one_time_decision_maker.make_decision(
            username=username,
            password=password,
            sample=sample,
            threshold=self.threshold
        )

    def logout_user(self):
        pass

    # Will be implemented later as for now we use modes for debugging purposes
    def issue_continuous_authentication(self):
        ...

    def issue_one_time_authentication(self):
        ...
