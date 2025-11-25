import os
import torch
import numpy as np

class AuthenticationDecisionMaker:
    def __init__(self, model, scaler, ref_sample):
        self.model = model
        self.scaler = scaler
        self.ref_sample = ref_sample
        self.fixed_password = ".tie5Roanl"

# this function is not used for now, password is checked in window class
    def make_decision(self, username, password, sample, threshold):
        if password != self.fixed_password:
            return False, float("inf")

        normalized = self.scaler.transform(sample.reshape(1, -1))[0]
        sample_emb = self.model.embed(normalized)
        ref_emb = self.model.embed(self.ref_sample)
        dist = np.linalg.norm(sample_emb - ref_emb)

        return (dist < threshold), dist
