import os

import pytest
import numpy as np
from pytest_mock import MockerFixture

from src import constants
from src.auth.authentication_decision_maker import AuthenticationDecisionMaker


@pytest.fixture
def auth():
    return AuthenticationDecisionMaker(threshold=0.1)


def test_auth_model_not_loaded(auth: AuthenticationDecisionMaker):

    success, dist, msg = auth.authenticate(
        username="user",
        password=".tie5Roanl",
        feature_dict={}
    )

    assert success is False
    assert dist == float("inf")
    assert msg == "Model not loaded."

