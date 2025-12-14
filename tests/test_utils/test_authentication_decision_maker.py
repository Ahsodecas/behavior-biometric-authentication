import pytest
import numpy as np
from pytest_mock import MockerFixture

from src.auth.authentication_decision_maker import AuthenticationDecisionMaker


@pytest.fixture
def auth():
    return AuthenticationDecisionMaker(password_fixed=".tie5Roanl", threshold=0.4)

"""
def test_load_model_success(auth: AuthenticationDecisionMaker, mocker: MockerFixture):

    mocker.patch("os.path.exists", return_value=True)

    fake_dataset = mocker.MagicMock()
    fake_dataset.X = np.random.randn(50, 10)
    fake_dataset.scaler = mocker.MagicMock()
    fake_dataset.feature_cols = [f"f{i}" for i in range(10)]

    mocker.patch(
        "src.auth.authentication_decision_maker.CMUDatasetTriplet",
        return_value=fake_dataset
    )

    fake_model = mocker.MagicMock()
    mocker.patch(
        "src.auth.authentication_decision_maker.TripletSNN",
        return_value=fake_model
    )

    mocker.patch(
        "src.auth.authentication_decision_maker.torch.load",
        return_value={"model_state": None}
    )

    auth.load_model("checkpoint.ckpt")

    assert auth.model is fake_model
    assert auth.scaler is fake_dataset.scaler
    assert auth.feature_cols == fake_dataset.feature_cols
    assert auth.ref_sample.shape[0] == 10
"""


def test_load_model_missing(auth: AuthenticationDecisionMaker, mocker: MockerFixture):

    mocker.patch("os.path.exists", return_value=False)

    with pytest.raises(FileNotFoundError):
        auth.load_model("missing.ckpt")

"""
def test_load_model_too_few_samples(auth: AuthenticationDecisionMaker, mocker: MockerFixture):

    mocker.patch("os.path.exists", return_value=True)

    fake_dataset = mocker.MagicMock()
    fake_dataset.X = np.random.randn(10, 8)  # too few
    fake_dataset.scaler = mocker.MagicMock()
    fake_dataset.feature_cols = [f"f{i}" for i in range(8)]

    mocker.patch(
        "src.auth.authentication_decision_maker.CMUDatasetTriplet",
        return_value=fake_dataset
    )

    with pytest.raises(ValueError):
        auth.load_model("checkpoint.ckpt")
"""

def test_auth_wrong_password(auth: AuthenticationDecisionMaker):

    success, dist, msg = auth.authenticate(
        username="user",
        password="wrong",
        feature_dict={}
    )

    assert success is False
    assert dist == float("inf")
    assert msg == "Incorrect password."


def test_auth_model_not_loaded(auth: AuthenticationDecisionMaker):

    success, dist, msg = auth.authenticate(
        username="user",
        password=".tie5Roanl",
        feature_dict={}
    )

    assert success is False
    assert dist == float("inf")
    assert msg == "Model not loaded."



def test_auth_datetime_conversion(auth: AuthenticationDecisionMaker, mocker: MockerFixture):

    auth.model = mocker.MagicMock()
    auth.scaler = mocker.MagicMock()
    auth.feature_cols = ["hold"]

    auth.scaler.transform.return_value = np.array([[0.1]])

    mocker.patch.object(auth, "compute_distance", return_value=0.1)

    success, dist, msg = auth.authenticate(
        username="user",
        password=".tie5Roanl",
        feature_dict={"hold": "0.0825465"}
    )

    assert success is True
    assert msg == "Authenticated successfully."


def test_auth_scaler_failure(auth: AuthenticationDecisionMaker, mocker: MockerFixture):

    auth.model = mocker.MagicMock()
    auth.scaler = mocker.MagicMock()
    auth.feature_cols = ["x"]

    auth.scaler.transform.side_effect = Exception("scaler error")

    success, dist, msg = auth.authenticate(
        username="user",
        password=".tie5Roanl",
        feature_dict={"x": 5}
    )

    assert success is False
    assert "Scaler transform failed" in msg


def test_auth_success(auth: AuthenticationDecisionMaker, mocker: MockerFixture):

    auth.model = mocker.MagicMock()
    auth.scaler = mocker.MagicMock()
    auth.feature_cols = ["a"]

    auth.scaler.transform.return_value = np.array([[0.0]])

    mocker.patch.object(auth, "compute_distance", return_value=0.1)

    success, dist, msg = auth.authenticate(
        username="u",
        password=".tie5Roanl",
        feature_dict={"a": 1}
    )

    assert success is True
    assert msg == "Authenticated successfully."


def test_auth_failure_distance(auth: AuthenticationDecisionMaker, mocker: MockerFixture):

    auth.model = mocker.MagicMock()
    auth.scaler = mocker.MagicMock()
    auth.feature_cols = ["a"]

    auth.scaler.transform.return_value = np.array([[0.0]])

    mocker.patch.object(auth, "compute_distance", return_value=1.0)

    success, dist, msg = auth.authenticate(
        username="u",
        password=".tie5Roanl",
        feature_dict={"a": 1}
    )

    assert success is False
    assert msg == "Authentication failed."
