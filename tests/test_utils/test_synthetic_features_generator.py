import pytest
import numpy as np
from src.utils.synthetic_features_generator import SyntheticFeaturesGenerator

@pytest.fixture
def generator():
    g = SyntheticFeaturesGenerator(username="u1")
    return g


@pytest.fixture
def simple_genuine_data():
    """
    Key sequence: a → b → c → d
    Hold times: 0.1, 0.2, 0.3, 0.4
    DD times:    0.01,0.02,0.03
    UD times:    0.11,0.12,0.13
    """
    data = [
        ("H.a", 0.1), ("H.b", 0.2), ("H.c", 0.3), ("H.d", 0.4),
        ("DD.a.b", 0.01), ("DD.b.c", 0.02), ("DD.c.d", 0.03),
        ("UD.a.b", 0.11), ("UD.b.c", 0.12), ("UD.c.d", 0.13),
    ]
    return data


def test_split_genuine_features(generator, simple_genuine_data):
    generator.genuine_features = simple_genuine_data
    hold, dd, ud = generator.split_genuine_features()

    assert len(hold) == 4
    assert len(dd) == 3
    assert len(ud) == 3

    assert hold == [("a", 0.1), ("b", 0.2), ("c", 0.3), ("d", 0.4)]
    assert dd == [("b", 0.01), ("c", 0.02), ("d", 0.03)]
    assert ud == [("b", 0.11), ("c", 0.12), ("d", 0.13)]


def test_fcm_sequence_si_hold_m1(generator, simple_genuine_data):
    generator.genuine_features = simple_genuine_data
    hold, _, _ = generator.split_genuine_features()

    training_U = hold

    # Context order = 1, context = ('a',), looking for 'b'
    S = generator.FCM_sequence_Si(training_U, m=1, context_tuple=("a",), target_key="b")

    assert S == {0.2}

def test_fcm_sequence_si_hold_m2(generator, simple_genuine_data):
    generator.genuine_features = simple_genuine_data
    hold, _, _ = generator.split_genuine_features()

    training_U = hold

    S = generator.FCM_sequence_Si(training_U, m=2, context_tuple=("a","b"), target_key="c")

    assert S == {0.3}


def test_fcm_no_context_match(generator, simple_genuine_data):
    generator.genuine_features = simple_genuine_data
    hold, _, _ = generator.split_genuine_features()

    S = generator.FCM_sequence_Si(hold, m=1, context_tuple=("x",), target_key="b")
    assert S == set()


def test_f_generating_mean(generator):
    Si = {1.0, 2.0, 3.0}
    val = generator.f_generating_func_mean(Si)
    assert val == pytest.approx(2.0)


def test_f_generating_icdf_uniform(generator):
    Si = {1.0, 2.0, 3.0, 4.0}
    samples = [generator.f_generating_func_icdf(Si) for _ in range(200)]

    # All samples must be between min and max because ICDF is monotonic
    assert min(samples) >= 1.0
    assert max(samples) <= 4.0


def test_select_contexts_for_sequence_hold(generator, simple_genuine_data):
    generator.genuine_features = simple_genuine_data
    hold, _, _ = generator.split_genuine_features()

    K = ["a", "b", "c", "d"]
    generator.min_context_cardinality = 1

    decisions = generator.select_contexts_for_sequence(
        training_U=hold,
        K_sequence=K,
        M=2,
        channel="hold"
    )

    assert len(decisions) == 4
    assert decisions[1]["key"] == "b"
    assert decisions[1]["chosen_order"] in {0, 1}
    assert isinstance(decisions[1]["Si"], set)
    assert not decisions[1]["use_random_fallback"]

def test_select_contexts_for_sequence_ud(generator, simple_genuine_data):
    generator.genuine_features = simple_genuine_data
    _, ud, _ = generator.split_genuine_features()

    K = ["a", "b", "c", "d"]
    generator.min_context_cardinality = 1

    decisions = generator.select_contexts_for_sequence(
        training_U=ud,
        K_sequence=K,
        M=2,
        channel="UD"
    )

    assert len(decisions) == 4
    assert decisions[1]["key"] == "b"
    assert decisions[1]["chosen_order"] in {0, 1}
    assert isinstance(decisions[1]["Si"], set)
    assert not decisions[1]["use_random_fallback"]

def test_select_contexts_for_sequence_dd(generator, simple_genuine_data):
    generator.genuine_features = simple_genuine_data
    _, _, dd = generator.split_genuine_features()

    K = ["a", "b", "c", "d"]
    generator.min_context_cardinality = 1

    decisions = generator.select_contexts_for_sequence(
        training_U=dd,
        K_sequence=K,
        M=2,
        channel="DD"
    )

    assert len(decisions) == 4
    assert decisions[1]["key"] == "b"
    assert decisions[1]["chosen_order"] in {0, 1}
    assert isinstance(decisions[1]["Si"], set)
    assert not decisions[1]["use_random_fallback"]


def test_select_contexts_random_fallback(generator, simple_genuine_data):
    generator.genuine_features = simple_genuine_data
    hold, _, _ = generator.split_genuine_features()

    generator.min_context_cardinality = 100

    K = ["a", "b", "c"]
    decisions = generator.select_contexts_for_sequence(
        hold, K, M=2, channel="hold"
    )

    assert all(d["use_random_fallback"] for d in decisions)


def test_generate_features(generator, simple_genuine_data):
    generator.genuine_features = simple_genuine_data
    generator.context_order = 2
    generator.min_context_cardinality = 1

    out = generator.generate()

    assert isinstance(out, dict)
    assert any(k.startswith("H.") for k in out.keys()), "Hold features missing"
    assert any(k.startswith("UD.") for k in out.keys()), "UD features missing"
    assert any(k.startswith("DD.") for k in out.keys()), "DD features missing"

def test_construct_feature_name(generator):
    K = ["a", "b", "c"]

    assert generator.construct_feature_name(K, "hold", 1) == "H.b"
    assert generator.construct_feature_name(K, "DD", 2) == "DD.b.c"
    assert generator.construct_feature_name(K, "UD", 2) == "UD.b.c"
