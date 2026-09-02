import numpy as np
import pandas as pd

from elmanguard.config import DEFAULT
from elmanguard.evaluate import evaluate, threshold_at_fpr, within_host
from elmanguard.features import FeaturePipeline, clean_flows, select_features


def test_identifier_columns_never_become_features():
    df = pd.DataFrame({
        "source_ip": [1, 2], "destination_ip": [1, 2], "source_port": [80, 443],
        "destination_port": [80, 443], "timestamp": [1, 2], "flow_duration": [10, 20],
    })
    feats = select_features(df, DEFAULT)
    assert feats == ["flow_duration"], f"leaky columns survived: {feats}"


def test_destination_port_can_be_re_enabled_explicitly():
    import dataclasses

    df = pd.DataFrame({"destination_port": [80, 443], "flow_duration": [10, 20]})
    assert "destination_port" not in select_features(df, DEFAULT)
    on = dataclasses.replace(DEFAULT, use_destination_port=True)
    assert "destination_port" in select_features(df, on)


def test_clean_flows_removes_inf_and_nan_but_keeps_duplicates():
    df = pd.DataFrame({"a": [1.0, np.inf, 3.0, 3.0], "b": [1.0, 2.0, np.nan, 4.0]})
    out, rep = clean_flows(df, DEFAULT, verbose=False)
    assert len(out) == 2
    assert rep["rows_dropped_duplicate"] == 0


def test_pipeline_is_fitted_on_training_data_only():
    rng = np.random.default_rng(0)
    train = rng.normal(0, 1, (200, 4, 3))
    pipe = FeaturePipeline(DEFAULT).fit(train)
    before = pipe.scaler.mean_.copy()
    pipe.transform(rng.normal(50, 1, (200, 4, 3)))
    assert np.allclose(before, pipe.scaler.mean_), "transform must not refit"


def test_pipeline_clips_outliers():
    train = np.ones((100, 3, 2)) * np.array([1.0, 2.0])
    train += np.random.default_rng(0).normal(0, 0.1, train.shape)
    pipe = FeaturePipeline(DEFAULT).fit(train)
    out = pipe.transform(np.full((1, 3, 2), 1e9))
    assert np.abs(out).max() <= DEFAULT.clip_sigma + 1e-6


def test_threshold_respects_the_false_positive_budget():
    rng = np.random.default_rng(1)
    y = np.r_[np.zeros(2000), np.ones(200)].astype(int)
    scores = np.r_[rng.normal(0, 1, 2000), rng.normal(2.5, 1, 200)]
    thr = threshold_at_fpr(y, scores, 0.01)
    realised = ((scores >= thr) & (y == 0)).sum() / (y == 0).sum()
    assert realised <= 0.01 + 1e-9


def test_evaluation_counts_add_up():
    y = np.array([0, 0, 1, 1])
    ev = evaluate("m", y, np.array([0.1, 0.9, 0.2, 0.8]), 0.5, np.array(["BENIGN"] * 2 + ["X"] * 2))
    assert ev.tp + ev.fp + ev.fn + ev.tn == len(y)
    assert ev.per_attack_recall["X"] == 0.5


def test_within_host_skips_single_class_hosts():
    y = np.r_[np.ones(40), np.zeros(40), np.ones(40)].astype(int)
    scores = np.random.default_rng(0).random(120)
    host = np.array(["mixed"] * 80 + ["all_attack"] * 40)
    out = within_host(y, scores, host, 0.5, min_positive=20, min_negative=20)
    assert list(out.index) == ["mixed"]
