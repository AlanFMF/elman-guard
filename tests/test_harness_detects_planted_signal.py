"""Does the measuring instrument work?

Before trusting a null result on real traffic -- "the recurrence does not help" --
the harness has to be shown capable of finding a temporal effect that is
definitely there. Without this check, a flat ablation table is indistinguishable
from a broken ablation.

The probe is deliberately extreme: windows whose positives are a sorted ramp and
whose negatives are the same numbers shuffled. Every order-blind statistic is
identical between the two classes, so order is the only thing to learn.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from elmanguard.config import DEFAULT
from elmanguard.experiments import chronological_validation_split, order_ablation, run_experiment
from elmanguard.sequences import SequenceDataset, build_sequences, split_random
from elmanguard.synthetic import FEATURE_COLUMNS, order_only_dataset

TINY = dataclasses.replace(
    DEFAULT, group_key="source_ip", window=12, units=64, epochs=60, patience=12,
    max_windows_per_group_day=200, target_fpr=0.05,
)


def _as_dataset(X: np.ndarray, y: np.ndarray, tag: str) -> SequenceDataset:
    n = len(y)
    return SequenceDataset(
        X=X, y=y,
        attack_type=np.where(y == 1, "RAMP", "BENIGN"),
        host=np.full(n, tag), day=np.full(n, "Monday"),
        end_time=np.arange(n),
        feature_names=[f"f{i}" for i in range(X.shape[-1])],
    )


@pytest.fixture(scope="module")
def order_only():
    parts = []
    for i, tag in enumerate(("train", "calib", "test")):
        X, y = order_only_dataset(n=3000 if i == 0 else 1200, window=TINY.window, seed=i)
        parts.append(_as_dataset(X, y, tag))
    return tuple(parts)


@pytest.fixture(scope="module")
def realistic(synthetic_flows):
    data = build_sequences(synthetic_flows, list(FEATURE_COLUMNS), TINY, stride=2)
    return split_random(data, seed=0)


def test_chronological_validation_split_takes_the_tail(realistic):
    train, _, _ = realistic
    fit, val = chronological_validation_split(train, frac=0.2)
    assert len(fit) + len(val) == len(train)
    assert train.end_time[val].min() >= train.end_time[fit].max()


@pytest.mark.slow
def test_elman_learns_a_purely_temporal_signal(order_only):
    """Sanity: the architecture can learn something only order reveals."""
    train, calib, test = order_only
    res = run_experiment(train, calib, test, TINY, models=("elman",),
                         include_logistic=False, verbose=0)
    assert res.results.loc["elman", "pr_auc"] > 0.85


@pytest.mark.slow
def test_order_blind_model_cannot(order_only):
    """Control: a model reading only the last step must be at chance here."""
    train, calib, test = order_only
    res = run_experiment(train, calib, test, TINY, models=("last_flow_mlp",),
                         include_logistic=False, verbose=0)
    assert res.results.loc["last_flow_mlp", "roc_auc"] < 0.60


@pytest.mark.slow
def test_ablation_detects_the_planted_signal(order_only):
    """The headline check: shuffling must visibly cost the Elman."""
    train, calib, test = order_only
    table = order_ablation(train, calib, test, TINY, models=("elman",), verbose=0)
    assert table.loc["elman", "pr_auc_drop"] > 0.20, (
        "the ablation cannot see a temporal effect planted on purpose; "
        "any null result on real data would be uninterpretable"
    )
