import dataclasses

import numpy as np
import pytest

from elmanguard.config import DEFAULT
from elmanguard.multiclass import (
    build_label_map,
    elman_mc_params,
    evaluate_mc,
    run_multiclass,
)
from elmanguard.sequences import build_sequences, split_temporal
from elmanguard.synthetic import BENIGN, FEATURE_COLUMNS

CFG = dataclasses.replace(
    DEFAULT, group_key="source_ip", window=8, units=16, epochs=4, patience=2,
    max_windows_per_group_day=200, cap_by_label=True,
)


@pytest.fixture(scope="module")
def data(synthetic_flows):
    return build_sequences(synthetic_flows, list(FEATURE_COLUMNS), CFG, stride=4)


def test_label_map_puts_benign_first_and_excludes_rare_types(data):
    lm = build_label_map(data, min_windows=10**9)
    assert lm.classes == [BENIGN]
    assert set(lm.excluded) == set(data.attack_type) - {BENIGN}


def test_encode_drops_only_excluded_windows(data):
    lm = build_label_map(data, min_windows=1)
    y, keep = lm.encode(data)
    assert keep.all() and len(y) == len(data)
    assert set(np.unique(y)) <= set(range(len(lm.classes)))


def test_macro_f1_weights_classes_equally():
    """Being perfect on the big class and wrong on the small one must hurt."""
    y = np.array([0] * 98 + [1] * 2)
    pred = np.zeros(100, dtype=int)
    ev = evaluate_mc("m", y, pred, ["BENIGN", "rare"])
    assert ev.weighted_f1 > 0.95
    assert ev.macro_f1 < 0.55


def test_confusion_rows_are_normalised():
    y = np.array([0, 0, 1, 1, 2, 2])
    ev = evaluate_mc("m", y, np.array([0, 1, 1, 1, 2, 0]), ["a", "b", "c"])
    assert np.allclose(ev.confusion.sum(axis=1), 1.0)


@pytest.mark.parametrize("n_features, n_classes", [(25, 4), (78, 12)])
def test_param_budget_matches_to_within_half_a_hidden_unit(n_features, n_classes):
    """Budgets can only match to the granularity of one hidden unit; nearest
    rounding must land within half of that."""
    from elmanguard.multiclass import build_window_mlp_mc

    cfg = dataclasses.replace(DEFAULT, window=16) if n_features == 78 else CFG
    budget = elman_mc_params(cfg, n_features, n_classes)
    mlp = build_window_mlp_mc(cfg, n_features, n_classes)
    one_unit = cfg.window * n_features + 1 + n_classes
    assert abs(mlp.count_params() - budget) <= one_unit / 2 + n_classes


@pytest.mark.slow
def test_multiclass_end_to_end_beats_chance(data):
    lm = build_label_map(data, min_windows=20)
    tr, ca, te = split_temporal(data)
    evs = run_multiclass(tr, ca, te, lm, CFG, models=("elman",), include_logistic=False, verbose=0)
    chance = 1.0 / len(lm.classes)
    assert evs[0].balanced_accuracy > chance + 0.15
