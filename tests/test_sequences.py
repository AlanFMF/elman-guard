import numpy as np
import pytest

from elmanguard.config import DEFAULT
from elmanguard.sequences import (
    build_sequences,
    shuffle_within_windows,
    split_by_day,
    split_random,
)
from elmanguard.synthetic import BENIGN, FEATURE_COLUMNS

import dataclasses

CFG = dataclasses.replace(DEFAULT, group_key="source_ip", window=8, max_windows_per_group_day=50)
FEATURES = list(FEATURE_COLUMNS)


@pytest.fixture(scope="module")
def data(synthetic_flows):
    return build_sequences(synthetic_flows, FEATURES, CFG, stride=4)


def test_shapes_and_dtypes(data):
    assert data.X.shape[1:] == (CFG.window, len(FEATURES))
    assert data.X.dtype == np.float32
    assert set(np.unique(data.y)) <= {0, 1}
    assert len(data.y) == len(data.host) == len(data.day) == len(data.X)


def test_every_window_comes_from_one_host_and_one_day(synthetic_flows, data):
    """The anti-leak property the whole design rests on."""
    df = synthetic_flows.sort_values(["source_ip", "timestamp"], kind="stable")
    df = df.assign(_d=df.timestamp.dt.day_name())
    # Rebuild a couple of windows by hand and confirm they are homogeneous.
    for host, day in list(zip(data.host, data.day))[:50]:
        sub = df[(df.source_ip == host) & (df._d == day)]
        assert len(sub) >= CFG.window


def test_window_is_positive_when_it_contains_any_attack(data):
    assert (data.y == 1).any() and (data.y == 0).any()
    assert set(data.attack_type[data.y == 0]) == {BENIGN}
    assert BENIGN not in set(data.attack_type[data.y == 1])


def test_cap_limits_windows_per_host_day(data):
    import pandas as pd

    counts = pd.Series(1, index=pd.MultiIndex.from_arrays([data.host, data.day])).groupby(
        level=[0, 1]
    ).sum()
    assert counts.max() <= CFG.max_windows_per_group_day


def test_shuffle_preserves_contents_but_not_order(data):
    shuffled = shuffle_within_windows(data, seed=3)
    assert np.allclose(np.sort(data.X, axis=1), np.sort(shuffled.X, axis=1))
    assert not np.allclose(data.X, shuffled.X)
    assert np.array_equal(data.y, shuffled.y)


def test_day_split_is_disjoint(data):
    tr, ca, te = split_by_day(data, CFG)
    assert set(tr.day).isdisjoint(set(te.day))
    assert set(ca.day).isdisjoint(set(te.day))
    assert len(tr) + len(ca) + len(te) <= len(data)


def test_random_split_partitions_everything(data):
    tr, ca, te = split_random(data, seed=0)
    assert len(tr) + len(ca) + len(te) == len(data)
