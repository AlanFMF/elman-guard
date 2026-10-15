"""The three evaluation protocols, and the guarantees each one claims.

The README makes a quantitative argument -- random splitting inflates scores
partly by leaking raw flows across the boundary. These tests pin down the
mechanics that argument rests on, on synthetic traffic, so it cannot silently
stop being true.
"""

import dataclasses

import numpy as np
import pytest

from elmanguard.config import DEFAULT
from elmanguard.sequences import (
    build_sequences,
    shared_flow_fraction,
    split_random,
    split_temporal,
)
from elmanguard.synthetic import BENIGN, FEATURE_COLUMNS

CFG = dataclasses.replace(DEFAULT, group_key="source_ip", window=8, max_windows_per_group_day=200)


@pytest.fixture(scope="module")
def data(synthetic_flows):
    return build_sequences(synthetic_flows, list(FEATURE_COLUMNS), CFG, stride=4)


def test_windows_carry_their_flow_range(data):
    assert len(data.flow_start) == len(data.flow_end) == len(data)
    assert np.all(data.flow_end - data.flow_start == CFG.window - 1)


def test_temporal_split_shares_no_flow_across_the_boundary(data):
    """The embargo, proven: zero test windows contain a training flow."""
    tr, ca, te = split_temporal(data)
    assert shared_flow_fraction(te, tr) == 0.0
    assert shared_flow_fraction(ca, tr) == 0.0
    assert shared_flow_fraction(te, ca) == 0.0


def test_random_split_does_leak_flows(data):
    """The defect being measured against must actually exist in the baseline."""
    tr, _, te = split_random(data, seed=0)
    assert shared_flow_fraction(te, tr) > 0.5


def test_every_attack_type_reaches_every_temporal_split(data):
    tr, ca, te = split_temporal(data)
    kinds = {k for k in set(data.attack_type) if k != BENIGN}
    for split in (tr, ca, te):
        assert kinds <= set(split.attack_type), "a known-types protocol lost a type"


def test_temporal_split_keeps_test_after_train_within_each_label(data):
    tr, _, te = split_temporal(data)
    for kind in set(te.attack_type):
        for host in set(te.host[te.attack_type == kind]):
            a = tr.flow_end[(tr.attack_type == kind) & (tr.host == host)]
            b = te.flow_start[(te.attack_type == kind) & (te.host == host)]
            if len(a) and len(b):
                assert b.min() > a.max()


def test_cap_by_label_protects_minority_types(synthetic_flows):
    """A proportional cap starves small classes; a per-label cap must not."""
    tight = dataclasses.replace(CFG, max_windows_per_group_day=10)
    proportional = build_sequences(synthetic_flows, list(FEATURE_COLUMNS), tight, stride=1)
    per_label = build_sequences(
        synthetic_flows, list(FEATURE_COLUMNS),
        dataclasses.replace(tight, cap_by_label=True), stride=1,
    )
    attacks = lambda d: int((d.attack_type != BENIGN).sum())  # noqa: E731
    assert attacks(per_label) > attacks(proportional)
