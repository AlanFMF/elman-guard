"""Turning a flat table of flows into per-host sequences.

This module is where the project either works or quietly cheats, so the two
decisions it makes are spelled out.

**Group by host, not by file order.** CICIDS2017 rows are already-aggregated
*flows*, stored roughly in capture order. Sliding a window over that global
ordering interleaves unrelated machines, and a recurrent model then learns
"what time of day is it" -- a property of the capture, not of the traffic. Here
every window contains consecutive flows *from one source host*, so the context
means "what has this host been doing".

**Windows never span a day boundary.** The chronological split is by weekday. If
a window could straddle Wednesday into Thursday, the same flows would appear in
both the training and the evaluation set. Grouping by ``(host, day)`` makes that
impossible by construction rather than by care.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .config import Config, DEFAULT
from .synthetic import BENIGN


@dataclass
class SequenceDataset:
    """Windows of consecutive flows, plus everything needed to audit them."""

    X: np.ndarray                 # (n, window, n_features)
    y: np.ndarray                 # (n,) 1 = window contains attack traffic
    attack_type: np.ndarray       # (n,) dominant non-benign label, or BENIGN
    host: np.ndarray              # (n,)
    day: np.ndarray               # (n,)
    end_time: np.ndarray          # (n,) timestamp of the last flow in the window
    feature_names: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.y)

    @property
    def positive_rate(self) -> float:
        return float(self.y.mean()) if len(self) else 0.0

    def subset(self, mask: np.ndarray) -> "SequenceDataset":
        return SequenceDataset(
            X=self.X[mask],
            y=self.y[mask],
            attack_type=self.attack_type[mask],
            host=self.host[mask],
            day=self.day[mask],
            end_time=self.end_time[mask],
            feature_names=list(self.feature_names),
        )

    def summary(self) -> pd.DataFrame:
        return (
            pd.Series(self.attack_type)
            .value_counts()
            .rename_axis("window label")
            .to_frame("windows")
        )


def _window_label(labels: list[str]) -> tuple[int, str]:
    attacks = [x for x in labels if x != BENIGN]
    if not attacks:
        return 0, BENIGN
    return 1, Counter(attacks).most_common(1)[0][0]


def _window_index(
    df: pd.DataFrame, cfg: Config, stride: int
) -> tuple[np.ndarray, np.ndarray]:
    """Row indices of every window, as an ``(n_windows, window)`` matrix.

    Built as an index rather than by copying slices: the capture is 2.8M rows
    wide by 78 columns, and materialising windows one at a time into a Python
    list costs several gigabytes before ``np.stack`` even runs.
    """
    keys = df[[cfg.group_key, "day"]].copy()
    keys["_pos"] = np.arange(len(df), dtype=np.int64)
    keys["_ts"] = df["timestamp"].to_numpy()
    keys = keys.sort_values([cfg.group_key, "day", "_ts"], kind="stable")

    order = keys["_pos"].to_numpy()
    group = pd.factorize(
        keys[cfg.group_key].astype("string") + "\u0000" + keys["day"].astype("string")
    )[0]
    bounds = np.flatnonzero(np.diff(group)) + 1
    segments = np.split(np.arange(len(order)), bounds)

    blocks, ends = [], []
    offsets = np.arange(cfg.window)
    for seg in segments:
        if len(seg) < max(cfg.window, cfg.min_flows_per_host):
            continue
        starts = np.arange(0, len(seg) - cfg.window + 1, stride)
        cap = cfg.max_windows_per_group_day
        if cap and len(starts) > cap:
            # Thin the origins evenly across the day rather than truncating it,
            # so a busy host still contributes its whole timeline.
            starts = starts[np.unique(np.linspace(0, len(starts) - 1, cap).round().astype(int))]
        rows = order[seg[starts[:, None] + offsets]]
        blocks.append(rows)
        ends.append(rows[:, -1])

    if not blocks:
        raise ValueError(
            "No sequences produced. Every (host, day) group has fewer flows than "
            f"window={cfg.window}; lower Config.window or check Config.group_key."
        )
    return np.concatenate(blocks), np.concatenate(ends)


def build_sequences(
    df: pd.DataFrame,
    feature_names: list[str],
    cfg: Config = DEFAULT,
    stride: int | None = None,
) -> SequenceDataset:
    """Slice ``df`` into ``(host, day)`` windows of ``cfg.window`` consecutive flows."""
    stride = cfg.stride_train if stride is None else stride
    required = {cfg.group_key, "timestamp", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns needed to build sequences: {sorted(missing)}")
    if "day" not in df.columns:
        df = df.assign(day=pd.to_datetime(df["timestamp"]).dt.day_name())

    idx, last_row = _window_index(df, cfg, stride)

    labels = df["label"].to_numpy()
    win_labels = labels[idx]
    is_attack = win_labels != BENIGN
    y = is_attack.any(axis=1).astype(np.int8)

    attack_type = np.full(len(y), BENIGN, dtype=object)
    for i in np.flatnonzero(y):
        attack_type[i] = Counter(win_labels[i][is_attack[i]]).most_common(1)[0][0]

    values = df[feature_names].to_numpy(dtype=np.float32)
    X = values[idx]
    del values

    return SequenceDataset(
        X=X,
        y=y,
        attack_type=attack_type.astype(str),
        host=df[cfg.group_key].to_numpy()[last_row],
        day=df["day"].to_numpy()[last_row],
        end_time=df["timestamp"].to_numpy()[last_row],
        feature_names=list(feature_names),
    )


def split_by_day(
    data: SequenceDataset, cfg: Config = DEFAULT
) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    """Chronological train / calibration / test split by weekday.

    Note what this buys and what it costs: because each weekday of the capture
    contains different attacks, the test set holds attack types the model has
    never seen. That is harder than the usual random split, and it is the setting
    an intrusion detector actually faces.
    """
    def pick(days: tuple[str, ...]) -> SequenceDataset:
        return data.subset(np.isin(data.day, list(days)))

    return pick(cfg.train_days), pick(cfg.calib_days), pick(cfg.test_days)


def shuffle_within_windows(data: SequenceDataset, seed: int = 0) -> SequenceDataset:
    """Randomly permute the flow order inside every window, keeping its contents.

    This is the central ablation of the project. A recurrent model that is really
    using temporal structure must get worse here; one whose score is unchanged was
    only ever reading the window as an unordered bag of flows, and the recurrence
    is decoration.
    """
    rng = np.random.default_rng(seed)
    X = data.X.copy()
    for i in range(len(X)):
        X[i] = X[i][rng.permutation(X.shape[1])]
    out = data.subset(np.ones(len(data), dtype=bool))
    out.X = X
    return out


def split_random(
    data: SequenceDataset, seed: int = 0, fractions: tuple[float, float] = (0.6, 0.2)
) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    """A stratified *random* split of windows -- the protocol to argue against.

    This is what most published CICIDS2017 results use, and it is included here
    precisely so the inflation can be measured. Random splitting puts windows
    from the same host, the same minute and the same attack burst on both sides
    of the divide; overlapping windows share up to ``window - 1`` of their flows
    outright. The score it produces is close to a memorisation test.
    """
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data))
    n_tr = int(len(idx) * fractions[0])
    n_ca = int(len(idx) * fractions[1])
    to_mask = lambda sel: np.isin(np.arange(len(data)), sel)  # noqa: E731
    return (
        data.subset(to_mask(idx[:n_tr])),
        data.subset(to_mask(idx[n_tr : n_tr + n_ca])),
        data.subset(to_mask(idx[n_tr + n_ca :])),
    )
