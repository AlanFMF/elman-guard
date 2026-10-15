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
    flow_start: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    flow_end: np.ndarray = field(default_factory=lambda: np.empty(0, dtype=np.int64))
    """Positions of the window's first and last flow in the (host, day, time)
    ordering. Every flow has a unique position, so two windows share a flow
    exactly when these ranges intersect -- which is what the embargo checks."""

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
            flow_start=self.flow_start[mask] if len(self.flow_start) else self.flow_start,
            flow_end=self.flow_end[mask] if len(self.flow_end) else self.flow_end,
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


def _dominant_attack(codes: np.ndarray, benign_code: int, n_codes: int) -> np.ndarray:
    """Most frequent non-benign label code per window row, or ``benign_code``."""
    counts = np.stack([(codes == c).sum(axis=1) for c in range(n_codes)], axis=1)
    counts[:, benign_code] = 0
    dominant = counts.argmax(axis=1)
    return np.where(counts.max(axis=1) > 0, dominant, benign_code)


def _window_index(
    df: pd.DataFrame, cfg: Config, stride: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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

    if cfg.cap_by_label:
        label_codes, label_names = pd.factorize(df["label"].to_numpy())
        benign_code = int(np.flatnonzero(label_names == BENIGN)[0]) if BENIGN in set(label_names) else -1

    def thin(candidates: np.ndarray, cap: int) -> np.ndarray:
        # Thin the origins evenly across the day rather than truncating it,
        # so a busy host still contributes its whole timeline.
        if not cap or len(candidates) <= cap:
            return candidates
        return candidates[np.unique(np.linspace(0, len(candidates) - 1, cap).round().astype(int))]

    blocks, ends, f_start, f_end = [], [], [], []
    offsets = np.arange(cfg.window)
    for seg in segments:
        if len(seg) < max(cfg.window, cfg.min_flows_per_host):
            continue
        starts = np.arange(0, len(seg) - cfg.window + 1, stride)
        cap = cfg.max_windows_per_group_day
        if cfg.cap_by_label and benign_code >= 0:
            # Thin each label separately. A proportional cap would leave a
            # minority attack in a busy host-day with a few dozen windows --
            # fine for detection, fatal for telling attack types apart.
            win_codes = label_codes[order[seg[starts[:, None] + offsets]]]
            dom = _dominant_attack(win_codes, benign_code, len(label_names))
            starts = np.sort(np.concatenate([thin(starts[dom == c], cap) for c in np.unique(dom)]))
        else:
            starts = thin(starts, cap)
        pos = seg[starts[:, None] + offsets]
        rows = order[pos]
        blocks.append(rows)
        ends.append(rows[:, -1])
        f_start.append(pos[:, 0])
        f_end.append(pos[:, -1])

    if not blocks:
        raise ValueError(
            "No sequences produced. Every (host, day) group has fewer flows than "
            f"window={cfg.window}; lower Config.window or check Config.group_key."
        )
    return (np.concatenate(blocks), np.concatenate(ends),
            np.concatenate(f_start), np.concatenate(f_end))


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

    idx, last_row, flow_start, flow_end = _window_index(df, cfg, stride)

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
        flow_start=flow_start.astype(np.int64),
        flow_end=flow_end.astype(np.int64),
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
    # Draw the permutations exactly as before (one rng.permutation per window, in
    # order) so earlier ablation numbers stay reproducible -- but gather them in
    # a single pass instead of copying the whole tensor twice, which on the full
    # capture is the difference between fitting in memory and not.
    order = np.empty((len(data), data.X.shape[1]), dtype=np.intp)
    for i in range(len(order)):
        order[i] = rng.permutation(data.X.shape[1])
    return SequenceDataset(
        X=np.take_along_axis(data.X, order[:, :, None], axis=1),
        y=data.y.copy(),
        attack_type=data.attack_type.copy(),
        host=data.host.copy(),
        day=data.day.copy(),
        end_time=data.end_time.copy(),
        feature_names=list(data.feature_names),
        flow_start=data.flow_start.copy(),
        flow_end=data.flow_end.copy(),
    )


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


def _intersects(
    cand_start: np.ndarray, cand_end: np.ndarray, ref_start: np.ndarray, ref_end: np.ndarray
) -> np.ndarray:
    """For each candidate range, does it intersect any reference range?"""
    if len(ref_start) == 0 or len(cand_start) == 0:
        return np.zeros(len(cand_start), dtype=bool)
    o = np.argsort(ref_start, kind="stable")
    rs, re = ref_start[o], ref_end[o]
    ms, me = [int(rs[0])], [int(re[0])]
    for a, b in zip(rs[1:], re[1:]):
        if a <= me[-1]:
            me[-1] = max(me[-1], int(b))
        else:
            ms.append(int(a)); me.append(int(b))
    ms, me = np.asarray(ms), np.asarray(me)
    i = np.searchsorted(me, cand_start, side="left")
    i_clip = np.minimum(i, len(ms) - 1)
    return (i < len(ms)) & (ms[i_clip] <= cand_end)


def shared_flow_fraction(test: SequenceDataset, train: SequenceDataset) -> float:
    """Share of test windows that contain at least one flow seen in training."""
    if not len(test) or not len(test.flow_start):
        return float("nan")
    hit = _intersects(test.flow_start, test.flow_end, train.flow_start, train.flow_end)
    return float(hit.mean())


def split_temporal(
    data: SequenceDataset, fractions: tuple[float, float] = (0.6, 0.2)
) -> tuple[SequenceDataset, SequenceDataset, SequenceDataset]:
    """Temporal split *within* every (host, day, window label), with a flow embargo.

    The protocol for "attack types the model has already seen". Every attack burst
    is cut in time: its first 60% of windows train, the next 20% set the
    threshold, the last 20% test. Stratifying by label is what guarantees each
    attack type reaches all three splits; cutting in time is what keeps the test
    later than the training data.

    The embargo then removes every calibration or test window that shares even one
    flow with an earlier split. Windows of 16 flows taken every 8 overlap by half,
    so without it a random-looking split leaks raw flows across the boundary --
    which is precisely the defect this protocol exists to measure against.

    What it does not do: it is not chronological across classes. Benign traffic
    from late in a day can train a model tested on an attack from earlier that
    day. The README says so.
    """
    if not len(data.flow_start):
        raise ValueError("This dataset has no flow ranges; rebuild it with build_sequences.")
    keys = pd.DataFrame({"h": data.host, "d": data.day, "l": data.attack_type})
    assign = np.full(len(data), -1, dtype=np.int8)
    for _, idx in keys.groupby(["h", "d", "l"], sort=False).indices.items():
        idx = idx[np.argsort(data.flow_start[idx], kind="stable")]
        a = int(round(len(idx) * fractions[0]))
        b = int(round(len(idx) * (fractions[0] + fractions[1])))
        assign[idx[:a]], assign[idx[a:b]], assign[idx[b:]] = 0, 1, 2

    tr = assign == 0
    ca = assign == 1
    te = assign == 2
    fs, fe = data.flow_start, data.flow_end
    ca &= ~_intersects(fs, fe, fs[tr], fe[tr])
    te &= ~_intersects(fs, fe, fs[tr | ca], fe[tr | ca])
    return data.subset(tr), data.subset(ca), data.subset(te)
