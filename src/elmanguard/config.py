"""Central configuration for ElmanGuard.

Every hyper-parameter and every column-hygiene decision lives here, so the
README can quote the exact setup that produced its numbers and a reader can
audit the choices in one file.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


# --------------------------------------------------------------------------- #
# Columns that must never become features.
# --------------------------------------------------------------------------- #
IDENTIFIER_COLUMNS: tuple[str, ...] = (
    "flow_id",
    "source_ip",
    "src_ip",
    "destination_ip",
    "dst_ip",
    "source_port",
    "src_port",
    "timestamp",
    "label",
    "attack_type",
)
"""Dropped from the feature matrix.

In CICIDS2017 the attacks originate from a small set of fixed addresses against
fixed victims. Leave the IPs in and a model reaches ~0.999 F1 by memorising the
attacker's address -- a result that means nothing and that a reviewer will spot
instantly. ``source_ip`` and ``timestamp`` are still *used*, but only to group
and order flows into sequences, never as inputs.
"""

LEAKY_BY_CONSTRUCTION: tuple[str, ...] = ("destination_port",)
"""Destination port is a legitimate signal in production and a near-label here.

In this capture, port 21/22 traffic is almost entirely the brute-force attack and
port 80 scanning is almost entirely PortScan. It is excluded by default; set
``Config.use_destination_port`` to include it and the reports will carry both
numbers so the difference is visible rather than hidden.
"""


@dataclass(frozen=True)
class Config:
    # --- sequence construction --------------------------------------------
    window: int = 16
    """Flows per sequence. The Elman's context spans this many events."""

    stride_train: int = 8
    stride_eval: int = 8
    min_flows_per_host: int = 16

    group_key: str = "destination_ip"
    """Sequences are built per *monitored host*, ordered in time.

    Two decisions in one. First, grouping at all: a sliding window over the
    globally time-sorted file mixes unrelated machines, and the recurrence then
    learns 'what time of day is it' instead of 'what has this host been doing'.

    Second, grouping by *destination* rather than source. In CICIDS2017 almost
    every attack flow originates from the single host 172.16.0.1, whose traffic
    is 99.4% attack. Group by source and the label collapses into "is this the
    attacker's machine" -- and columns like ``init_win_bytes_forward`` and
    ``min_seg_size_forward`` are effectively a TCP-stack fingerprint of that one
    box, so dropping the IP does not remove the shortcut. Grouping by destination
    is the view an operator actually has (you monitor your own servers), and it
    puts benign and attack traffic for the same host in the same population.

    Set to ``"source_ip"`` to reproduce the shortcut and measure how large it is.
    """

    max_windows_per_group_day: int = 1500
    """Cap on windows drawn from one (host, day).

    Without it, DoS Hulk flooding 192.168.10.50 on Wednesday contributes more
    windows than the rest of the capture combined and the model trains on one
    attack. The cap thins the *origins* sampled from a busy host; each window is
    still a contiguous run of real flows.
    """

    cap_by_label: bool = False
    """Apply ``max_windows_per_group_day`` per window label instead of per group.

    Off for detection, so the reported binary numbers stay reproducible. On for
    the multiclass task, where a proportional cap leaves minority attack types
    with too few windows to learn or to score.
    """

    multiclass_min_windows: int = 60
    """Attack types with fewer windows than this are excluded from the
    multiclass task and listed in the report, rather than scored on a handful of
    examples."""

    control_host: str = "192.168.10.50"
    """The server that receives both heavy attack traffic and ordinary traffic.

    Restricting the evaluation to it removes host identity as a usable signal --
    every window comes from the same machine -- so whatever skill survives is
    detection rather than recognition.
    """

    # --- features ----------------------------------------------------------
    use_destination_port: bool = False
    log1p_transform: bool = True
    """Flow counters are heavy-tailed; log1p before standardising."""

    clip_sigma: float = 10.0

    drop_duplicate_flows: bool = False
    """Deliberately off -- and this is a real departure from the usual recipe.

    Advice to de-duplicate CICIDS2017 comes from the *row classification*
    framing, where identical rows split across train and test inflate the score.
    Here the split is chronological and by day, so that overlap cannot happen.
    Worse, de-duplicating would delete the signal: a brute-force attack *is* a
    run of near-identical flows, and a port scan *is* a run of near-identical
    probes. Removing them would erase the very pattern the recurrence exists to
    detect. The count of duplicates is reported instead of hidden.
    """

    # --- model -------------------------------------------------------------
    units: int = 64
    dropout: float = 0.2
    l2: float = 1e-5
    learning_rate: float = 3e-3
    batch_size: int = 256
    epochs: int = 80
    patience: int = 15
    """These are not defaults picked by feel.

    An earlier configuration (48 units, dropout 0.3, lr 1e-3, 60 epochs) scored
    chance -- PR-AUC 0.48 -- on ``synthetic.order_only_dataset``, a probe where
    temporal order is the *only* signal and a plain flattened MLP reaches 0.83.
    Reported on real traffic, that would have produced a confident and completely
    false conclusion: "the recurrence adds nothing". With 64 units and lr 3e-3
    the same network reaches **PR-AUC 0.985** on that probe.

    The lesson is in ``tests/test_harness_detects_planted_signal.py``: a null
    result is only worth reporting once the instrument has been shown able to
    detect a positive one.
    """
    seed: int = 42

    class_weight: bool = True
    """Attack windows are a small minority; weight the loss instead of resampling."""

    # --- evaluation --------------------------------------------------------
    target_fpr: float = 0.01
    """An IDS is judged at a fixed false-positive budget, not at 0.5 threshold."""

    train_days: tuple[str, ...] = ("Monday", "Tuesday", "Wednesday")
    calib_days: tuple[str, ...] = ("Thursday",)
    test_days: tuple[str, ...] = ("Friday",)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT = Config()
