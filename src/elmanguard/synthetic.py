"""A synthetic network-flow generator, shaped like CICIDS2017.

Why this exists
---------------
Two reasons, and neither is "instead of the real data".

1. **The test suite must run anywhere.** Nobody cloning the repository should
   need a 1 GB download to run ``pytest``.
2. **It validates the measuring instrument.** Before trusting an experiment that
   reports "the recurrence does / does not help", you want to know the harness
   can detect a temporal effect *when one is definitely there*. This generator
   plants a known, order-dependent signal; if the ablation cannot find it here,
   the ablation is broken, and any null result on real data is uninterpretable.

The design rule for every attack pattern below: **a single flow in isolation is
ambiguous.** The discriminative information lives in the sequence -- rhythm,
ramping, and repetition -- which is precisely what a recurrent context is for.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLUMNS: tuple[str, ...] = (
    "flow_duration",
    "total_fwd_packets",
    "total_backward_packets",
    "total_length_of_fwd_packets",
    "total_length_of_bwd_packets",
    "fwd_packet_length_max",
    "fwd_packet_length_min",
    "fwd_packet_length_mean",
    "fwd_packet_length_std",
    "bwd_packet_length_mean",
    "flow_bytes_s",
    "flow_packets_s",
    "flow_iat_mean",
    "flow_iat_std",
    "flow_iat_max",
    "flow_iat_min",
    "fwd_iat_mean",
    "syn_flag_count",
    "ack_flag_count",
    "psh_flag_count",
    "rst_flag_count",
    "init_win_bytes_forward",
    "min_seg_size_forward",
    "active_mean",
    "idle_mean",
)

META_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "source_ip",
    "destination_ip",
    "destination_port",
    "protocol",
    "label",
)

BENIGN = "BENIGN"


@dataclass
class SyntheticSpec:
    n_benign_hosts: int = 90
    n_attack_hosts: int = 22
    flows_per_benign_host: tuple[int, int] = (40, 220)
    flows_per_attack_host: tuple[int, int] = (60, 180)
    attack_share_of_host: tuple[float, float] = (0.25, 0.7)
    """Attack hosts also emit benign traffic; the attack is a burst inside it."""
    seed: int = 0


def _benign_flow(rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
    """Ordinary traffic: irregular timing, mixed sizes, mostly completed sessions."""
    fwd = rng.lognormal(1.6, 1.0, n).clip(1, 4000)
    bwd = fwd * rng.uniform(0.4, 2.2, n)
    dur = rng.lognormal(11.5, 1.7, n).clip(50, 1.2e8)          # microseconds
    fwd_len_mean = rng.lognormal(4.4, 1.0, n).clip(0, 1500)
    iat_mean = dur / np.maximum(fwd + bwd, 1)
    return {
        "flow_duration": dur,
        "total_fwd_packets": fwd,
        "total_backward_packets": bwd,
        "total_length_of_fwd_packets": fwd * fwd_len_mean,
        "total_length_of_bwd_packets": bwd * fwd_len_mean * rng.uniform(0.5, 3.0, n),
        "fwd_packet_length_max": fwd_len_mean * rng.uniform(1.0, 3.0, n),
        "fwd_packet_length_min": fwd_len_mean * rng.uniform(0.0, 0.9, n),
        "fwd_packet_length_mean": fwd_len_mean,
        "fwd_packet_length_std": fwd_len_mean * rng.uniform(0.1, 0.9, n),
        "bwd_packet_length_mean": fwd_len_mean * rng.uniform(0.4, 2.5, n),
        "flow_bytes_s": (fwd * fwd_len_mean) / np.maximum(dur / 1e6, 1e-3),
        "flow_packets_s": (fwd + bwd) / np.maximum(dur / 1e6, 1e-3),
        "flow_iat_mean": iat_mean,
        "flow_iat_std": iat_mean * rng.uniform(0.4, 2.5, n),
        "flow_iat_max": iat_mean * rng.uniform(2.0, 12.0, n),
        "flow_iat_min": iat_mean * rng.uniform(0.0, 0.5, n),
        "fwd_iat_mean": iat_mean * rng.uniform(0.7, 1.6, n),
        "syn_flag_count": rng.integers(0, 2, n).astype(float),
        "ack_flag_count": rng.integers(0, 3, n).astype(float),
        "psh_flag_count": rng.integers(0, 3, n).astype(float),
        "rst_flag_count": (rng.random(n) < 0.05).astype(float),
        "init_win_bytes_forward": rng.choice([8192, 29200, 64240, 65535], n).astype(float),
        "min_seg_size_forward": rng.choice([20, 24, 28, 32], n).astype(float),
        "active_mean": dur * rng.uniform(0.05, 0.6, n),
        "idle_mean": dur * rng.uniform(0.0, 0.5, n),
    }


def _apply_portscan(flows: dict, rng, n: int) -> None:
    """Many tiny probes in a tight, accelerating rhythm.

    Each probe on its own looks like an ordinary short connection. What gives it
    away is the cadence: inter-arrival times that shrink monotonically across the
    burst as the scanner speeds up.
    """
    ramp = np.linspace(1.0, 0.12, n)                     # <- the order-dependent part
    flows["total_fwd_packets"][:] = rng.integers(1, 3, n)
    flows["total_backward_packets"][:] = rng.integers(0, 2, n)
    flows["flow_duration"][:] = rng.lognormal(8.0, 0.6, n) * ramp
    flows["fwd_packet_length_mean"][:] = rng.uniform(0, 40, n)
    flows["syn_flag_count"][:] = 1.0
    flows["rst_flag_count"][:] = (rng.random(n) < 0.7).astype(float)
    flows["flow_iat_mean"][:] = flows["flow_duration"] * rng.uniform(0.6, 1.0, n)
    flows["flow_iat_std"][:] = flows["flow_iat_mean"] * 0.05
    flows["init_win_bytes_forward"][:] = 1024.0


def _apply_bruteforce(flows: dict, rng, n: int) -> None:
    """Repeated login attempts: near-identical flows on a metronome.

    The tell is periodicity. One attempt is indistinguishable from a user
    mistyping a password; forty at a fixed interval is not.
    """
    base_dur = rng.lognormal(13.0, 0.2)
    flows["flow_duration"][:] = base_dur * rng.normal(1.0, 0.03, n)
    flows["total_fwd_packets"][:] = rng.normal(11, 0.6, n).clip(6, 20)
    flows["total_backward_packets"][:] = rng.normal(9, 0.6, n).clip(4, 18)
    flows["fwd_packet_length_mean"][:] = rng.normal(58, 2.0, n)
    flows["flow_iat_mean"][:] = base_dur / 12 * rng.normal(1.0, 0.02, n)
    flows["flow_iat_std"][:] = flows["flow_iat_mean"] * 0.02
    flows["psh_flag_count"][:] = 2.0
    # The final attempt succeeds and looks different -- only visible in context.
    flows["total_backward_packets"][-1] *= 4
    flows["flow_duration"][-1] *= 3


def _apply_ddos(flows: dict, rng, n: int) -> None:
    """A rate ramp: volume climbing across the window, not a constant flood."""
    ramp = np.linspace(0.3, 6.0, n)                       # <- the order-dependent part
    flows["total_fwd_packets"][:] = (rng.normal(6, 1.5, n).clip(2, 30)) * ramp
    flows["flow_duration"][:] = rng.lognormal(9.0, 0.4, n) / ramp
    flows["fwd_packet_length_mean"][:] = rng.normal(120, 15, n).clip(20, 400)
    flows["syn_flag_count"][:] = 1.0
    flows["flow_packets_s"][:] = flows["total_fwd_packets"] / np.maximum(
        flows["flow_duration"] / 1e6, 1e-3
    )
    flows["flow_iat_mean"][:] = flows["flow_duration"] / np.maximum(
        flows["total_fwd_packets"], 1
    )
    flows["flow_iat_std"][:] = flows["flow_iat_mean"] * 0.08


ATTACKS = {
    "PortScan": (_apply_portscan, 21.0),
    "FTP-Patator": (_apply_bruteforce, 900.0),
    "DDoS": (_apply_ddos, 55.0),
}


def _recompute_derived(flows: dict) -> None:
    """Keep the derived rate columns consistent after a pattern rewrites counters."""
    dur_s = np.maximum(flows["flow_duration"] / 1e6, 1e-6)
    fwd, bwd = flows["total_fwd_packets"], flows["total_backward_packets"]
    mean_len = flows["fwd_packet_length_mean"]
    flows["total_length_of_fwd_packets"] = fwd * mean_len
    flows["flow_bytes_s"] = flows["total_length_of_fwd_packets"] / dur_s
    flows["flow_packets_s"] = (fwd + bwd) / dur_s
    flows["fwd_packet_length_max"] = mean_len * 1.6
    flows["fwd_packet_length_min"] = mean_len * 0.4


def generate(spec: SyntheticSpec | None = None, days: int = 5) -> pd.DataFrame:
    """Build a labelled flow table with the CICIDS2017 column layout."""
    spec = spec or SyntheticSpec()
    rng = np.random.default_rng(spec.seed)
    rows: list[pd.DataFrame] = []
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"][:days]

    for d, day in enumerate(day_names):
        day_start = pd.Timestamp("2017-07-03") + pd.Timedelta(days=d) + pd.Timedelta(hours=8)
        # Attack mix changes across days, exactly as the real capture does --
        # which is what makes the chronological split a real generalisation test.
        todays_attacks = list(ATTACKS)[: 1 + (d % len(ATTACKS))]

        for h in range(spec.n_benign_hosts):
            n = int(rng.integers(*spec.flows_per_benign_host))
            flows = _benign_flow(rng, n)
            _recompute_derived(flows)
            rows.append(_assemble(flows, rng, day_start, f"192.168.{d}.{h + 2}", [BENIGN] * n))

        for h in range(spec.n_attack_hosts):
            n = int(rng.integers(*spec.flows_per_attack_host))
            flows = _benign_flow(rng, n)
            attack = str(rng.choice(todays_attacks))
            fn, port = ATTACKS[attack]

            share = rng.uniform(*spec.attack_share_of_host)
            burst = max(spec.n_attack_hosts // 2, int(n * share))
            start = int(rng.integers(0, max(1, n - burst)))
            sl = slice(start, start + burst)

            sub = {k: v[sl].copy() for k, v in flows.items()}
            fn(sub, rng, len(sub["flow_duration"]))
            _recompute_derived(sub)
            for k in flows:
                flows[k][sl] = sub[k]

            labels = [BENIGN] * n
            labels[sl] = [attack] * burst
            ports = np.where(np.arange(n) >= start, port, 0)
            rows.append(
                _assemble(flows, rng, day_start, f"172.16.{d}.{h + 2}", labels, forced_port=ports)
            )

    df = pd.concat(rows, ignore_index=True)
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


def _assemble(
    flows: dict,
    rng: np.random.Generator,
    day_start: pd.Timestamp,
    src_ip: str,
    labels: list[str],
    forced_port: np.ndarray | None = None,
) -> pd.DataFrame:
    n = len(labels)
    gaps = rng.exponential(9.0, n).cumsum()
    ts = day_start + pd.to_timedelta(gaps, unit="s")
    port = (
        forced_port
        if forced_port is not None
        else rng.choice([80, 443, 53, 22, 8080, 3389], n)
    )
    port = np.where(port == 0, rng.choice([80, 443, 53], n), port)
    data = {c: np.asarray(flows[c], dtype=float) for c in FEATURE_COLUMNS}
    data.update(
        timestamp=ts,
        source_ip=src_ip,
        destination_ip=f"192.168.10.{rng.integers(2, 50)}",
        destination_port=port.astype(int),
        protocol=6,
        label=labels,
    )
    return pd.DataFrame(data)


def order_only_dataset(
    n: int = 4000, window: int = 12, n_features: int = 4, rho: float = 0.9, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    """A dataset in which temporal order is the *only* signal.

    Every window is generated by the same AR(1) process, so smooth runs appear in
    both classes' raw material. The negatives are then permuted in time. The two
    classes therefore share an *identical multiset of values* window by window --
    same mean, variance, min, max, and even the same distribution for the last
    step -- and differ only in whether consecutive steps are correlated.

    Any order-blind model is at chance by construction, and shuffling the windows
    must erase the signal completely. This is the probe used to check that the
    ablation harness can see a temporal effect at all, before any null result on
    real traffic is believed.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(size=(n, window, n_features))
    X = np.empty_like(noise)
    X[:, 0] = noise[:, 0]
    for t in range(1, window):
        X[:, t] = rho * X[:, t - 1] + np.sqrt(1 - rho ** 2) * noise[:, t]

    y = (rng.random(n) < 0.5).astype(np.int8)
    for i in np.flatnonzero(y == 0):
        X[i] = X[i][rng.permutation(window)]      # same values, order destroyed
    return X.astype(np.float32), y
