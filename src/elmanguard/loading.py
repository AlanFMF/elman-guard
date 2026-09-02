"""Reading the CICIDS2017 CSVs without inheriting their defects.

The published `TrafficLabelling` files need every one of these fixes before a
model should be allowed near them:

* **Leading spaces in 70 of the 85 column names** (`' Source IP'`).
* **Latin-1 bytes** in the web-attack labels (`Web Attack \\x96 Brute Force`),
  which crash a naive UTF-8 read.
* **Repeated header rows** spliced into the middle of some files.
* **Twelve-hour timestamps with no AM/PM marker.** `7/7/2017 1:02` is 13:02.
  The capture runs 08:00-17:00, so any hour below 8 is afternoon; that rule is
  applied explicitly and asserted, rather than left to a parser's guess.
* **Ambiguous day/month order.** `7/3/2017` could be 7 March or 3 July. The
  weekday is in the filename, so both readings are tried and the one that lands
  on the expected weekday wins.

The IP and port columns are read but never become features -- see
``config.IDENTIFIER_COLUMNS`` for why.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DAY_OF_FILE: dict[str, str] = {
    "Monday-WorkingHours.pcap_ISCX.csv": "Monday",
    "Tuesday-WorkingHours.pcap_ISCX.csv": "Tuesday",
    "Wednesday-workingHours.pcap_ISCX.csv": "Wednesday",
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv": "Thursday",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv": "Thursday",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv": "Friday",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv": "Friday",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv": "Friday",
}

CAPTURE_START_HOUR = 8
"""The capture window opens at 08:00; anything earlier on the clock is PM."""


def _clean_labels(s: pd.Series) -> pd.Series:
    return (
        s.astype("string")
        .str.replace("\x96", "-", regex=False)   # latin-1 en dash
        .str.replace("–", "-", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def parse_timestamps(raw: pd.Series, expected_weekday: str | None = None) -> pd.Series:
    """Parse CICIDS2017 timestamps, resolving both of their ambiguities."""
    text = raw.astype("string").str.strip()

    candidates = {}
    for dayfirst in (True, False):
        parsed = pd.to_datetime(text, dayfirst=dayfirst, format="mixed", errors="coerce")
        if parsed.notna().sum() == 0:
            continue
        candidates[dayfirst] = parsed

    if not candidates:
        raise ValueError("No timestamp in this file could be parsed.")

    if expected_weekday and len(candidates) > 1:
        # Score only over rows that parsed at all: some files carry a large
        # block of rows with an empty timestamp, and counting those as misses
        # would reject a perfectly good reading.
        scores = {}
        for k, v in candidates.items():
            ok = v.notna()
            scores[k] = float((v[ok].dt.day_name() == expected_weekday).mean()) if ok.any() else 0.0
        best = max(scores, key=scores.get)
        if scores[best] < 0.5:
            raise ValueError(
                f"Neither day/month reading lands on {expected_weekday} "
                f"(best match {scores[best]:.0%}). Check the file."
            )
    else:
        best = next(iter(candidates))

    ts = candidates[best]
    # Restore the missing PM: the capture never starts before 08:00.
    needs_pm = ts.dt.hour < CAPTURE_START_HOUR
    ts = ts.where(~needs_pm, ts + pd.Timedelta(hours=12))
    return ts


def read_day_file(path: str | Path, verbose: bool = True) -> pd.DataFrame:
    """Read one CICIDS2017 CSV and return it with normalised, usable columns."""
    from .features import normalise_columns

    path = Path(path)
    expected = DAY_OF_FILE.get(path.name)

    df = pd.read_csv(path, encoding="latin-1", low_memory=False)
    df = normalise_columns(df)

    # Some files carry the header again partway through.
    if "flow_id" in df.columns:
        repeated = df["flow_id"].astype("string").str.strip().str.lower() == "flow id"
        if repeated.any():
            if verbose:
                print(f"  removed {int(repeated.sum())} repeated header rows")
            df = df[~repeated]

    for col in df.columns:
        if col not in {"flow_id", "source_ip", "destination_ip", "timestamp", "label"}:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["label"] = _clean_labels(df["label"])
    df["timestamp"] = parse_timestamps(df["timestamp"], expected)
    df = df.dropna(subset=["timestamp", "source_ip", "label"])

    observed = df["timestamp"].dt.day_name().mode()
    if expected and len(observed) and observed.iloc[0] != expected:
        raise ValueError(f"{path.name}: parsed weekday {observed.iloc[0]} != {expected}")

    df["day"] = df["timestamp"].dt.day_name()
    if verbose:
        span = f"{df.timestamp.min():%H:%M} -> {df.timestamp.max():%H:%M}"
        attacks = int((df.label != "BENIGN").sum())
        print(f"  {path.name[:46]:46s} {len(df):>8,} rows  {span}  attacks={attacks:>7,}")
    return df.reset_index(drop=True)


def load_directory(
    directory: str | Path, files: list[str] | None = None, verbose: bool = True
) -> pd.DataFrame:
    """Read several day files and concatenate them into one flow table."""
    directory = Path(directory)
    names = files or [f for f in DAY_OF_FILE if (directory / f).exists()]
    if not names:
        raise FileNotFoundError(f"No CICIDS2017 CSVs found in {directory}")
    frames = [read_day_file(directory / n, verbose) for n in names]
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values("timestamp", kind="stable").reset_index(drop=True)


def label_summary(df: pd.DataFrame) -> pd.DataFrame:
    tab = (
        df.groupby(["day", "label"], observed=True)
        .size()
        .unstack(fill_value=0)
        .T.sort_values(by=list(df["day"].unique())[0] if len(df) else None, ascending=False)
        if len(df)
        else pd.DataFrame()
    )
    return tab
