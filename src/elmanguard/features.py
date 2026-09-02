"""Column hygiene and feature scaling.

The scaler is fitted on the training windows only. That sounds obvious and is
the single most common way a network-intrusion notebook leaks: fitting a scaler
on the whole file lets the test set's distribution influence the training inputs.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from .config import IDENTIFIER_COLUMNS, LEAKY_BY_CONSTRUCTION, Config, DEFAULT


def normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Snake-case the headers.

    The published CICIDS2017 CSVs have leading spaces in most column names
    (`' Flow Duration'`), inconsistent capitalisation, and a stray `'Fwd Header
    Length.1'`. Normalising once here means nothing downstream has to know that.
    """
    out = df.copy()
    out.columns = (
        pd.Index(out.columns)
        .map(str)
        .str.strip()
        .str.replace(r"[/\s\-]+", "_", regex=True)
        .str.replace(r"[^\w]+", "", regex=True)
        .str.replace(r"_+", "_", regex=True)
        .str.strip("_")
        .str.lower()
    )
    return out


def clean_flows(
    df: pd.DataFrame,
    cfg: Config = DEFAULT,
    drop_constant: bool = True,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Drop the rows and columns that make results meaningless.

    CICIDS2017 specifically needs all four of these: infinities in the rate
    columns (division by a zero-duration flow), a handful of NaNs, a large block
    of exactly duplicated rows, and columns that are constant across the whole
    capture.
    """
    report: dict[str, int] = {"rows_in": len(df)}
    out = df.replace([np.inf, -np.inf], np.nan)

    numeric = out.select_dtypes(include=[np.number]).columns
    before = len(out)
    out = out.dropna(subset=list(numeric))
    report["rows_dropped_nan_inf"] = before - len(out)

    report["duplicate_rows_present"] = int(out.duplicated().sum())
    if cfg.drop_duplicate_flows:
        before = len(out)
        out = out.drop_duplicates()
        report["rows_dropped_duplicate"] = before - len(out)
    else:
        report["rows_dropped_duplicate"] = 0

    constant = (
        [c for c in numeric if c in out.columns and out[c].nunique(dropna=False) <= 1]
        if drop_constant
        else []
    )
    out = out.drop(columns=constant)
    report["columns_dropped_constant"] = len(constant)
    report["constant_columns"] = constant
    report["rows_out"] = len(out)

    if verbose:
        print(
            f"  cleaned: {report['rows_in']:,} -> {report['rows_out']:,} rows "
            f"(-{report['rows_dropped_nan_inf']:,} NaN/Inf, "
            f"{report['duplicate_rows_present']:,} duplicates kept); "
            f"dropped {len(constant)} constant columns"
        )
    return out.reset_index(drop=True), report


def select_features(df: pd.DataFrame, cfg: Config = DEFAULT) -> list[str]:
    """Numeric columns that are safe to learn from.

    Everything in ``IDENTIFIER_COLUMNS`` is removed because it identifies *who*
    rather than describes *what*, and in this capture the attacker's address is
    perfectly predictive of the label.
    """
    blocked = set(IDENTIFIER_COLUMNS)
    if not cfg.use_destination_port:
        blocked |= set(LEAKY_BY_CONSTRUCTION)
    numeric = df.select_dtypes(include=[np.number]).columns
    return [c for c in numeric if c not in blocked]


class FeaturePipeline:
    """log1p on heavy-tailed counters, then standardise, then clip outliers."""

    def __init__(self, cfg: Config = DEFAULT) -> None:
        self.cfg = cfg
        self.scaler = StandardScaler()
        self.signed_log = cfg.log1p_transform

    @staticmethod
    def _log(X: np.ndarray) -> np.ndarray:
        # Signed log1p: flow counters span nine orders of magnitude, and a few
        # CICIDS2017 columns legitimately go negative.
        return np.sign(X) * np.log1p(np.abs(X))

    def fit(self, X: np.ndarray) -> "FeaturePipeline":
        flat = X.reshape(-1, X.shape[-1])
        self.scaler.fit(self._log(flat) if self.signed_log else flat)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        shape = X.shape
        flat = X.reshape(-1, shape[-1])
        if self.signed_log:
            flat = self._log(flat)
        out = self.scaler.transform(flat)
        return np.clip(out, -self.cfg.clip_sigma, self.cfg.clip_sigma).reshape(shape).astype(
            np.float32
        )

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        return {
            "mean": self.scaler.mean_.tolist(),
            "scale": self.scaler.scale_.tolist(),
            "signed_log": bool(self.signed_log),
        }

    @classmethod
    def from_dict(cls, d: dict, cfg: Config = DEFAULT) -> "FeaturePipeline":
        obj = cls(cfg)
        obj.scaler.mean_ = np.asarray(d["mean"])
        obj.scaler.scale_ = np.asarray(d["scale"])
        obj.scaler.var_ = obj.scaler.scale_ ** 2
        obj.scaler.n_features_in_ = len(obj.scaler.mean_)
        obj.signed_log = d["signed_log"]
        return obj
