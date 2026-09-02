"""Metrics for a detector whose positive class is rare.

Accuracy is excluded on purpose. On this data a model that answers "benign" to
everything scores about 88%, and every percentage point after that is noise. The
numbers here are the ones an operator would actually ask for.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def pr_auc(y: np.ndarray, scores: np.ndarray) -> float:
    """Average precision -- the headline number when positives are scarce."""
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, scores))


def roc_auc(y: np.ndarray, scores: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, scores))


def threshold_at_fpr(y: np.ndarray, scores: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold whose false-positive rate stays within the budget.

    An intrusion detector is deployed against an alert budget, not against a 0.5
    cut-off: an analyst can work through so many false alarms a day and no more.
    The threshold is chosen on the calibration split and then applied unchanged
    to the test split, which is what deployment looks like.
    """
    if len(np.unique(y)) < 2:
        return 0.5
    fpr, _, thresholds = roc_curve(y, scores)
    ok = np.where(fpr <= target_fpr)[0]
    return float(thresholds[ok[-1]]) if len(ok) else float(thresholds[0])


@dataclass
class Evaluation:
    model: str
    pr_auc: float
    roc_auc: float
    threshold: float
    precision: float
    recall: float
    f1: float
    fpr: float
    tp: int
    fp: int
    fn: int
    tn: int
    per_attack_recall: dict[str, float] = field(default_factory=dict)

    def as_row(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "per_attack_recall"}
        return d


def evaluate(
    name: str,
    y: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    attack_type: np.ndarray | None = None,
) -> Evaluation:
    pred = (scores >= threshold).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0

    per_attack: dict[str, float] = {}
    if attack_type is not None:
        for kind in sorted(set(attack_type[y == 1])):
            m = (y == 1) & (attack_type == kind)
            per_attack[kind] = float(pred[m].mean()) if m.any() else float("nan")

    return Evaluation(
        model=name,
        pr_auc=pr_auc(y, scores),
        roc_auc=roc_auc(y, scores),
        threshold=float(threshold),
        precision=precision,
        recall=recall,
        f1=f1,
        fpr=fpr,
        tp=tp, fp=fp, fn=fn, tn=tn,
        per_attack_recall=per_attack,
    )


def results_frame(evaluations: list[Evaluation]) -> pd.DataFrame:
    return pd.DataFrame([e.as_row() for e in evaluations]).set_index("model")


def per_attack_frame(evaluations: list[Evaluation]) -> pd.DataFrame:
    return pd.DataFrame(
        {e.model: e.per_attack_recall for e in evaluations}
    ).rename_axis("attack type")


def within_host(
    y: np.ndarray,
    scores: np.ndarray,
    host: np.ndarray,
    threshold: float,
    min_positive: int = 20,
    min_negative: int = 20,
) -> pd.DataFrame:
    """Metrics computed separately inside each monitored host.

    This is the control that matters on CICIDS2017. Pooled across hosts, a model
    can score well by recognising *which machine* a window came from, because the
    attacked server is almost always under attack and the others almost never
    are. Inside a single host that information is constant, so any separation
    left is the model reading the traffic.

    Hosts without enough of both classes are skipped rather than scored, since a
    single-class group has no discrimination to measure.
    """
    rows = []
    for h in pd.unique(host):
        m = host == h
        yh, sh = y[m], scores[m]
        if yh.sum() < min_positive or (1 - yh).sum() < min_negative:
            continue
        pred = (sh >= threshold).astype(int)
        tp = int(((pred == 1) & (yh == 1)).sum())
        fp = int(((pred == 1) & (yh == 0)).sum())
        fn = int(((pred == 0) & (yh == 1)).sum())
        tn = int(((pred == 0) & (yh == 0)).sum())
        rows.append({
            "host": h,
            "n_windows": int(m.sum()),
            "positive_rate": float(yh.mean()),
            "pr_auc": pr_auc(yh, sh),
            "roc_auc": roc_auc(yh, sh),
            "recall": tp / (tp + fn) if tp + fn else float("nan"),
            "fpr": fp / (fp + tn) if fp + tn else float("nan"),
        })
    return pd.DataFrame(rows).set_index("host") if rows else pd.DataFrame()
