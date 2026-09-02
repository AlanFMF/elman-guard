"""The experiment harness: train every architecture, score them the same way.

One function runs the whole comparison so that no model can accidentally get a
different split, a different scaler, or a different threshold rule than another.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .config import Config, DEFAULT
from .evaluate import Evaluation, evaluate, per_attack_frame, results_frame, threshold_at_fpr
from .features import FeaturePipeline
from .models import BUILDERS, train
from .sequences import SequenceDataset, shuffle_within_windows


def chronological_validation_split(
    data: SequenceDataset, frac: float = 0.15
) -> tuple[np.ndarray, np.ndarray]:
    """Carve an early-stopping split off the END of the training block.

    Using the calibration day for early stopping looks tidy and is wrong here:
    each weekday of CICIDS2017 contains different attacks, so a validation signal
    taken from another day measures transfer to unseen attack types rather than
    convergence, and training stops on noise. Early stopping therefore watches
    held-out windows from the *training* days; the calibration day is reserved for
    one job only, choosing the alert threshold.
    """
    order = np.argsort(data.end_time, kind="stable")
    n_val = max(1, int(len(order) * frac))
    return order[:-n_val], order[-n_val:]

NEURAL_MODELS = ("elman", "window_mlp", "window_mlp_wide", "last_flow_mlp")


@dataclass
class ExperimentResult:
    results: pd.DataFrame
    per_attack: pd.DataFrame
    scores: dict[str, np.ndarray] = field(default_factory=dict)
    histories: dict[str, dict] = field(default_factory=dict)
    n_train: int = 0
    n_calib: int = 0
    n_test: int = 0
    config: dict = field(default_factory=dict)


def _logistic_scores(Xtr, ytr, Xca, Xte) -> tuple[np.ndarray, np.ndarray]:
    """Linear baseline on the flattened window -- the 'is a net needed?' control."""
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", n_jobs=None)
    clf.fit(Xtr.reshape(len(Xtr), -1), ytr)
    return (
        clf.predict_proba(Xca.reshape(len(Xca), -1))[:, 1],
        clf.predict_proba(Xte.reshape(len(Xte), -1))[:, 1],
    )


def run_experiment(
    train_set: SequenceDataset,
    calib_set: SequenceDataset,
    test_set: SequenceDataset,
    cfg: Config = DEFAULT,
    models: tuple[str, ...] = NEURAL_MODELS,
    include_logistic: bool = True,
    verbose: int = 1,
) -> ExperimentResult:
    fit_idx, val_idx = chronological_validation_split(train_set)
    pipeline = FeaturePipeline(cfg).fit(train_set.X[fit_idx])
    Xtr, Xca, Xte = (pipeline.transform(d.X) for d in (train_set, calib_set, test_set))
    ytr, yca, yte = train_set.y, calib_set.y, test_set.y
    Xfit, yfit, Xval, yval = Xtr[fit_idx], ytr[fit_idx], Xtr[val_idx], ytr[val_idx]
    n_features = Xtr.shape[-1]

    evaluations: list[Evaluation] = []
    scores: dict[str, np.ndarray] = {}
    histories: dict[str, dict] = {}

    for name in models:
        model = BUILDERS[name](cfg, n_features)
        histories[name] = train(model, Xfit, yfit, Xval, yval, cfg, verbose=0)
        s_ca = model.predict(Xca, verbose=0).ravel()
        s_te = model.predict(Xte, verbose=0).ravel()
        thr = threshold_at_fpr(yca, s_ca, cfg.target_fpr)
        ev = evaluate(name, yte, s_te, thr, test_set.attack_type)
        evaluations.append(ev)
        scores[name] = s_te
        if verbose:
            print(f"  {name:16s} PR-AUC={ev.pr_auc:.4f}  recall@{cfg.target_fpr:.0%}FPR="
                  f"{ev.recall:.4f}  F1={ev.f1:.4f}  params={model.count_params():,}")

    if include_logistic:
        s_ca, s_te = _logistic_scores(Xtr, ytr, Xca, Xte)
        thr = threshold_at_fpr(yca, s_ca, cfg.target_fpr)
        ev = evaluate("logistic", yte, s_te, thr, test_set.attack_type)
        evaluations.append(ev)
        scores["logistic"] = s_te
        if verbose:
            print(f"  {'logistic':16s} PR-AUC={ev.pr_auc:.4f}  recall@{cfg.target_fpr:.0%}FPR="
                  f"{ev.recall:.4f}  F1={ev.f1:.4f}")

    # The prevalence floor: what a coin weighted by the base rate would score.
    prior = float(yte.mean())
    evaluations.append(
        Evaluation(model="prevalence_floor", pr_auc=prior, roc_auc=0.5, threshold=float("nan"),
                   precision=prior, recall=1.0, f1=2 * prior / (1 + prior), fpr=1.0,
                   tp=int(yte.sum()), fp=int((1 - yte).sum()), fn=0, tn=0)
    )

    return ExperimentResult(
        results=results_frame(evaluations),
        per_attack=per_attack_frame(evaluations),
        scores=scores,
        histories=histories,
        n_train=len(train_set), n_calib=len(calib_set), n_test=len(test_set),
        config=cfg.to_dict(),
    )


def repeat_over_seeds(
    train_set: SequenceDataset,
    calib_set: SequenceDataset,
    test_set: SequenceDataset,
    cfg: Config = DEFAULT,
    models: tuple[str, ...] = NEURAL_MODELS,
    seeds: tuple[int, ...] = (0, 1, 2),
    include_logistic: bool = True,
    verbose: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the comparison once per seed and report mean and spread.

    Single-run numbers are not evidence here. Re-initialising the same
    architecture on this data moves PR-AUC by more than the gaps between
    architectures, so a table of one-shot scores would mostly rank random seeds.
    """
    import dataclasses

    runs = []
    for seed in seeds:
        if verbose:
            print(f"· seed {seed}")
        res = run_experiment(
            train_set, calib_set, test_set,
            dataclasses.replace(cfg, seed=seed),
            models, include_logistic=include_logistic, verbose=verbose,
        )
        runs.append(res.results.assign(seed=seed))

    allruns = pd.concat(runs)
    cols = ["pr_auc", "roc_auc", "recall", "precision", "f1", "fpr"]
    summary = allruns.groupby(level=0)[cols].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    order = [m for m in list(models) + ["logistic", "prevalence_floor"] if m in summary.index]
    return summary.reindex(order), allruns


def order_ablation(
    train_set: SequenceDataset,
    calib_set: SequenceDataset,
    test_set: SequenceDataset,
    cfg: Config = DEFAULT,
    models: tuple[str, ...] = ("elman", "window_mlp"),
    seeds: tuple[int, ...] = (0, 1, 2),
    verbose: int = 1,
) -> pd.DataFrame:
    """Train on the real order, then on a shuffled order, and compare.

    This is the question the whole project exists to answer. Shuffling the flows
    inside every window destroys temporal structure while preserving the window's
    exact contents. A model that scores the same either way was never using the
    order, and its recurrence is decoration.
    """
    if verbose:
        print("· ordered")
    ordered, _ = repeat_over_seeds(train_set, calib_set, test_set, cfg, models,
                                   seeds, include_logistic=False, verbose=verbose)
    if verbose:
        print("· shuffled within each window")
    shuffled, _ = repeat_over_seeds(
        shuffle_within_windows(train_set, 101),
        shuffle_within_windows(calib_set, 102),
        shuffle_within_windows(test_set, 103),
        cfg, models, seeds, include_logistic=False, verbose=verbose,
    )

    rows = []
    for m in models:
        a, b = ordered.loc[m], shuffled.loc[m]
        rows.append({
            "model": m,
            "pr_auc_ordered": a.pr_auc_mean,
            "pr_auc_ordered_sd": a.pr_auc_std,
            "pr_auc_shuffled": b.pr_auc_mean,
            "pr_auc_shuffled_sd": b.pr_auc_std,
            "pr_auc_drop": a.pr_auc_mean - b.pr_auc_mean,
            "recall_drop": a.recall_mean - b.recall_mean,
        })
    return pd.DataFrame(rows).set_index("model")
