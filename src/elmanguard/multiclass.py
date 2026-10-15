"""Multiclass: naming the attack, not just flagging it.

Detection asks "is something wrong in this window?". This module asks the
operator's next question -- "what kind of attack is it?" -- and keeps the same
discipline as the binary task: matched parameter budgets, several seeds, a
linear baseline, and the order-shuffle ablation.

Two choices are forced by the data, and both are stated rather than hidden.

**The protocol is the temporal, embargoed split, not the chronological one.**
In CICIDS2017 every weekday carries different attacks; the Friday test day holds
only types never seen Monday to Wednesday. A classifier cannot name a class it
has never been shown, so the chronological split is undefined here. The temporal
split cuts each attack burst in time and removes every window that shares a flow
across the boundary: the model is tested later in the same burst, on flows it has
never seen.

**Rare types are excluded, and listed.** Heartbleed has 11 flows in the whole
capture, SQL injection 21. A handful of windows can be neither learned nor
scored, so types below ``Config.multiclass_min_windows`` leave the task and the
report says exactly which ones and how many windows they had.

The headline metric is **macro-F1**: every class counts the same, so a model
cannot score well by being right about BENIGN and DoS Hulk alone.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

from .config import Config, DEFAULT
from .features import FeaturePipeline
from .sequences import SequenceDataset, shuffle_within_windows
from .synthetic import BENIGN

MC_MODELS = ("elman", "window_mlp")


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
@dataclass
class LabelMap:
    classes: list[str]
    excluded: dict[str, int]

    def encode(self, data: SequenceDataset) -> tuple[np.ndarray, np.ndarray]:
        """Integer targets, and the mask of windows that belong to the task."""
        index = {c: i for i, c in enumerate(self.classes)}
        keep = np.isin(data.attack_type, self.classes)
        y = np.array([index[a] for a in data.attack_type[keep]], dtype=np.int64)
        return y, keep


def build_label_map(data: SequenceDataset, min_windows: int) -> LabelMap:
    counts = pd.Series(data.attack_type).value_counts()
    attacks = [c for c in counts.index if c != BENIGN]
    kept = sorted(c for c in attacks if counts[c] >= min_windows)
    excluded = {c: int(counts[c]) for c in attacks if counts[c] < min_windows}
    return LabelMap(classes=[BENIGN] + kept, excluded=dict(sorted(excluded.items())))


# --------------------------------------------------------------------------- #
# Models -- same shapes as the binary ones, with a softmax head
# --------------------------------------------------------------------------- #
def _seed(seed: int) -> None:
    from .models import set_seed
    set_seed(seed)


def _compile(model, cfg: Config, n_classes: int):
    from tensorflow import keras

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="categorical_crossentropy",
        # Macro-F1 on the validation split drives early stopping: watching the
        # loss instead would reward getting BENIGN right and little else.
        metrics=[keras.metrics.F1Score(average="macro", name="macro_f1")],
    )
    return model


def build_elman_mc(cfg: Config, n_features: int, n_classes: int):
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    _seed(cfg.seed)
    reg = regularizers.l2(cfg.l2) if cfg.l2 else None
    inp = keras.Input(shape=(cfg.window, n_features), name="flow_window")
    x = layers.SimpleRNN(cfg.units, activation="tanh", kernel_regularizer=reg,
                         recurrent_regularizer=reg, name="elman_context")(inp)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="attack_type")(x)
    return _compile(keras.Model(inp, out, name="elman_mc"), cfg, n_classes)


def elman_mc_params(cfg: Config, n_features: int, n_classes: int) -> int:
    u = cfg.units
    return u * n_features + u * u + u + (u + 1) * n_classes


def build_window_mlp_mc(cfg: Config, n_features: int, n_classes: int):
    from tensorflow import keras
    from tensorflow.keras import layers, regularizers

    _seed(cfg.seed)
    reg = regularizers.l2(cfg.l2) if cfg.l2 else None
    width = cfg.window * n_features
    budget = elman_mc_params(cfg, n_features, n_classes)
    units = max(2, round((budget - n_classes) / (width + 1 + n_classes)))
    inp = keras.Input(shape=(cfg.window, n_features), name="flow_window")
    x = layers.Flatten()(inp)
    x = layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(n_classes, activation="softmax", name="attack_type")(x)
    return _compile(keras.Model(inp, out, name="window_mlp_mc"), cfg, n_classes)


MC_BUILDERS = {"elman": build_elman_mc, "window_mlp": build_window_mlp_mc}


def _balanced_weights(y: np.ndarray, n_classes: int) -> np.ndarray:
    counts = np.bincount(y, minlength=n_classes).astype(float)
    w = len(y) / (n_classes * np.maximum(counts, 1.0))
    return w[y]


def _train(model, Xtr, ytr, Xval, yval, cfg: Config, n_classes: int) -> dict:
    from tensorflow import keras

    _seed(cfg.seed)
    eye = np.eye(n_classes, dtype=np.float32)
    hist = model.fit(
        Xtr, eye[ytr],
        validation_data=(Xval, eye[yval]),
        sample_weight=_balanced_weights(ytr, n_classes),
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_macro_f1", mode="max",
                                          patience=cfg.patience, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(monitor="val_macro_f1", mode="max", factor=0.5,
                                              patience=max(3, cfg.patience // 3), min_lr=1e-5),
        ],
        verbose=0,
    )
    return hist.history


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@dataclass
class MCEvaluation:
    model: str
    macro_f1: float
    balanced_accuracy: float
    weighted_f1: float
    per_class: pd.DataFrame = field(repr=False, default_factory=pd.DataFrame)
    confusion: np.ndarray = field(repr=False, default_factory=lambda: np.empty(0))


def evaluate_mc(name: str, y: np.ndarray, pred: np.ndarray, classes: list[str]) -> MCEvaluation:
    labels = np.arange(len(classes))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = pd.DataFrame(
        {"precision": p, "recall": r, "f1": f, "support": s}, index=classes
    ).rename_axis("attack type")
    cm = confusion_matrix(y, pred, labels=labels, normalize="true")
    return MCEvaluation(
        model=name,
        macro_f1=float(f1_score(y, pred, labels=labels, average="macro", zero_division=0)),
        balanced_accuracy=float(balanced_accuracy_score(y, pred)),
        weighted_f1=float(f1_score(y, pred, labels=labels, average="weighted", zero_division=0)),
        per_class=per_class,
        confusion=np.nan_to_num(cm),
    )


# --------------------------------------------------------------------------- #
# Harness
# --------------------------------------------------------------------------- #
def run_multiclass(
    train_set: SequenceDataset,
    calib_set: SequenceDataset,
    test_set: SequenceDataset,
    label_map: LabelMap,
    cfg: Config = DEFAULT,
    models: tuple[str, ...] = MC_MODELS,
    include_logistic: bool = True,
    verbose: int = 1,
) -> list[MCEvaluation]:
    """Train each model once and score it on the test split.

    The calibration split -- later in each burst than training, same classes --
    is the validation set for early stopping. The test split is touched once.
    """
    ytr, ktr = label_map.encode(train_set)
    yca, kca = label_map.encode(calib_set)
    yte, kte = label_map.encode(test_set)
    pipe = FeaturePipeline(cfg).fit(train_set.X[ktr])
    Xtr = pipe.transform(train_set.X[ktr])
    Xca = pipe.transform(calib_set.X[kca])
    Xte = pipe.transform(test_set.X[kte])
    n_classes, n_features = len(label_map.classes), Xtr.shape[-1]

    out: list[MCEvaluation] = []
    for name in models:
        model = MC_BUILDERS[name](cfg, n_features, n_classes)
        _train(model, Xtr, ytr, Xca, yca, cfg, n_classes)
        pred = model.predict(Xte, verbose=0).argmax(axis=1)
        ev = evaluate_mc(name, yte, pred, label_map.classes)
        out.append(ev)
        if verbose:
            print(f"  {name:12s} macro-F1={ev.macro_f1:.4f}  balanced-acc={ev.balanced_accuracy:.4f}"
                  f"  params={model.count_params():,}")

    if include_logistic:
        clf = LogisticRegression(max_iter=3000, class_weight="balanced")
        clf.fit(Xtr.reshape(len(Xtr), -1), ytr)
        pred = clf.predict(Xte.reshape(len(Xte), -1))
        ev = evaluate_mc("logistic", yte, pred, label_map.classes)
        out.append(ev)
        if verbose:
            print(f"  {'logistic':12s} macro-F1={ev.macro_f1:.4f}  balanced-acc={ev.balanced_accuracy:.4f}")
    return out


def repeat_multiclass(
    train_set, calib_set, test_set, label_map: LabelMap, cfg: Config = DEFAULT,
    models: tuple[str, ...] = MC_MODELS, seeds: tuple[int, ...] = (0, 1, 2),
    include_logistic: bool = True, verbose: int = 1,
):
    """Repeat over seeds. Returns (summary, per-class F1 means, first-seed evaluations)."""
    rows, per_class, first = [], [], None
    for seed in seeds:
        if verbose:
            print(f"· seed {seed}")
        evs = run_multiclass(train_set, calib_set, test_set, label_map,
                             dataclasses.replace(cfg, seed=seed), models,
                             include_logistic=include_logistic and first is None,
                             verbose=verbose)
        first = first or evs
        for ev in evs:
            rows.append({"model": ev.model, "seed": seed, "macro_f1": ev.macro_f1,
                         "balanced_accuracy": ev.balanced_accuracy, "weighted_f1": ev.weighted_f1})
            per_class.append(ev.per_class["f1"].rename(ev.model).to_frame().T.assign(seed=seed))
    runs = pd.DataFrame(rows)
    summary = runs.groupby("model")[["macro_f1", "balanced_accuracy", "weighted_f1"]].agg(["mean", "std"])
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.fillna(0.0)  # logistic: one deterministic fit
    order = [m for m in list(models) + ["logistic"] if m in summary.index]
    pc = pd.concat(per_class).drop(columns="seed").groupby(level=0).mean().T.reindex(columns=order)
    return summary.reindex(order), pc, first


def multiclass_order_ablation(
    train_set, calib_set, test_set, label_map: LabelMap, cfg: Config = DEFAULT,
    models: tuple[str, ...] = MC_MODELS, seeds: tuple[int, ...] = (0, 1, 2), verbose: int = 1,
    ordered: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Does the order of flows help *name* the attack, even if it never helped flag it?

    ``ordered`` may be the summary of a run already made with the same seeds:
    training is seeded before the weights are built, so re-running it would
    reproduce it to the last digit and only cost time.
    """
    if ordered is None:
        ordered, _, _ = repeat_multiclass(train_set, calib_set, test_set, label_map, cfg,
                                          models, seeds, include_logistic=False, verbose=verbose)
    shuffled, _, _ = repeat_multiclass(
        shuffle_within_windows(train_set, 101), shuffle_within_windows(calib_set, 102),
        shuffle_within_windows(test_set, 103), label_map, cfg, models, seeds,
        include_logistic=False, verbose=verbose,
    )
    rows = []
    for m in models:
        a, b = ordered.loc[m], shuffled.loc[m]
        rows.append({
            "model": m,
            "macro_f1_ordered": a.macro_f1_mean, "macro_f1_ordered_sd": a.macro_f1_std,
            "macro_f1_shuffled": b.macro_f1_mean, "macro_f1_shuffled_sd": b.macro_f1_std,
            "macro_f1_drop": a.macro_f1_mean - b.macro_f1_mean,
        })
    return pd.DataFrame(rows).set_index("model")
