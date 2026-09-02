"""The Elman network and the architectures it is measured against.

The comparison is deliberately unfair in the *right* direction: every rival sees
exactly the same window of exactly the same features, at a comparable parameter
budget. The only thing that varies is whether, and how, the model can use the
order of the flows inside the window.

| model | sees the whole window? | can use order? |
|---|---|---|
| `elman` | yes | yes, through the recurrent context |
| `window_mlp` | yes | only positionally, after flattening |
| `window_mlp_wide` | yes | same, with ~5x the parameters |
| `last_flow_mlp` | no, only the final flow | no |

Parameter budgets are matched to the Elman's by default, because a flattened
window is 16x wider than a single step and an unmatched dense layer would win or
lose on capacity rather than on architecture. `window_mlp_wide` is kept as the
deliberately over-provisioned rival: if the Elman still wins against it, capacity
was not the explanation.
"""

from __future__ import annotations

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

from .config import Config, DEFAULT


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    tf.random.set_seed(seed)
    keras.utils.set_random_seed(seed)


def seeded(build):
    """Seed the RNG *before* the weights are created.

    Seeding only inside ``train`` is not enough: Keras draws the initial weights
    when the layers are built, so two runs of the same configuration in one
    process started from different initialisations. On this data that alone moved
    PR-AUC by more than 0.25, which is larger than most of the differences the
    experiment is trying to measure.
    """

    def wrapper(cfg: Config, n_features: int, *args, **kwargs):
        set_seed(cfg.seed)
        return build(cfg, n_features, *args, **kwargs)

    wrapper.__name__ = build.__name__
    wrapper.__doc__ = build.__doc__
    return wrapper


def _compile(model: keras.Model, cfg: Config) -> keras.Model:
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="binary_crossentropy",
        # PR-AUC, not accuracy: with a small minority class, accuracy is a
        # measure of how often the model says "benign".
        metrics=[keras.metrics.AUC(curve="PR", name="pr_auc"),
                 keras.metrics.AUC(name="roc_auc")],
    )
    return model


def build_elman(cfg: Config, n_features: int) -> keras.Model:
    """Elman recurrent network: h_t = tanh(W_x x_t + W_h h_{t-1} + b).

    ``SimpleRNN`` is exactly this recurrence -- the hidden state of the previous
    step is fed back as extra input, which is what Elman's original 'context
    units' do. (A Jordan network feeds back the *output* instead.)
    """
    reg = regularizers.l2(cfg.l2) if cfg.l2 else None
    inp = keras.Input(shape=(cfg.window, n_features), name="flow_window")
    x = layers.SimpleRNN(
        cfg.units,
        activation="tanh",
        kernel_regularizer=reg,
        recurrent_regularizer=reg,
        name="elman_context",
    )(inp)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="attack")(x)
    return _compile(keras.Model(inp, out, name="elman"), cfg)


def elman_param_count(cfg: Config, n_features: int) -> int:
    u = cfg.units
    return u * n_features + u * u + u + u + 1


def _units_for_budget(input_width: int, budget: int) -> int:
    """Hidden units that bring a one-layer dense net to roughly ``budget`` weights."""
    return max(2, round((budget - 1) / (input_width + 2)))


def build_window_mlp(cfg: Config, n_features: int, match_params: bool = True) -> keras.Model:
    """Same window, no recurrence: flatten and use a dense net.

    With ``match_params`` the hidden layer is sized so the model has roughly the
    Elman's parameter count, isolating architecture from capacity.
    """
    reg = regularizers.l2(cfg.l2) if cfg.l2 else None
    width = cfg.window * n_features
    units = (
        _units_for_budget(width, elman_param_count(cfg, n_features))
        if match_params
        else cfg.units
    )
    inp = keras.Input(shape=(cfg.window, n_features), name="flow_window")
    x = layers.Flatten()(inp)
    x = layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="attack")(x)
    name = "window_mlp" if match_params else "window_mlp_wide"
    return _compile(keras.Model(inp, out, name=name), cfg)


def build_window_mlp_wide(cfg: Config, n_features: int) -> keras.Model:
    """The over-provisioned control: same information, far more capacity."""
    return build_window_mlp(cfg, n_features, match_params=False)


def build_last_flow_mlp(cfg: Config, n_features: int, match_params: bool = True) -> keras.Model:
    """Only the final flow of the window -- the 'no context at all' control.

    The gap between this and the two above is the value of looking at more than
    one flow; the gap between the two above is the value of the recurrence.
    """
    reg = regularizers.l2(cfg.l2) if cfg.l2 else None
    units = (
        _units_for_budget(n_features, elman_param_count(cfg, n_features))
        if match_params
        else cfg.units
    )
    inp = keras.Input(shape=(cfg.window, n_features), name="flow_window")
    x = layers.Lambda(lambda t: t[:, -1, :], output_shape=(n_features,))(inp)
    x = layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
    x = layers.Dropout(cfg.dropout)(x)
    out = layers.Dense(1, activation="sigmoid", name="attack")(x)
    return _compile(keras.Model(inp, out, name="last_flow_mlp"), cfg)


BUILDERS = {
    "elman": seeded(build_elman),
    "window_mlp": seeded(build_window_mlp),
    "window_mlp_wide": seeded(build_window_mlp_wide),
    "last_flow_mlp": seeded(build_last_flow_mlp),
}


def train(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    cfg: Config = DEFAULT,
    verbose: int = 0,
) -> dict:
    set_seed(cfg.seed)
    weights = None
    if cfg.class_weight:
        pos = max(int(y_train.sum()), 1)
        neg = max(len(y_train) - pos, 1)
        weights = {0: 1.0, 1: neg / pos}

    hist = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        batch_size=cfg.batch_size,
        epochs=cfg.epochs,
        class_weight=weights,
        callbacks=[
            keras.callbacks.EarlyStopping(
                monitor="val_pr_auc", mode="max",
                patience=cfg.patience, restore_best_weights=True,
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_pr_auc", mode="max", factor=0.5,
                patience=max(3, cfg.patience // 3), min_lr=1e-5,
            ),
        ],
        verbose=verbose,
    )
    return hist.history
