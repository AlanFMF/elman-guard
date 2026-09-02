#!/usr/bin/env python3
"""Render the figures the README embeds.

    python scripts/make_figures.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink_soft": "#52514e", "grid": "#e8e7e3",
     "model": "#2a78d6", "alt": "#eb6834", "good": "#1baf7a", "bad": "#e34948"}
NICE = {"elman": "Elman RNN", "window_mlp": "window MLP", "window_mlp_wide": "window MLP (wide)",
        "last_flow_mlp": "last-flow MLP", "logistic": "logistic regression"}


def style(ax) -> None:
    ax.set_facecolor(C["surface"])
    ax.grid(True, axis="y", color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"])
    ax.tick_params(colors=C["ink_soft"], labelsize=9, length=3, color=C["grid"])
    ax.yaxis.label.set_color(C["ink_soft"])


def architecture_figure(pooled: pd.DataFrame, out: Path) -> None:
    df = pooled[pooled.index != "prevalence_floor"].copy()
    floor = float(pooled.loc["prevalence_floor", "pr_auc_mean"])

    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=160, facecolor=C["surface"])
    style(ax)
    x = np.arange(len(df))
    err = df["pr_auc_std"].fillna(0).to_numpy()
    colours = [C["alt"] if i == "logistic" else C["model"] for i in df.index]
    ax.bar(x, df["pr_auc_mean"], color=colours, width=0.62,
           yerr=err, capsize=5, ecolor=C["ink_soft"], error_kw={"elinewidth": 1.2})
    for xi, (m, s) in enumerate(zip(df["pr_auc_mean"], err)):
        ax.annotate(f"{m:.3f}", (xi, m + s + 0.012), ha="center",
                    color=C["ink_soft"], fontsize=9)
    ax.axhline(floor, color=C["bad"], linewidth=1.2, linestyle=":")
    ax.annotate(f"prevalence floor — flag everything ({floor:.3f})", (-0.42, floor + 0.012),
                ha="left", color=C["bad"], fontsize=9)
    ax.set_xticks(x, [NICE.get(i, i) for i in df.index], rotation=12, ha="right")
    ax.set_ylabel("PR-AUC on the unseen-attack test day\n(mean ± sd over 3 seeds)")
    ax.set_ylim(0, max(df["pr_auc_mean"] + err) * 1.22)
    fig.tight_layout()
    fig.savefig(out, facecolor=C["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def protocol_figure(chrono: pd.DataFrame, random: pd.DataFrame, out: Path) -> None:
    models = [m for m in chrono.index if m != "prevalence_floor" and m in random.index]
    fig, ax = plt.subplots(figsize=(8.6, 4.2), dpi=160, facecolor=C["surface"])
    style(ax)
    x = np.arange(len(models))
    w = 0.36
    a = chrono.loc[models, "pr_auc_mean"]
    b = random.loc[models, "pr_auc_mean"]
    ax.bar(x - w / 2, a, w, color=C["model"], label="chronological split (honest)")
    ax.bar(x + w / 2, b, w, color=C["alt"], label="random split (the usual protocol)")
    for xi, (va, vb) in enumerate(zip(a, b)):
        ax.annotate(f"+{vb - va:.2f}", (xi + w / 2, vb + 0.012), ha="center",
                    color=C["ink_soft"], fontsize=9)
    ax.set_xticks(x, [NICE.get(m, m) for m in models], rotation=12, ha="right")
    ax.set_ylabel("PR-AUC")
    leg = ax.legend(frameon=False, fontsize=9, loc="upper left")
    for t in leg.get_texts():
        t.set_color(C["ink_soft"])
    ax.set_ylim(0, max(b.max(), a.max()) * 1.25)
    fig.tight_layout()
    fig.savefig(out, facecolor=C["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--assets", type=Path, default=Path("assets"))
    args = ap.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)

    pooled = pd.read_csv(args.reports / "pooled_results.csv", index_col=0)
    architecture_figure(pooled, args.assets / "architecture_comparison.png")

    rnd = args.reports / "random_split" / "pooled_results.csv"
    if rnd.exists():
        protocol_figure(pooled, pd.read_csv(rnd, index_col=0),
                        args.assets / "protocol_inflation.png")


if __name__ == "__main__":
    main()
