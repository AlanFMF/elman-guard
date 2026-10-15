#!/usr/bin/env python3
"""Render the figures the READMEs embed, in Portuguese and in English.

    python scripts/make_figures.py

Palette: categorical slots 1-3 (blue, orange, aqua) for the three models, a
single-hue blue ramp for the confusion matrix, red reserved for the prevalence
floor. Gridlines are solid hairlines; marks are thin.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

C = {"surface": "#fcfcfb", "ink": "#0b0b0b", "ink_soft": "#52514e", "grid": "#e8e7e3",
     "model": "#2a78d6", "alt": "#eb6834", "good": "#1baf7a", "bad": "#e34948"}
SERIES = {"elman": "#2a78d6", "window_mlp": "#eb6834", "logistic": "#1baf7a"}
BLUE_RAMP = ["#fcfcfb", "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

T = {
    "en": {
        "models": {"elman": "Elman RNN", "window_mlp": "window MLP",
                   "window_mlp_wide": "window MLP (wide)", "last_flow_mlp": "last-flow MLP",
                   "logistic": "logistic regression"},
        "floor_bar": "prevalence floor — flag everything ({v:.3f})",
        "arch_y": "PR-AUC on the unseen-attack test day\n(mean ± sd over 3 seeds)",
        "protocols": ["random split\n(the usual protocol)", "temporal, embargoed\n(known attack types)",
                      "chronological by day\n(unseen attack types)"],
        "proto_y": "PR-AUC (mean ± sd over 3 seeds)",
        "gap1": "flows shared\nacross the split",
        "gap2": "attack types\nnever seen",
        "floor": "prevalence floor",
        "cm_true": "true attack type", "cm_pred": "predicted",
        "cm_title": "Elman RNN — share of each true class (rows sum to 100%)",
    },
    "pt": {
        "models": {"elman": "Rede de Elman", "window_mlp": "MLP da janela",
                   "window_mlp_wide": "MLP da janela (larga)", "last_flow_mlp": "MLP do último fluxo",
                   "logistic": "regressão logística"},
        "floor_bar": "piso de prevalência — alertar tudo ({v:.3f})",
        "arch_y": "PR-AUC no dia de teste com ataques inéditos\n(média ± desvio em 3 sementes)",
        "protocols": ["split aleatório\n(o protocolo usual)", "temporal com embargo\n(tipos conhecidos)",
                      "cronológico por dia\n(tipos inéditos)"],
        "proto_y": "PR-AUC (média ± desvio em 3 sementes)",
        "gap1": "fluxos compartilhados\nentre treino e teste",
        "gap2": "tipos de ataque\nnunca vistos",
        "floor": "piso de prevalência",
        "cm_true": "tipo de ataque real", "cm_pred": "previsto",
        "cm_title": "Rede de Elman — fração de cada classe real (linhas somam 100%)",
    },
}


def num(v: float, lang: str, digits: int = 2) -> str:
    text = f"{v:.{digits}f}"
    return text.replace(".", ",") if lang == "pt" else text


def spread(values: list[float], min_gap: float) -> list[float]:
    """Nudge label positions apart so close end-points do not print on top of each other."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    out = list(values)
    for a, b in zip(order, order[1:]):
        if out[b] - out[a] < min_gap:
            out[b] = out[a] + min_gap
    return out


def style(ax, grid_axis: str = "y") -> None:
    ax.set_facecolor(C["surface"])
    if grid_axis:
        ax.grid(True, axis=grid_axis, color=C["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(C["grid"])
    ax.tick_params(colors=C["ink_soft"], labelsize=9, length=3, color=C["grid"])
    ax.yaxis.label.set_color(C["ink_soft"])
    ax.xaxis.label.set_color(C["ink_soft"])


def architecture_figure(pooled: pd.DataFrame, out: Path, lang: str) -> None:
    t = T[lang]
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
        ax.annotate(num(m, lang, 3), (xi, m + s + 0.012), ha="center", color=C["ink_soft"], fontsize=9)
    ax.axhline(floor, color=C["bad"], linewidth=1.2, linestyle=":")
    ax.annotate(t["floor_bar"].replace("{v:.3f}", num(floor, lang, 3)), (-0.42, floor + 0.012),
                ha="left", color=C["bad"], fontsize=9)
    ax.set_xticks(x, [t["models"].get(i, i) for i in df.index], rotation=12, ha="right")
    ax.set_ylabel(t["arch_y"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v, lang, 1)))
    ax.set_ylim(0, max(df["pr_auc_mean"] + err) * 1.22)
    fig.tight_layout()
    fig.savefig(out, facecolor=C["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def protocol_figure(frames: list[pd.DataFrame], floors: list[float], out: Path, lang: str) -> None:
    """Slope chart: each model's PR-AUC across the three protocols."""
    t = T[lang]
    fig, ax = plt.subplots(figsize=(9.2, 4.8), dpi=160, facecolor=C["surface"])
    style(ax)
    x = np.arange(3)
    ends, starts = {}, {}
    for model, colour in SERIES.items():
        y = np.array([f.loc[model, "pr_auc_mean"] for f in frames])
        e = np.array([f.loc[model, "pr_auc_std"] for f in frames])
        ax.errorbar(x, y, yerr=e, color=colour, linewidth=2, marker="o", markersize=8,
                    markeredgecolor=C["surface"], markeredgewidth=2, capsize=4,
                    label=t["models"][model])
        starts[model], ends[model] = y[0], y[-1]
    top = max(f.loc[list(SERIES), "pr_auc_mean"].max() for f in frames)
    for side, vals, xpos, dx, ha in (("start", starts, x[0], -12, "right"), ("end", ends, x[-1], 12, "left")):
        keys = list(vals)
        placed = spread([vals[k] for k in keys], min_gap=top * 0.045)
        for k, yv in zip(keys, placed):
            ax.annotate(num(vals[k], lang), (xpos, vals[k]), xytext=(dx, (yv - vals[k]) / top * 260),
                        textcoords="offset points", va="center", ha=ha,
                        color=SERIES[k], fontsize=9, fontweight="bold")
    for xi, fl in zip(x, floors):
        ax.plot([xi - 0.18, xi + 0.18], [fl, fl], color=C["bad"], linewidth=1.4, linestyle=":")
    ax.annotate(t["floor"], (x[-1] + 0.2, floors[-1]), va="center", color=C["bad"], fontsize=8.5)

    for (a, b), label in (((0, 1), t["gap1"]), ((1, 2), t["gap2"])):
        ax.annotate(label, ((a + b) / 2, top * 1.06), ha="center", va="bottom",
                    color=C["ink_soft"], fontsize=8.5, linespacing=1.15)
    ax.set_xticks(x, t["protocols"])
    ax.set_xlim(-0.55, 2.75)
    ax.set_ylim(0, top * 1.26)
    ax.set_ylabel(t["proto_y"])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: num(v, lang, 1)))
    leg = ax.legend(frameon=False, fontsize=9, loc="upper right", ncols=1)
    for txt in leg.get_texts():
        txt.set_color(C["ink_soft"])
    fig.tight_layout()
    fig.savefig(out, facecolor=C["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def confusion_figure(cm: pd.DataFrame, out: Path, lang: str) -> None:
    t = T[lang]
    n = len(cm)
    fig, ax = plt.subplots(figsize=(0.62 * n + 3.2, 0.5 * n + 2.0), dpi=160, facecolor=C["surface"])
    style(ax, grid_axis="")
    cmap = LinearSegmentedColormap.from_list("blue_ramp", BLUE_RAMP)
    ax.imshow(cm.to_numpy(), cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i in range(n):
        for j in range(n):
            v = cm.iat[i, j]
            if v < 0.005:
                continue
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8,
                    color="#ffffff" if v >= 0.5 else C["ink"])
    short = [c.replace("Web Attack - ", "Web: ").replace("DoS ", "DoS ") for c in cm.index]
    ax.set_xticks(range(n), short, rotation=40, ha="right", fontsize=8.5)
    ax.set_yticks(range(n), short, fontsize=8.5)
    ax.set_xlabel(t["cm_pred"])
    ax.set_ylabel(t["cm_true"])
    ax.set_title(t["cm_title"], fontsize=10, color=C["ink_soft"], loc="left")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, facecolor=C["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def floor_of(directory: Path) -> float:
    pooled = pd.read_csv(directory / "pooled_results.csv", index_col=0)
    return float(pooled.loc["prevalence_floor", "pr_auc_mean"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports", type=Path, default=Path("reports"))
    ap.add_argument("--assets", type=Path, default=Path("assets"))
    args = ap.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    names = {"en": {"arch": "architecture_comparison.png", "proto": "protocols.png",
                    "cm": "confusion_matrix.png"},
             "pt": {"arch": "comparacao_arquiteturas.png", "proto": "protocolos.png",
                    "cm": "matriz_confusao.png"}}

    pooled = pd.read_csv(args.reports / "pooled_results.csv", index_col=0)
    dirs = [args.reports / "random_split", args.reports / "temporal_split", args.reports]
    have_ladder = all((d / "pooled_results.csv").exists() for d in dirs[:2])
    cm_path = args.reports / "multiclass" / "confusion_elman.csv"

    for lang in ("pt", "en"):
        architecture_figure(pooled, args.assets / names[lang]["arch"], lang)
        if have_ladder:
            frames = [pd.read_csv(d / "pooled_results.csv", index_col=0) for d in dirs]
            protocol_figure(frames, [floor_of(d) for d in dirs], args.assets / names[lang]["proto"], lang)
        if cm_path.exists():
            confusion_figure(pd.read_csv(cm_path, index_col=0), args.assets / names[lang]["cm"], lang)


if __name__ == "__main__":
    main()
