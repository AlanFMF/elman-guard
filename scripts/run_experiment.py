#!/usr/bin/env python3
"""Run the full ElmanGuard comparison on the prepared CICIDS2017 data.

    python scripts/run_experiment.py --processed data/processed --out reports

Produces, in order: the pooled comparison of four architectures plus a linear
baseline, the within-host control, the temporal-order ablation, and the
source-vs-destination grouping experiment that measures the attacker-fingerprint
shortcut.

Three evaluation protocols, selected with ``--split``:

* ``chronological`` -- train Mon-Wed, threshold Thu, test Fri. Unseen attack types.
* ``temporal``      -- each attack burst cut in time, with a flow-level embargo.
                       Known attack types, no flow shared across splits.
* ``random``        -- shuffled windows. The common protocol, kept to measure
                       how much it inflates.
"""

from __future__ import annotations

import argparse, dataclasses, glob, json, os, sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from elmanguard.config import DEFAULT, Config  # noqa: E402
from elmanguard.evaluate import within_host  # noqa: E402
from elmanguard.experiments import order_ablation, repeat_over_seeds, run_experiment  # noqa: E402
from elmanguard.features import select_features  # noqa: E402
from elmanguard.experiments import NEURAL_MODELS  # noqa: E402
from elmanguard.sequences import (  # noqa: E402
    build_sequences, shared_flow_fraction, split_by_day, split_random, split_temporal,
)


def load_processed(directory: Path) -> pd.DataFrame:
    files = sorted(glob.glob(str(directory / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"No parquet files in {directory}; run prepare_data.py first.")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return df.sort_values("timestamp", kind="stable").reset_index(drop=True)


SPLITTERS = {
    "chronological": lambda w, cfg: split_by_day(w, cfg),
    "temporal": lambda w, cfg: split_temporal(w),
    "random": lambda w, cfg: split_random(w),
}
OUT_DIRS = {"chronological": "", "temporal": "temporal_split", "random": "random_split"}


def make_splits(df: pd.DataFrame, cfg: Config, features: list[str], how: str = "chronological"):
    windows = build_sequences(df, features, cfg, stride=cfg.stride_train)
    return SPLITTERS[how](windows, cfg)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("reports"))
    ap.add_argument("--skip-grouping-experiment", action="store_true")
    ap.add_argument("--split", choices=list(SPLITTERS), default="chronological")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--models", nargs="*", default=list(NEURAL_MODELS))
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()
    args.out = args.out / OUT_DIRS[args.split] if OUT_DIRS[args.split] else args.out
    args.out.mkdir(parents=True, exist_ok=True)
    models = tuple(args.models)

    cfg = DEFAULT
    print("Loading prepared flows ...")
    df = load_processed(args.processed)
    features = select_features(df, cfg)
    print(f"  {len(df):,} flows · {len(features)} features · "
          f"{df.label.ne('BENIGN').mean():.2%} attack flows")

    print(f"\nBuilding sequences grouped by {cfg.group_key} (window={cfg.window}) ...")
    tr, ca, te = make_splits(df, cfg, features, args.split)
    for name, d in (("train", tr), ("calib", ca), ("test", te)):
        kinds = sorted(set(d.attack_type)) if len(d) else []
        print(f"  {name:6s} {len(d):>7,} windows · {d.positive_rate:6.2%} attack · "
              f"{len(set(d.host))} hosts · types={[k for k in kinds if k != 'BENIGN']}")
    leak = shared_flow_fraction(te, tr)
    print(f"  test windows sharing at least one flow with training: {leak:.1%}")

    print(f"\n=== 1. Pooled comparison over {len(args.seeds)} seeds (all monitored hosts) ===")
    summary, allruns, result = repeat_over_seeds(
        tr, ca, te, cfg, models=models, seeds=tuple(args.seeds), return_first=True
    )
    summary.to_csv(args.out / "pooled_results.csv")
    allruns.to_csv(args.out / "pooled_runs.csv")
    print(summary[["pr_auc_mean", "pr_auc_std", "recall_mean", "recall_std",
                   "precision_mean", "f1_mean"]]
          .to_string(float_format=lambda v: f"{v: .4f}"))

    # The first seed's run supplies the score vectors for the within-host control.
    result.per_attack.to_csv(args.out / "per_attack_recall.csv")
    print("\nrecall by attack type (test day):")
    print(result.per_attack.to_string(float_format=lambda v: f"{v: .3f}"))

    print(f"\n=== 2. Within-host control ===")
    frames = {}
    for model, scores in result.scores.items():
        thr = float(result.results.loc[model, "threshold"])
        wh = within_host(te.y, scores, te.host, thr)
        if len(wh):
            frames[model] = wh
            print(f"  {model}:")
            print(wh.to_string(float_format=lambda v: f"{v: .4f}"))
    if frames:
        pd.concat(frames, names=["model"]).to_csv(args.out / "within_host.csv")

    if not args.skip_ablation:
        print("\n=== 3. Temporal-order ablation ===")
        ablation = order_ablation(tr, ca, te, cfg, seeds=tuple(args.seeds))
        ablation.to_csv(args.out / "order_ablation.csv")
        print(ablation.to_string(float_format=lambda v: f"{v: .4f}"))

    summary = {
        "n_flows": int(len(df)),
        "n_features": len(features),
        "features": features,
        "windows": {"train": len(tr), "calib": len(ca), "test": len(te)},
        "positive_rate": {"train": tr.positive_rate, "calib": ca.positive_rate,
                          "test": te.positive_rate},
        "split": args.split,
        "test_windows_sharing_a_training_flow": leak,
        "attack_types": {"train": sorted(set(tr.attack_type) - {"BENIGN"}),
                         "test": sorted(set(te.attack_type) - {"BENIGN"})},
        "config": cfg.to_dict(),
    }

    if not args.skip_grouping_experiment:
        print("\n=== 4. Grouping by source IP instead: how big is the shortcut? ===")
        shortcut_cfg = dataclasses.replace(cfg, group_key="source_ip")
        s_tr, s_ca, s_te = make_splits(df, shortcut_cfg, features, args.split)
        print(f"  train={len(s_tr):,}  calib={len(s_ca):,}  test={len(s_te):,} windows "
              f"({s_te.positive_rate:.2%} attack)")
        shortcut = run_experiment(s_tr, s_ca, s_te, shortcut_cfg,
                                  models=("elman",), include_logistic=False)
        shortcut.results.to_csv(args.out / "grouping_shortcut.csv")
        print(shortcut.results[["pr_auc", "recall", "precision", "f1"]]
              .to_string(float_format=lambda v: f"{v: .4f}"))
        summary["shortcut_pr_auc"] = float(shortcut.results.loc["elman", "pr_auc"])
        summary["destination_pr_auc"] = float(result.results.loc["elman", "pr_auc"])

    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote artefacts to {args.out}/")


if __name__ == "__main__":
    main()
