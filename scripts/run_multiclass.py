#!/usr/bin/env python3
"""Multiclass experiment: which attack is it?

    python scripts/run_multiclass.py --processed data/processed --out reports/multiclass

Uses the temporal, embargoed split (see ``elmanguard.multiclass`` for why the
chronological split is undefined for this task) and a per-label window cap so
minority attack types keep enough windows to learn and to score.
"""

from __future__ import annotations

import argparse, dataclasses, glob, json, os, sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from elmanguard.config import DEFAULT  # noqa: E402
from elmanguard.features import select_features  # noqa: E402
from elmanguard.multiclass import (  # noqa: E402
    MC_MODELS, build_label_map, multiclass_order_ablation, repeat_multiclass,
)
from elmanguard.sequences import build_sequences, shared_flow_fraction, split_temporal  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--out", type=Path, default=Path("reports/multiclass"))
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2])
    ap.add_argument("--skip-ablation", action="store_true")
    ap.add_argument("--ablation-only", action="store_true",
                    help="reuse summary.csv from a previous run for the ordered side")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    cfg = dataclasses.replace(DEFAULT, cap_by_label=True)
    files = sorted(glob.glob(str(args.processed / "*.parquet")))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    df = df.sort_values("timestamp", kind="stable").reset_index(drop=True)
    features = select_features(df, cfg)

    print("Building windows (per-label cap) ...")
    windows = build_sequences(df, features, cfg, stride=cfg.stride_train)
    del df
    label_map = build_label_map(windows, cfg.multiclass_min_windows)
    tr, ca, te = split_temporal(windows)
    del windows  # the splits are copies; holding both does not fit in 7 GB

    counts = pd.DataFrame({
        name: pd.Series(d.attack_type).value_counts() for name, d in
        (("train", tr), ("calib", ca), ("test", te))
    }).fillna(0).astype(int)
    counts["in_task"] = counts.index.isin(label_map.classes)
    counts = counts.sort_values("train", ascending=False)
    counts.to_csv(args.out / "class_counts.csv")
    print(counts.to_string())
    print(f"\nclasses in the task ({len(label_map.classes)}): {label_map.classes}")
    print(f"excluded as too rare: {label_map.excluded}")
    print(f"test windows sharing a training flow: {shared_flow_fraction(te, tr):.1%}")

    result = {
        "classes": label_map.classes,
        "excluded": label_map.excluded,
        "windows": {"train": len(tr), "calib": len(ca), "test": len(te)},
        "test_windows_sharing_a_training_flow": shared_flow_fraction(te, tr),
        "config": cfg.to_dict(),
    }
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, default=str))

    if args.ablation_only:
        summary = pd.read_csv(args.out / "summary.csv", index_col=0)
    else:
        print(f"\n=== Multiclass comparison over {len(args.seeds)} seeds ===")
        summary, per_class, first = repeat_multiclass(
            tr, ca, te, label_map, cfg, MC_MODELS, tuple(args.seeds)
        )
        summary.to_csv(args.out / "summary.csv")
        per_class.to_csv(args.out / "per_class_f1.csv")
        print(summary.to_string(float_format=lambda v: f"{v: .4f}"))
        print("\nper-class F1 (mean over seeds):")
        print(per_class.to_string(float_format=lambda v: f"{v: .3f}"))
        for ev in first:
            pd.DataFrame(ev.confusion, index=label_map.classes, columns=label_map.classes) \
                .to_csv(args.out / f"confusion_{ev.model}.csv")
            ev.per_class.to_csv(args.out / f"per_class_{ev.model}_seed0.csv")

    if not args.skip_ablation:
        print("\n=== Does order help name the attack? ===")
        ablation = multiclass_order_ablation(
            tr, ca, te, label_map, cfg, MC_MODELS, tuple(args.seeds),
            ordered=summary.loc[list(MC_MODELS)],
        )
        ablation.to_csv(args.out / "order_ablation.csv")
        print(ablation.to_string(float_format=lambda v: f"{v: .4f}"))

    print(f"\nWrote artefacts to {args.out}/")


if __name__ == "__main__":
    main()
