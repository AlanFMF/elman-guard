#!/usr/bin/env python3
"""Train the shipped model and export a small sample so the app runs offline.

    python scripts/export_demo.py

Writes: models/elman.keras, models/artifacts.json, data/demo_windows.npz
"""

from __future__ import annotations

import argparse, glob, json, os, sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from elmanguard.config import DEFAULT  # noqa: E402
from elmanguard.evaluate import threshold_at_fpr  # noqa: E402
from elmanguard.experiments import chronological_validation_split  # noqa: E402
from elmanguard.features import FeaturePipeline, select_features  # noqa: E402
from elmanguard.models import BUILDERS, train  # noqa: E402
from elmanguard.sequences import build_sequences, split_by_day  # noqa: E402

DEMO_HOSTS = ["192.168.10.50", "192.168.10.9", "192.168.10.15", "205.174.165.73"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=Path, default=Path("data/processed"))
    ap.add_argument("--models", type=Path, default=Path("models"))
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--arch", default="elman")
    args = ap.parse_args()
    args.models.mkdir(parents=True, exist_ok=True)

    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(str(args.processed / "*.parquet")))],
                   ignore_index=True).sort_values("timestamp", kind="stable")
    features = select_features(df, DEFAULT)
    tr, ca, te = split_by_day(build_sequences(df, features, DEFAULT, DEFAULT.stride_train), DEFAULT)

    fit_idx, val_idx = chronological_validation_split(tr)
    pipe = FeaturePipeline(DEFAULT).fit(tr.X[fit_idx])
    Xtr = pipe.transform(tr.X)
    model = BUILDERS[args.arch](DEFAULT, Xtr.shape[-1])
    train(model, Xtr[fit_idx], tr.y[fit_idx], Xtr[val_idx], tr.y[val_idx], DEFAULT, verbose=2)

    s_ca = model.predict(pipe.transform(ca.X), verbose=0).ravel()
    thr = threshold_at_fpr(ca.y, s_ca, DEFAULT.target_fpr)

    model.save(args.models / "elman.keras")
    (args.models / "artifacts.json").write_text(json.dumps({
        "pipeline": pipe.to_dict(), "threshold": float(thr),
        "features": features, "config": DEFAULT.to_dict(), "arch": args.arch,
    }, indent=2))

    keep = np.isin(te.host, DEMO_HOSTS)
    np.savez_compressed(
        args.data / "demo_windows.npz",
        X=te.X[keep].astype(np.float32), y=te.y[keep],
        attack_type=te.attack_type[keep], host=te.host[keep], day=te.day[keep],
        end_time=te.end_time[keep].astype("datetime64[ns]").astype(np.int64),
    )
    print(f"\nsaved model + {int(keep.sum()):,} demo windows "
          f"({args.data / 'demo_windows.npz'}), threshold={thr:.4f}")


if __name__ == "__main__":
    main()
