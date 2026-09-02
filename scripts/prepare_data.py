#!/usr/bin/env python3
"""Turn the raw CICIDS2017 CSVs into one compact, analysis-ready table.

    python scripts/prepare_data.py --raw /path/to/TrafficLabelling --out data/processed

Each day file is read, repaired and written to Parquet in float32 one at a time,
so the whole 1.2 GB capture never has to sit in memory at once.
"""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from elmanguard.config import DEFAULT  # noqa: E402
from elmanguard.features import clean_flows  # noqa: E402
from elmanguard.loading import DAY_OF_FILE, read_day_file  # noqa: E402

KEEP_META = ["timestamp", "source_ip", "destination_ip", "destination_port", "protocol",
             "label", "day"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    reports, summaries = {}, []
    for name in DAY_OF_FILE:
        path = args.raw / name
        if not path.exists():
            print(f"  (missing) {name}")
            continue
        df = read_day_file(path)
        df, rep = clean_flows(df, DEFAULT, drop_constant=False)
        reports[name] = rep

        numeric = [c for c in df.select_dtypes(include=[np.number]).columns]
        df[numeric] = df[numeric].astype(np.float32)
        keep, seen = [], set()
        for c in [c for c in KEEP_META if c in df.columns] + numeric:
            if c not in seen:
                keep.append(c); seen.add(c)
        out_path = args.out / (name.replace(".pcap_ISCX.csv", "") + ".parquet")
        df[keep].to_parquet(out_path, index=False, compression="zstd")

        summaries.append(
            df.groupby("label", observed=True).size().rename(name.split("-")[0]).to_frame().T
        )
        print(f"  -> {out_path.name}  {out_path.stat().st_size / 1e6:.0f} MB\n")
        del df

    (args.out / "prepare_report.json").write_text(json.dumps(reports, indent=2, default=str))

    table = pd.concat(summaries).fillna(0).astype(int).groupby(level=0).sum()
    print("\n=== flows per label, per day ===")
    print(table.T.sort_values(list(table.index)[0], ascending=False).to_string())
    table.T.to_csv(args.out / "label_counts.csv")


if __name__ == "__main__":
    main()
