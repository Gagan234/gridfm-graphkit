"""Inspect a gridfm-datakit-generated dataset directory.

Reports shape, columns, scenario/bus counts, and summary stats for every
parquet file/directory under a `raw/` dataset, plus the contents of the
`*.log` and `n_scenarios.txt` breadcrumbs gridfm-datakit writes alongside.
Useful as a quick sanity check after running `generate_power_flow_data`.

Usage:

    python examples/scripts/inspect_dataset.py <path-to-raw-dir>

Example:

    python examples/scripts/inspect_dataset.py \\
        /mnt/home/gsapkota/thesis-runs/data/case118_ieee/raw
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def _print_text_file(path: Path) -> None:
    print(f"--- {path.name} ---")
    text = path.read_text().rstrip()
    if text:
        print(text)
    else:
        print("(empty)")
    print()


def _print_parquet(path: Path) -> None:
    print(f"--- {path.name} ---")
    try:
        df = pd.read_parquet(path)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  ERROR reading: {exc}\n")
        return

    print(f"  shape : {df.shape}")
    print(f"  cols  : {list(df.columns)}")

    for col_name in ("scenario", "bus", "branch", "gen"):
        if col_name in df.columns:
            print(f"  {col_name}: {df[col_name].nunique()} unique")

    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    interesting = [c for c in ("Pd", "Qd", "Pg", "Qg", "Vm", "Va", "Pf", "Qf") if c in numeric_cols]
    if interesting:
        print("  stats :")
        for col in interesting:
            s = df[col]
            print(f"    {col:>3s}: min={s.min():+.4f}  max={s.max():+.4f}  mean={s.mean():+.4f}")

    print("  head  :")
    print(df.head(3).to_string(index=False, max_cols=10))
    print()


def inspect_dataset(raw_dir: Path) -> None:
    if not raw_dir.is_dir():
        sys.exit(f"Not a directory: {raw_dir}")

    print(f"Inspecting dataset at: {raw_dir}\n")

    for log in sorted(p for p in raw_dir.glob("*.log") if p.is_file()):
        _print_text_file(log)

    n_scenarios = raw_dir / "n_scenarios.txt"
    if n_scenarios.exists():
        _print_text_file(n_scenarios)

    parquet_targets = sorted(
        list(raw_dir.glob("*.parquet")),
        key=lambda p: p.name,
    )
    for pq in parquet_targets:
        _print_parquet(pq)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "raw_dir",
        type=Path,
        help="Path to <data_dir>/<network>/raw/ produced by gridfm-datakit.",
    )
    args = parser.parse_args()
    inspect_dataset(args.raw_dir)


if __name__ == "__main__":
    main()
