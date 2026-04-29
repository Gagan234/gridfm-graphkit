#!/usr/bin/env python3
"""Aggregate the thesis ablation matrix's MLflow runs into a results table.

Walks an MLflow tracking directory, reads every run that matches the
``ablation_case118`` experiment, and emits:

1. A long-format CSV (one row per run) with all hyperparameters and
   final metrics — useful for downstream analysis (per-seed plots,
   significance tests, etc.).
2. A wide-format Markdown table (one row per (architecture × strategy),
   columns: mean ± std test loss across seeds) — paste-ready for the
   thesis experiments chapter.

Run name convention (set by ``launch_ablation_matrix.py``):
``<arch>__<strategy>__seed<N>``. The aggregator parses these to group
runs into ablation cells.

Usage::

    python examples/scripts/aggregate_ablation_results.py \\
        --runs-dir /mnt/lustre/suny/gsapkota/runs \\
        --exp-name ablation_case118 \\
        --csv-out ~/thesis-runs/ablation_long.csv \\
        --md-out  ~/thesis-runs/ablation_table.md
"""

from __future__ import annotations

import argparse
import re
import statistics
import sys
from pathlib import Path
from typing import Dict, List, Optional


_RUN_NAME_RE = re.compile(
    r"^(?P<arch>baseline|factorized)__"
    r"(?P<strategy>random_point|block_temporal|causal|block_spatial|tube|topology)__"
    r"seed(?P<seed>\d+)$",
)

_TEST_METRIC_KEYS = (
    # The Lightning logger prefixes test metrics with "Test " (capital T).
    "Test loss",
    "Test Masked bus MSE loss",
)


def _parse_run_name(run_name: str) -> Optional[dict]:
    m = _RUN_NAME_RE.match(run_name)
    if not m:
        return None
    return {
        "architecture": m.group("arch"),
        "strategy": m.group("strategy"),
        "seed": int(m.group("seed")),
    }


def _load_runs(runs_dir: Path, exp_name: str) -> List[dict]:
    """Read all runs in `runs_dir` belonging to experiment `exp_name`.

    Uses MLflow's tracking client via the local file URI. Returns one
    dict per run with the parsed run-name fields, hyperparameters,
    and final metric values.
    """
    try:
        import mlflow
    except ImportError as e:
        raise SystemExit(
            "mlflow is not importable. Activate the gridfm conda env "
            "(or install mlflow) before running this script.",
        ) from e

    runs_dir = runs_dir.expanduser().resolve()
    if not runs_dir.is_dir():
        raise SystemExit(f"Runs directory does not exist: {runs_dir}")

    mlflow.set_tracking_uri(f"file://{runs_dir}")
    client = mlflow.tracking.MlflowClient()

    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        raise SystemExit(
            f"Experiment {exp_name!r} not found under {runs_dir}. "
            "Available experiments: "
            f"{[e.name for e in client.search_experiments()]}",
        )

    rows: List[dict] = []
    for run in client.search_runs([exp.experiment_id], max_results=1000):
        name = run.info.run_name or ""
        parsed = _parse_run_name(name)
        if parsed is None:
            print(
                f"  ! skipping run with unparseable name: {name!r}",
                file=sys.stderr,
            )
            continue
        row = {
            **parsed,
            "run_name": name,
            "run_id": run.info.run_id,
            "status": run.info.status,
            "start_time": run.info.start_time,
            "end_time": run.info.end_time,
            **{f"param_{k}": v for k, v in run.data.params.items()},
        }
        for key in _TEST_METRIC_KEYS:
            if key in run.data.metrics:
                row[key] = run.data.metrics[key]
        rows.append(row)
    return rows


def _write_long_csv(rows: List[dict], out_path: Path) -> None:
    if not rows:
        print("  (no rows; long CSV not written)", file=sys.stderr)
        return
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({k for row in rows for k in row.keys()})

    import csv

    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"  wrote long-format CSV: {out_path} ({len(rows)} rows)")


def _format_md_cell(values: List[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) >= 2 else 0.0
    return f"{mean:.4f} ± {sd:.4f}"


def _write_markdown_table(
    rows: List[dict],
    out_path: Path,
    metric_key: str = "Test Masked bus MSE loss",
) -> None:
    """Build a `(strategy × architecture)` mean±std table.

    Strategies are rows (clearer in a thesis report); architectures
    are columns (the factorized one is the contribution being
    evaluated).
    """
    grid: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        if metric_key not in row:
            continue
        grid.setdefault(row["strategy"], {}).setdefault(
            row["architecture"], [],
        ).append(float(row[metric_key]))

    strategies = [
        s for s in (
            "random_point",
            "block_temporal",
            "causal",
            "block_spatial",
            "tube",
            "topology",
        ) if s in grid
    ]
    architectures = ["baseline", "factorized"]

    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write(f"# Ablation results — {metric_key}\n\n")
        f.write(
            "Mean ± standard deviation across seeds (lower is better). "
            "Cells are blank when no runs of that (strategy, architecture) "
            "combination were found.\n\n",
        )
        f.write("| Masking strategy | " + " | ".join(architectures) + " |\n")
        f.write("|" + "|".join(["---"] * (1 + len(architectures))) + "|\n")
        for strategy in strategies:
            cells = [strategy]
            for arch in architectures:
                values = grid.get(strategy, {}).get(arch, [])
                cells.append(_format_md_cell(values))
            f.write("| " + " | ".join(cells) + " |\n")

        # Summary row: pooled across strategies.
        f.write("| **all (pooled)** | ")
        f.write(
            " | ".join(
                _format_md_cell(
                    [v for s in grid for v in grid[s].get(arch, [])],
                )
                for arch in architectures
            ),
        )
        f.write(" |\n")

        # Annotate the run count per cell so the reader can see which
        # cells are still incomplete.
        f.write("\n## Runs per cell\n\n")
        f.write("| Masking strategy | " + " | ".join(architectures) + " |\n")
        f.write("|" + "|".join(["---"] * (1 + len(architectures))) + "|\n")
        for strategy in strategies:
            counts = [strategy]
            for arch in architectures:
                values = grid.get(strategy, {}).get(arch, [])
                counts.append(str(len(values)))
            f.write("| " + " | ".join(counts) + " |\n")
    print(f"  wrote markdown table: {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("/mnt/lustre/suny/gsapkota/runs"),
        help="Path to MLflow tracking root.",
    )
    p.add_argument(
        "--exp-name",
        default="ablation_case118",
        help="MLflow experiment name.",
    )
    p.add_argument(
        "--csv-out",
        type=Path,
        default=Path("~/thesis-runs/ablation_long.csv"),
    )
    p.add_argument(
        "--md-out",
        type=Path,
        default=Path("~/thesis-runs/ablation_table.md"),
    )
    args = p.parse_args()

    rows = _load_runs(args.runs_dir, args.exp_name)
    print(f"Loaded {len(rows)} parseable runs from "
          f"experiment {args.exp_name!r}.")
    _write_long_csv(rows, args.csv_out)
    _write_markdown_table(rows, args.md_out)


if __name__ == "__main__":
    main()
