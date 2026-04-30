#!/usr/bin/env python3
"""End-to-end thesis evaluation: walk MLflow runs, run forecasting eval, build the tables.

This script is the "press one button to get the thesis results"
endpoint. After the ablation matrix has finished training, run::

    python examples/scripts/run_full_evaluation.py \\
        --runs-dir /mnt/lustre/suny/gsapkota/runs \\
        --exp-name ablation_case118 \\
        --data-path /mnt/lustre/suny/gsapkota/data \\
        --output-dir ~/thesis-runs/results

It will:

1. Discover every MLflow run in the named experiment.
2. For each run, find the frozen training config and the
   ``best_model_state_dict.pt`` checkpoint.
3. Run forecasting evaluation (calls
   :func:`forecasting_eval.run_forecasting_eval` in-process so we
   avoid 36× Python-startup overhead).
4. Collect the per-run JSON outputs into a long-format CSV.
5. Render Markdown tables for the experiments chapter:

   - **Pretraining table:** masked-reconstruction loss (from MLflow's
     ``Test Masked bus MSE loss`` metric), strategies × architectures,
     mean ± std across seeds.
   - **Forecasting table:** Vm RMSE / NRMSE on held-out future windows,
     same layout, plus a persistence-baseline reference column.
   - **Non-foundation baselines table** (if ``--baseline-results-dir``
     is provided and contains JSON outputs from
     ``examples/scripts/train_baseline.py``): one row per baseline
     (linear / MLP / LSTM) with mean ± std forecasting metrics across
     seeds. This is the trained-but-non-foundation reference for the
     thesis comparison.

All tables are paste-ready for the thesis methodology + experiments
chapters.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
import traceback
from pathlib import Path
from typing import Dict, List, Optional

# Add the scripts directory to sys.path so we can import forecasting_eval.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from forecasting_eval import run_forecasting_eval  # noqa: E402


_RUN_NAME_RE = re.compile(
    r"^(?P<arch>baseline|factorized)__"
    r"(?P<strategy>random_point|block_temporal|causal|block_spatial|tube|topology)__"
    r"seed(?P<seed>\d+)$",
)

_ARCH_ORDER = ["baseline", "factorized"]
_STRATEGY_ORDER = [
    "random_point",
    "block_temporal",
    "causal",
    "block_spatial",
    "tube",
    "topology",
]


def _discover_runs(runs_dir: Path, exp_name: str):
    try:
        import mlflow
    except ImportError as e:
        raise SystemExit(
            "mlflow is not importable. Activate the gridfm conda env "
            "before running this script.",
        ) from e

    mlflow.set_tracking_uri(f"file://{runs_dir.expanduser().resolve()}")
    client = mlflow.tracking.MlflowClient()
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        raise SystemExit(
            f"Experiment {exp_name!r} not found under {runs_dir}. "
            "Available experiments: "
            f"{[e.name for e in client.search_experiments()]}",
        )

    return client, exp.experiment_id, list(
        client.search_runs([exp.experiment_id], max_results=1000),
    )


def _resolve_artifacts(
    runs_dir: Path, run, configs_dir: Path,
) -> Dict[str, Path]:
    """Locate the frozen config and best-model state dict for a run."""
    name = run.info.run_name or ""
    return {
        "config": (configs_dir / f"{name}.yaml").resolve(),
        "checkpoint": (
            runs_dir
            / run.info.experiment_id
            / run.info.run_id
            / "artifacts"
            / "model"
            / "best_model_state_dict.pt"
        ).resolve(),
    }


def _format_cell(values: List[float]) -> str:
    if not values:
        return "—"
    if len(values) == 1:
        return f"{values[0]:.4f}"
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) >= 2 else 0.0
    return f"{mean:.4f} ± {sd:.4f}"


def _write_long_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    keys = sorted({k for row in rows for k in row.keys()})
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _build_grid(
    rows: List[dict], metric_extractor,
) -> Dict[str, Dict[str, List[float]]]:
    grid: Dict[str, Dict[str, List[float]]] = {}
    for row in rows:
        v = metric_extractor(row)
        if v is None:
            continue
        grid.setdefault(row["strategy"], {}).setdefault(
            row["architecture"], [],
        ).append(float(v))
    return grid


def _render_table(
    grid: Dict[str, Dict[str, List[float]]],
    title: str,
    description: str,
    extra_columns: Optional[Dict[str, Dict[str, List[float]]]] = None,
) -> str:
    """Build a markdown table with strategies as rows and architectures
    (plus optional extra columns) as columns. Cells are mean ± std.
    """
    extra_columns = extra_columns or {}
    lines = [f"## {title}", "", description, ""]
    columns = list(_ARCH_ORDER) + list(extra_columns.keys())
    lines.append("| Masking strategy | " + " | ".join(columns) + " |")
    lines.append("|" + "|".join(["---"] * (1 + len(columns))) + "|")

    strategies = [s for s in _STRATEGY_ORDER if s in grid or any(
        s in extra_columns[k] for k in extra_columns
    )]
    for strategy in strategies:
        cells = [strategy]
        for arch in _ARCH_ORDER:
            cells.append(_format_cell(grid.get(strategy, {}).get(arch, [])))
        for col_name, col_grid in extra_columns.items():
            cells.append(
                _format_cell(col_grid.get(strategy, {}).get("any", [])),
            )
        lines.append("| " + " | ".join(cells) + " |")

    # Pooled-across-strategies summary row.
    pooled_arch = {
        a: [v for s in grid for v in grid[s].get(a, [])] for a in _ARCH_ORDER
    }
    pooled_extra = {
        col_name: [
            v for s in col_grid for v in col_grid[s].get("any", [])
        ]
        for col_name, col_grid in extra_columns.items()
    }
    summary_cells = ["**all (pooled)**"]
    for arch in _ARCH_ORDER:
        summary_cells.append(_format_cell(pooled_arch[arch]))
    for col_name in extra_columns:
        summary_cells.append(_format_cell(pooled_extra[col_name]))
    lines.append("| " + " | ".join(summary_cells) + " |")
    lines.append("")

    # Run counts so the reader sees missing cells.
    lines.append("### Runs per cell")
    lines.append("")
    lines.append(
        "| Masking strategy | " + " | ".join(_ARCH_ORDER) + " |"
    )
    lines.append("|" + "|".join(["---"] * (1 + len(_ARCH_ORDER))) + "|")
    for strategy in strategies:
        cells = [strategy]
        for arch in _ARCH_ORDER:
            cells.append(
                str(len(grid.get(strategy, {}).get(arch, []))),
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=Path("/mnt/lustre/suny/gsapkota/runs"),
        help="MLflow tracking root.",
    )
    p.add_argument(
        "--exp-name",
        default="ablation_case118",
        help="MLflow experiment name.",
    )
    p.add_argument(
        "--data-path",
        type=Path,
        default=Path("/mnt/lustre/suny/gsapkota/data"),
        help="Root containing per-network parquet directories.",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=4,
        help="Forecasting horizon (last N time steps masked).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("~/thesis-runs/results"),
        help="Where to write the CSV + Markdown tables.",
    )
    p.add_argument(
        "--skip-forecasting",
        action="store_true",
        help=(
            "Only build the pretraining-loss table from MLflow metrics; "
            "skip forecasting eval. Useful for a quick check before "
            "all checkpoints have finished."
        ),
    )
    p.add_argument(
        "--baseline-results-dir",
        type=Path,
        default=None,
        help=(
            "Optional directory containing JSON outputs from "
            "examples/scripts/train_baseline.py. If provided, a third "
            "Markdown table summarising the non-foundation baselines "
            "(linear / MLP / LSTM) is rendered."
        ),
    )
    args = p.parse_args()

    runs_dir = args.runs_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Discovering MLflow runs in {runs_dir}, experiment={args.exp_name!r}")
    client, exp_id, runs = _discover_runs(runs_dir, args.exp_name)
    print(f"  found {len(runs)} runs")

    configs_dir = runs_dir / "configs"

    long_rows: List[dict] = []
    for run in runs:
        name = run.info.run_name or ""
        m = _RUN_NAME_RE.match(name)
        if not m:
            print(f"  ! skipping unparseable run_name: {name!r}")
            continue

        row = {
            "architecture": m.group("arch"),
            "strategy": m.group("strategy"),
            "seed": int(m.group("seed")),
            "run_name": name,
            "run_id": run.info.run_id,
            "status": run.info.status,
        }

        # Pretraining loss from MLflow metrics.
        for key in (
            "Test loss",
            "Test Masked bus MSE loss",
        ):
            if key in run.data.metrics:
                row[key] = run.data.metrics[key]

        # Forecasting eval (optional).
        if not args.skip_forecasting:
            artifacts = _resolve_artifacts(runs_dir, run, configs_dir)
            if (
                artifacts["config"].is_file()
                and artifacts["checkpoint"].is_file()
            ):
                try:
                    record = run_forecasting_eval(
                        config_path=artifacts["config"],
                        checkpoint_path=artifacts["checkpoint"],
                        data_path=args.data_path,
                        horizon=args.horizon,
                        verbose=False,
                    )
                    metrics = record["metrics"]
                    row["forecast_vm_rmse_model"] = metrics["model"]["vm"]["rmse"]
                    row["forecast_vm_nrmse_model"] = metrics["model"]["vm"]["nrmse"]
                    row["forecast_va_rmse_model"] = metrics["model"]["va"]["rmse"]
                    row["forecast_va_nrmse_model"] = metrics["model"]["va"]["nrmse"]
                    row["forecast_vm_rmse_persistence"] = metrics["persistence"]["vm"]["rmse"]
                    row["forecast_vm_nrmse_persistence"] = metrics["persistence"]["vm"]["nrmse"]
                    row["forecast_va_rmse_persistence"] = metrics["persistence"]["va"]["rmse"]
                    row["forecast_va_nrmse_persistence"] = metrics["persistence"]["va"]["nrmse"]
                except Exception as e:
                    print(
                        f"  ! forecasting eval failed for {name}: {e}",
                        file=sys.stderr,
                    )
                    traceback.print_exc()
            else:
                missing = [
                    k for k, v in artifacts.items() if not v.is_file()
                ]
                print(
                    f"  ! skipping forecasting eval for {name}: missing "
                    f"{missing}",
                )

        long_rows.append(row)

    # Long-format CSV.
    csv_path = output_dir / "thesis_ablation_long.csv"
    _write_long_csv(long_rows, csv_path)
    print(f"  wrote {csv_path} ({len(long_rows)} rows)")

    # Pretraining table.
    pretraining_grid = _build_grid(
        long_rows,
        lambda r: r.get("Test Masked bus MSE loss"),
    )
    pretraining_md = _render_table(
        pretraining_grid,
        title="Pretraining results — Test Masked bus MSE loss",
        description=(
            "Mean ± standard deviation across seeds (lower is better). "
            "This is the masked-reconstruction loss reported on the "
            "held-out test windows during the pretraining run; the "
            "loss is computed on the same masking strategy that the "
            "model was trained with."
        ),
    )
    (output_dir / "thesis_pretraining_table.md").write_text(pretraining_md)
    print(f"  wrote {output_dir / 'thesis_pretraining_table.md'}")

    if not args.skip_forecasting:
        # Forecasting table.
        # Persistence is a strategy-independent baseline — we collect its
        # values once across the strategies that have ANY model run, since
        # the persistence prediction is the same for the same test data
        # regardless of pretraining strategy.
        forecasting_grid = _build_grid(
            long_rows,
            lambda r: r.get("forecast_vm_rmse_model"),
        )
        # Persistence "extra column" — pull values from any architecture
        # since the persistence baseline doesn't depend on the model.
        persistence_grid: Dict[str, Dict[str, List[float]]] = {}
        for r in long_rows:
            v = r.get("forecast_vm_rmse_persistence")
            if v is None:
                continue
            persistence_grid.setdefault(r["strategy"], {}).setdefault(
                "any", [],
            ).append(float(v))

        forecasting_md = _render_table(
            forecasting_grid,
            title="Forecasting results — Vm RMSE on held-out future windows",
            description=(
                f"Held-out test windows of {args.horizon} time steps "
                "are masked at the trailing block; the model "
                "reconstructs them from the prefix context. Mean ± "
                "standard deviation across seeds; lower is better. "
                "The persistence column is the naive 'future = last "
                "observed' reference (zero-parameter), pooled across "
                "models within each strategy because the baseline "
                "doesn't depend on the trained model."
            ),
            extra_columns={"persistence": persistence_grid},
        )
        (output_dir / "thesis_forecasting_table.md").write_text(forecasting_md)
        print(f"  wrote {output_dir / 'thesis_forecasting_table.md'}")

    # Non-foundation baselines table (optional).
    if args.baseline_results_dir is not None:
        _render_baselines_table(
            args.baseline_results_dir,
            output_dir / "thesis_baselines_table.md",
        )


def _render_baselines_table(
    results_dir: Path,
    out_path: Path,
) -> None:
    """Aggregate the per-baseline-per-seed JSON outputs into a table.

    Reads every ``*.json`` file in ``results_dir`` (assuming each is
    the output of ``examples/scripts/train_baseline.py``) and emits a
    Markdown table with one row per baseline (linear / MLP / LSTM),
    columns for Vm RMSE / NRMSE / Va RMSE, mean ± std across seeds.
    Includes the persistence reference (computed inline by each
    baseline's training script on the same test windows; pooled across
    baselines because the persistence value is independent of the
    model).
    """
    results_dir = results_dir.expanduser().resolve()
    if not results_dir.is_dir():
        print(
            f"  ! baseline-results-dir does not exist: {results_dir}",
            file=sys.stderr,
        )
        return

    rows: List[dict] = []
    for json_path in sorted(results_dir.glob("*.json")):
        with open(json_path) as f:
            record = json.load(f)
        rows.append(record)

    if not rows:
        print(
            f"  (no baseline JSONs found in {results_dir})",
            file=sys.stderr,
        )
        return

    # Group by baseline type.
    grouped: Dict[str, List[dict]] = {}
    for r in rows:
        grouped.setdefault(r.get("baseline", "?"), []).append(r)

    persistence_vm_rmse: List[float] = []
    persistence_va_rmse: List[float] = []
    for r in rows:
        try:
            persistence_vm_rmse.append(
                float(r["metrics"]["persistence"]["vm"]["rmse"]),
            )
            persistence_va_rmse.append(
                float(r["metrics"]["persistence"]["va"]["rmse"]),
            )
        except (KeyError, TypeError, ValueError):
            continue

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("## Non-foundation baseline forecasting results\n\n")
        f.write(
            "Mean ± standard deviation across seeds; lower is better. "
            "Each baseline is trained directly on the forecasting "
            "objective (predict the trailing 4 time steps' Vm and Va "
            "from the preceding 8 time steps of bus features). Weights "
            "are shared across all buses, so the baselines do not "
            "exploit graph topology — this isolates the value of the "
            "graph-aware foundation-model contribution. The persistence "
            "row is the naive 'future = last observed' reference, "
            "pooled across all baseline runs (the persistence "
            "prediction is independent of the trained model).\n\n",
        )
        f.write("| Method | Vm RMSE | Vm NRMSE | Va RMSE |\n")
        f.write("|---|---|---|---|\n")

        # Persistence row first (the reference floor).
        f.write(
            "| persistence (zero-param) | "
            f"{_format_cell(persistence_vm_rmse)} | "
            f"— | "
            f"{_format_cell(persistence_va_rmse)} |\n",
        )

        for baseline_name in ("linear", "mlp", "lstm"):
            group = grouped.get(baseline_name, [])
            if not group:
                f.write(
                    f"| {baseline_name} | — | — | — |\n",
                )
                continue
            vm_rmse_vals: List[float] = []
            vm_nrmse_vals: List[float] = []
            va_rmse_vals: List[float] = []
            for r in group:
                try:
                    vm_rmse_vals.append(
                        float(r["metrics"]["model"]["vm"]["rmse"]),
                    )
                    vm_nrmse_vals.append(
                        float(r["metrics"]["model"]["vm"]["nrmse"]),
                    )
                    va_rmse_vals.append(
                        float(r["metrics"]["model"]["va"]["rmse"]),
                    )
                except (KeyError, TypeError, ValueError):
                    continue
            f.write(
                f"| {baseline_name} | "
                f"{_format_cell(vm_rmse_vals)} | "
                f"{_format_cell(vm_nrmse_vals)} | "
                f"{_format_cell(va_rmse_vals)} |\n",
            )

        # Per-baseline run counts.
        f.write("\n### Runs per baseline\n\n")
        f.write("| baseline | seeds |\n|---|---|\n")
        for baseline_name in ("linear", "mlp", "lstm"):
            f.write(
                f"| {baseline_name} | "
                f"{len(grouped.get(baseline_name, []))} |\n",
            )
    print(f"  wrote {out_path}")


if __name__ == "__main__":
    main()
