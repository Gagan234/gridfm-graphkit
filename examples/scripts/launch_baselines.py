#!/usr/bin/env python3
"""Launch the non-foundation baseline matrix as Slurm jobs.

Submits 9 jobs by default: 3 baselines (linear / mlp / lstm) × 3 seeds.
The linear and MLP baselines run on the cpu partition (they're tiny
and don't benefit from GPU); the LSTM baseline runs on the suny
partition with one GPU.

Usage::

    python examples/scripts/launch_baselines.py            # dry-run
    python examples/scripts/launch_baselines.py --submit   # actually submit

Subset support::

    python examples/scripts/launch_baselines.py --submit --baselines linear mlp
    python examples/scripts/launch_baselines.py --submit --seeds 0 1 2 3 4
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
SBATCH_SCRIPT = (
    REPO_ROOT / "examples" / "scripts" / "run_baseline_job.sbatch"
)
BASE_CONFIG = (
    REPO_ROOT / "examples" / "config" / "baselines" / "baseline_case118.yaml"
)
DEFAULT_DATA_DIR = Path("/mnt/lustre/suny/gsapkota/data")
DEFAULT_RUNS_DIR = Path("/mnt/lustre/suny/gsapkota/runs")

BASELINES = ["linear", "mlp", "lstm"]


def _partition_args_for(baseline: str) -> list[str]:
    """Return the Slurm partition + GPU args appropriate for this baseline."""
    if baseline == "lstm":
        return [
            "--partition=suny",
            "--gpus-per-node=1",
        ]
    # Linear and MLP run on CPU; the cpu partition has no GPU and no
    # SUNY-only restriction.
    return ["--partition=cpu"]


def _submit_one(
    baseline: str,
    seed: int,
    data_dir: Path,
    runs_dir: Path,
    submit: bool,
) -> None:
    run_name = f"baseline_{baseline}_seed{seed}"
    metrics_path = runs_dir / "baseline_results" / f"{run_name}.json"
    ckpt_path = runs_dir / "baseline_checkpoints" / f"{run_name}.pt"

    export_dict = {
        "CONFIG": str(BASE_CONFIG.resolve()),
        "BASELINE": baseline,
        "SEED": str(seed),
        "DATA_PATH": str(data_dir),
        "OUTPUT": str(metrics_path),
        "CHECKPOINT_OUT": str(ckpt_path),
    }
    export_str = ",".join(f"{k}={v}" for k, v in export_dict.items())

    cmd = [
        "sbatch",
        *_partition_args_for(baseline),
        "--job-name", run_name,
        "--export", export_str,
        str(SBATCH_SCRIPT),
    ]
    print("+ " + " ".join(shlex.quote(c) for c in cmd))
    if submit:
        result = subprocess.run(cmd, check=False, capture_output=True, text=True)
        if result.returncode != 0:
            print(
                f"  ! sbatch failed (rc={result.returncode}): "
                f"{result.stderr.strip()}",
                file=sys.stderr,
            )
        elif result.stdout:
            print(f"  -> {result.stdout.strip()}")


def _check_environment(submit: bool) -> None:
    if not BASE_CONFIG.is_file():
        raise SystemExit(f"Missing baseline config: {BASE_CONFIG}")
    if not SBATCH_SCRIPT.is_file():
        raise SystemExit(f"Missing sbatch template: {SBATCH_SCRIPT}")
    if submit and shutil.which("sbatch") is None:
        raise SystemExit(
            "`sbatch` not on PATH; you're not on the cluster. Drop "
            "--submit (default dry-run) to preview, or run on the cluster.",
        )


def launch(
    seeds: Iterable[int],
    baselines: Iterable[str],
    data_dir: Path,
    runs_dir: Path,
    submit: bool,
) -> None:
    _check_environment(submit)

    seeds = list(seeds)
    baselines = list(baselines)
    total = len(seeds) * len(baselines)

    mode = "SUBMITTING" if submit else "DRY-RUN"
    print(
        f"[{mode}] {len(baselines)} baselines × {len(seeds)} seeds = {total} jobs",
    )
    print(f"  data:   {data_dir}")
    print(f"  runs:   {runs_dir}")
    print(f"  config: {BASE_CONFIG}")
    print()

    for baseline in baselines:
        for seed in seeds:
            _submit_one(
                baseline=baseline,
                seed=seed,
                data_dir=data_dir.resolve(),
                runs_dir=runs_dir.resolve(),
                submit=submit,
            )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Launch the non-foundation baseline training matrix.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to run per baseline (default: 0 1 2).",
    )
    p.add_argument(
        "--baselines",
        nargs="+",
        choices=BASELINES,
        default=BASELINES,
        help="Subset of baselines to run (default: all three).",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help="Actually submit jobs (default: dry-run).",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    launch(
        seeds=args.seeds,
        baselines=args.baselines,
        data_dir=args.data_dir,
        runs_dir=args.runs_dir,
        submit=args.submit,
    )


if __name__ == "__main__":
    main()
