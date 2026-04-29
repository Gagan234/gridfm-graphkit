#!/usr/bin/env python3
"""Launch the full thesis ablation matrix as Slurm jobs.

For each (architecture × masking strategy × seed) cell of the matrix,
this script:

1. Reads the base config emitted by ``generate_ablation_configs.py``.
2. Stamps in the seed and writes a per-run config to the runs
   directory (so every cell is reproducible from a frozen YAML).
3. Submits an sbatch job using ``run_ablation_job.sbatch`` with the
   per-run config and a structured ``run_name`` MLflow can join on
   later.

By default this is a **dry run** — it prints the commands it would
execute. Pass ``--submit`` to actually submit jobs to Slurm.

Usage::

    # dry-run (default): print everything, submit nothing
    python examples/scripts/launch_ablation_matrix.py

    # actually submit:
    python examples/scripts/launch_ablation_matrix.py --submit

    # different seed list:
    python examples/scripts/launch_ablation_matrix.py --seeds 0 1 2 3 4 --submit

    # only the factorized architecture (skip baseline):
    python examples/scripts/launch_ablation_matrix.py --architectures factorized --submit
"""

from __future__ import annotations

import argparse
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Iterable, List

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
ABLATION_CONFIG_DIR = REPO_ROOT / "examples" / "config" / "ablation_case118"
SBATCH_SCRIPT = (
    REPO_ROOT / "examples" / "scripts" / "run_ablation_job.sbatch"
)
DEFAULT_RUNS_DIR = Path("/mnt/lustre/suny/gsapkota/runs")
DEFAULT_DATA_DIR = Path("/mnt/lustre/suny/gsapkota/data")
DEFAULT_EXP_NAME = "ablation_case118"

ARCHITECTURES = ["baseline", "factorized"]
STRATEGIES = [
    "random_point",
    "block_temporal",
    "causal",
    "block_spatial",
    "tube",
    "topology",
]


def _config_path_for(arch: str, strategy: str) -> Path:
    return ABLATION_CONFIG_DIR / f"{arch}__{strategy}__case118.yaml"


def _stamp_seed(base_cfg_path: Path, seed: int, out_path: Path) -> None:
    """Write a copy of ``base_cfg_path`` with ``seed:`` overridden."""
    with open(base_cfg_path) as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = int(seed)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True, default_flow_style=False)


def _submit_one(
    config_path: Path,
    run_name: str,
    data_dir: Path,
    log_dir: Path,
    exp_name: str,
    submit: bool,
) -> None:
    export = (
        f"CONFIG={config_path},DATA_PATH={data_dir},"
        f"LOG_DIR={log_dir},EXP_NAME={exp_name},RUN_NAME={run_name}"
    )
    cmd = [
        "sbatch",
        "--job-name", run_name,
        "--export", export,
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
    if not ABLATION_CONFIG_DIR.is_dir():
        raise SystemExit(
            f"Ablation config directory does not exist: "
            f"{ABLATION_CONFIG_DIR}\n"
            "Run examples/scripts/generate_ablation_configs.py first.",
        )
    if not SBATCH_SCRIPT.is_file():
        raise SystemExit(f"Missing sbatch template: {SBATCH_SCRIPT}")
    if submit and shutil.which("sbatch") is None:
        raise SystemExit(
            "`sbatch` not on PATH; you're not on the cluster. Drop "
            "--submit (default dry-run) to preview, or run on the cluster.",
        )


def launch(
    seeds: Iterable[int],
    architectures: Iterable[str],
    strategies: Iterable[str],
    data_dir: Path,
    runs_dir: Path,
    exp_name: str,
    submit: bool,
) -> None:
    _check_environment(submit)

    seeds = list(seeds)
    archs = list(architectures)
    strats = list(strategies)
    total = len(seeds) * len(archs) * len(strats)

    mode = "SUBMITTING" if submit else "DRY-RUN"
    print(
        f"[{mode}] {len(archs)} architectures × {len(strats)} strategies "
        f"× {len(seeds)} seeds = {total} jobs",
    )
    print(f"  data: {data_dir}")
    print(f"  runs: {runs_dir}")
    print(f"  exp:  {exp_name}")
    print()

    for seed in seeds:
        for arch in archs:
            for strategy in strats:
                base_cfg = _config_path_for(arch, strategy)
                if not base_cfg.is_file():
                    print(
                        f"  SKIP (missing base config): {base_cfg.name}",
                        file=sys.stderr,
                    )
                    continue

                run_name = f"{arch}__{strategy}__seed{seed}"
                stamped_cfg = (
                    runs_dir / "configs" / f"{run_name}.yaml"
                ).resolve()
                _stamp_seed(base_cfg, seed, stamped_cfg)

                _submit_one(
                    config_path=stamped_cfg,
                    run_name=run_name,
                    data_dir=data_dir,
                    log_dir=runs_dir,
                    exp_name=exp_name,
                    submit=submit,
                )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Launch the gridfm thesis ablation matrix as Slurm jobs.",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Seeds to run per (architecture, strategy) cell. Default: 0 1 2.",
    )
    p.add_argument(
        "--architectures",
        nargs="+",
        choices=ARCHITECTURES,
        default=ARCHITECTURES,
        help="Subset of architectures to run. Default: both.",
    )
    p.add_argument(
        "--strategies",
        nargs="+",
        choices=STRATEGIES,
        default=STRATEGIES,
        help="Subset of masking strategies to run. Default: all six.",
    )
    p.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Path to dataset directory. Default: {DEFAULT_DATA_DIR}.",
    )
    p.add_argument(
        "--runs-dir",
        type=Path,
        default=DEFAULT_RUNS_DIR,
        help=f"Path to MLflow runs root. Default: {DEFAULT_RUNS_DIR}.",
    )
    p.add_argument(
        "--exp-name",
        default=DEFAULT_EXP_NAME,
        help=f"MLflow experiment name. Default: {DEFAULT_EXP_NAME}.",
    )
    p.add_argument(
        "--submit",
        action="store_true",
        help=(
            "Actually submit jobs to Slurm. Without this flag, the script "
            "prints the sbatch commands it would run but does not submit "
            "anything (default: dry-run)."
        ),
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    launch(
        seeds=args.seeds,
        architectures=args.architectures,
        strategies=args.strategies,
        data_dir=args.data_dir.resolve(),
        runs_dir=args.runs_dir.resolve(),
        exp_name=args.exp_name,
        submit=args.submit,
    )


if __name__ == "__main__":
    main()
