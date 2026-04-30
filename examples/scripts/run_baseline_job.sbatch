#!/bin/bash
#SBATCH --job-name=gridfm_baseline
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# ---------------------------------------------------------------------------
# Single-cell baseline training job (one of linear / mlp / lstm × one seed).
#
# This script does NOT request a GPU at the SBATCH level — the launcher
# adds --gpus-per-node=1 only for the LSTM baseline (linear and MLP run
# fine on CPU and would otherwise consume scarce GPU resources).
#
# Required env vars:
#   CONFIG          path to the baseline YAML config
#   BASELINE        one of {linear, mlp, lstm}
#   SEED            integer seed
#   DATA_PATH       path to per-network parquet directory root
#   OUTPUT          path to write the metrics JSON
#   CHECKPOINT_OUT  path to write the trained state_dict
#
# Submit through examples/scripts/launch_baselines.py — the launcher
# adds --partition and --gpus-per-node correctly per baseline type.
# ---------------------------------------------------------------------------

set -euo pipefail

: "${CONFIG:?Must export CONFIG=path/to/config.yaml}"
: "${BASELINE:?Must export BASELINE=linear|mlp|lstm}"
: "${SEED:?Must export SEED=integer}"
: "${DATA_PATH:=/mnt/lustre/suny/gsapkota/data}"
: "${OUTPUT:?Must export OUTPUT=path/to/metrics.json}"
: "${CHECKPOINT_OUT:=}"

echo "=== gridfm baseline job ==="
echo "Job ID:       $SLURM_JOB_ID"
echo "Hostname:     $(hostname)"
echo "Date:         $(date)"
echo "Config:       $CONFIG"
echo "Baseline:     $BASELINE"
echo "Seed:         $SEED"
echo "Data:         $DATA_PATH"
echo "Output:       $OUTPUT"
echo "Checkpoint:   ${CHECKPOINT_OUT:-<not saved>}"
echo

# Resolve HOME robustly. On the `cpu` partition, the job environment
# strips both $HOME and $USER, so we fall back through every standard
# way to get the running user's identity. `id -un` always works
# because it queries the kernel for the effective UID.
JOB_USER="${USER:-${LOGNAME:-$(id -un)}}"
if [ -z "${HOME:-}" ] || [ "$HOME" = "/" ]; then
    HOME=$(getent passwd "$JOB_USER" | cut -d: -f6)
    export HOME
fi

source "$HOME/miniforge3/etc/profile.d/conda.sh"
conda activate gridfm

cd "$HOME/gridfm-graphkit"

CKPT_ARGS=()
if [ -n "$CHECKPOINT_OUT" ]; then
    CKPT_ARGS=(--checkpoint-out "$CHECKPOINT_OUT")
fi

python examples/scripts/train_baseline.py \
    --config "$CONFIG" \
    --baseline "$BASELINE" \
    --seed "$SEED" \
    --data-path "$DATA_PATH" \
    --output "$OUTPUT" \
    "${CKPT_ARGS[@]}"

echo
echo "=== job complete ==="
echo "Date: $(date)"
