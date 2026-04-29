#!/bin/bash
#SBATCH --job-name=gridfm_ablation
#SBATCH --partition=suny
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.err
# ---------------------------------------------------------------------------
# Single-cell ablation training job.
#
# Trains one (architecture × masking-strategy × seed) cell of the
# thesis ablation matrix. Invoked by examples/scripts/launch_ablation_matrix.py;
# can also be submitted manually with:
#
#   sbatch --export=CONFIG=path/to/config.yaml,RUN_NAME=name run_ablation_job.sbatch
#
# Environment variables this script consumes:
#   CONFIG     — path to a temporal-reconstruction YAML config (required).
#   DATA_PATH  — root containing the per-network parquet directories
#                (default: /mnt/lustre/suny/gsapkota/data).
#   LOG_DIR    — MLflow / training logs root
#                (default: /mnt/lustre/suny/gsapkota/runs).
#   EXP_NAME   — MLflow experiment name (default: ablation_case118).
#   RUN_NAME   — MLflow run name (recommended: <arch>__<strategy>__seed<N>).
# ---------------------------------------------------------------------------

set -euo pipefail

# Defaults (override via --export when submitting).
: "${CONFIG:?Must export CONFIG=path/to/config.yaml}"
: "${DATA_PATH:=/mnt/lustre/suny/gsapkota/data}"
: "${LOG_DIR:=/mnt/lustre/suny/gsapkota/runs}"
: "${EXP_NAME:=ablation_case118}"
: "${RUN_NAME:=$(basename "$CONFIG" .yaml)}"

echo "=== gridfm ablation job ==="
echo "Job ID:    $SLURM_JOB_ID"
echo "Hostname:  $(hostname)"
echo "Date:      $(date)"
echo "Config:    $CONFIG"
echo "Data:      $DATA_PATH"
echo "Logs:      $LOG_DIR"
echo "Exp:       $EXP_NAME"
echo "Run:       $RUN_NAME"
echo

# Activate the thesis conda env. Path is fixed because the cluster's
# Lmod modules don't ship with conda — we installed Miniforge ourselves.
source ~/miniforge3/etc/profile.d/conda.sh
conda activate gridfm

cd ~/gridfm-graphkit

# Write the actual command to a log line so failed-job postmortems can
# reproduce the exact invocation.
echo "+ gridfm_graphkit train --config $CONFIG --data_path $DATA_PATH --log_dir $LOG_DIR --exp_name $EXP_NAME --run_name $RUN_NAME"

gridfm_graphkit train \
    --config "$CONFIG" \
    --data_path "$DATA_PATH" \
    --log_dir "$LOG_DIR" \
    --exp_name "$EXP_NAME" \
    --run_name "$RUN_NAME"

echo
echo "=== job complete ==="
echo "Date:      $(date)"
