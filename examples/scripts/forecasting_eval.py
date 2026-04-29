#!/usr/bin/env python3
"""Downstream forecasting evaluation for a pre-trained temporal model.

The thesis claim is "spatio-temporal masked pre-training yields a
foundation model whose representations transfer to forecasting." This
script measures the second half of that claim: given a pre-trained
checkpoint, it evaluates how accurately the model forecasts the last
``horizon`` time steps of each test-set window from the unmasked
context.

Forecasting mask: ``block_temporal`` with ``anchor=trailing`` —
deterministically masks the last ``horizon`` time steps of every
test window. This is the same mask the ``block_temporal trailing``
pretraining objective applies; for models pretrained with other
masking strategies (random_point, topology, etc.) the eval is a
*transfer* measurement.

Metrics reported per bus output column (Vm, Va) at masked positions:

- MAE: mean absolute error
- RMSE: root mean squared error
- NRMSE: RMSE / std(targets), a scale-free dispersion measure

Output: JSON file (one per checkpoint), with metadata + metrics.
The aggregator script in this same directory can be extended to
collect these JSONs across the ablation matrix into a downstream
table.

Usage::

    python examples/scripts/forecasting_eval.py \\
        --config /path/to/runs/configs/factorized__random_point__seed0.yaml \\
        --checkpoint /path/to/runs/<exp_id>/<run_id>/artifacts/model/best_model_state_dict.pt \\
        --data-path /mnt/lustre/suny/gsapkota/data \\
        --horizon 4 \\
        --output /tmp/forecasting_metrics.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml

from gridfm_graphkit.datasets.globals import VA_H, VM_H, VM_OUT, VA_OUT
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import (
    LitGridHeteroDataModule,
)
from gridfm_graphkit.io.param_handler import NestedNamespace, get_task


class _DummyTrainer:
    is_global_zero = True
    logger = None


def _override_masking_for_forecasting(cfg: dict, horizon: int) -> dict:
    """Return a copy of ``cfg`` with masking forced to trailing-block."""
    cfg_eval = yaml.safe_load(yaml.dump(cfg))  # deep copy
    cfg_eval["masking"] = {
        "strategy": "block_temporal",
        "block_length": int(horizon),
        "anchor": "trailing",
    }
    return cfg_eval


def _build_eval_datamodule(
    cfg: dict, data_path: str,
) -> LitGridHeteroDataModule:
    args = NestedNamespace(**cfg)
    dm = LitGridHeteroDataModule(args, data_dir=data_path)
    dm.trainer = _DummyTrainer()
    dm.setup("test")
    return dm


def _load_task(cfg: dict, dm: LitGridHeteroDataModule, checkpoint: Path):
    """Instantiate the LightningModule from cfg and load its state_dict."""
    args = NestedNamespace(**cfg)
    task = get_task(args, dm.data_normalizers)
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    missing, unexpected = task.load_state_dict(state, strict=False)
    if missing:
        print(
            f"  ! state_dict missing keys (non-fatal, but suspicious): "
            f"{missing[:5]}{' ...' if len(missing) > 5 else ''}",
            file=sys.stderr,
        )
    if unexpected:
        print(
            f"  ! state_dict unexpected keys: "
            f"{unexpected[:5]}{' ...' if len(unexpected) > 5 else ''}",
            file=sys.stderr,
        )
    task.eval()
    return task


def _accumulate_errors(task, dataloader) -> Dict[str, torch.Tensor]:
    """Collect per-position prediction errors at the masked time steps.

    Returns a dict with keys 'vm_err', 'va_err', 'vm_target', 'va_target',
    each a 1-D tensor of all per-(bus, masked-timestep) entries
    flattened across the test set.
    """
    vm_err = []
    va_err = []
    vm_target = []
    va_target = []

    with torch.no_grad():
        for batch in dataloader:
            x_dict = {nt: batch[nt].x for nt in batch.node_types}
            edge_index_dict = {
                et: batch[et].edge_index for et in batch.edge_types
            }
            # Only the bus-bus relation has edge_attr in real data.
            edge_attr_dict = {}
            for et in batch.edge_types:
                if hasattr(batch[et], "edge_attr"):
                    edge_attr_dict[et] = batch[et].edge_attr
            mask_dict = batch.mask_dict

            out = task.model(
                x_dict, edge_index_dict, edge_attr_dict, mask_dict,
            )

            # Bus prediction shape: [B*N, T, 4] — [Vm, Va, Pg, Qg].
            pred_bus = out["bus"]
            target_bus = batch["bus"].y  # [B*N, T, 5] — [PD, QD, QG, VM, VA]

            # The forecasting mask is identical across (bus, feature)
            # positions for any given window — pick the time-axis pattern
            # from the first bus / first feature. With trailing anchor,
            # this is True for the last `horizon` time steps and False
            # otherwise.
            time_mask = mask_dict["bus"][0, :, 0]  # [T] bool

            # Slice: only score positions where time_mask is True.
            pred_vm = pred_bus[..., VM_OUT][:, time_mask]
            pred_va = pred_bus[..., VA_OUT][:, time_mask]
            true_vm = target_bus[..., VM_H][:, time_mask]
            true_va = target_bus[..., VA_H][:, time_mask]

            vm_err.append((pred_vm - true_vm).flatten())
            va_err.append((pred_va - true_va).flatten())
            vm_target.append(true_vm.flatten())
            va_target.append(true_va.flatten())

    return {
        "vm_err": torch.cat(vm_err),
        "va_err": torch.cat(va_err),
        "vm_target": torch.cat(vm_target),
        "va_target": torch.cat(va_target),
    }


def _summarize_errors(errs: Dict[str, torch.Tensor]) -> Dict[str, dict]:
    """Compute MAE / RMSE / NRMSE per output column."""

    def _stat(err: torch.Tensor, target: torch.Tensor) -> dict:
        if err.numel() == 0:
            return {"mae": None, "rmse": None, "nrmse": None, "n": 0}
        mae = err.abs().mean().item()
        rmse = err.pow(2).mean().sqrt().item()
        target_std = target.std(unbiased=False).item()
        nrmse = rmse / target_std if target_std > 0 else None
        return {
            "mae": mae,
            "rmse": rmse,
            "nrmse": nrmse,
            "n": int(err.numel()),
        }

    return {
        "vm": _stat(errs["vm_err"], errs["vm_target"]),
        "va": _stat(errs["va_err"], errs["va_target"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the training YAML config used for the run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help=(
            "Path to best_model_state_dict.pt (or last.ckpt) — the "
            "Lightning state dict produced by SaveBestModelStateDict."
        ),
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/mnt/lustre/suny/gsapkota/data"),
        help="Root containing per-network parquet directories.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
        help=(
            "Number of trailing time steps to forecast. Should be < "
            "data.window_size in the training config (default: 4)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Path to write JSON metrics output.",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        raise SystemExit(f"Config not found: {args.config}")
    if not args.checkpoint.is_file():
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")
    if not args.data_path.is_dir():
        raise SystemExit(f"Data path is not a directory: {args.data_path}")

    print(f"Loading training config: {args.config}")
    with open(args.config) as f:
        cfg_train = yaml.safe_load(f)

    cfg_eval = _override_masking_for_forecasting(cfg_train, args.horizon)
    if int(cfg_eval["data"]["window_size"]) <= args.horizon:
        raise SystemExit(
            f"horizon={args.horizon} must be strictly less than "
            f"window_size={cfg_eval['data']['window_size']}; otherwise "
            "no context is left to forecast from.",
        )

    print(f"Building eval datamodule with trailing-block forecasting mask "
          f"(horizon={args.horizon})...")
    dm = _build_eval_datamodule(cfg_eval, str(args.data_path))

    print(f"Loading model from checkpoint: {args.checkpoint}")
    task = _load_task(cfg_eval, dm, args.checkpoint)

    test_loaders = dm.test_dataloader()
    if not test_loaders:
        raise SystemExit("Test dataloader list is empty — no networks?")
    print(f"Running forecasting eval on {len(test_loaders)} test loader(s)...")

    errs = _accumulate_errors(task, test_loaders[0])
    metrics = _summarize_errors(errs)

    record = {
        "config": str(args.config),
        "checkpoint": str(args.checkpoint),
        "horizon": int(args.horizon),
        "n_masked_positions": int(errs["vm_err"].numel()),
        "metrics": metrics,
        "training_masking_strategy": cfg_train.get("masking", {}).get(
            "strategy",
        ),
        "model_type": cfg_train.get("model", {}).get("type"),
        "training_seed": cfg_train.get("seed"),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote metrics: {args.output}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
