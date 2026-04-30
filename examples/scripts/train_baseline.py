#!/usr/bin/env python3
"""Train and evaluate one non-foundation baseline (Linear / MLP / LSTM).

Self-contained: reads a YAML config, builds the same temporal datamodule
the foundation models use (so the train/val/test split, normalizer fit,
and per-window context match exactly), trains the requested baseline
on the forecasting objective, evaluates on the test set with the same
trailing-block metric, writes a JSON output.

The baseline reads only ``bus.x`` (the past context) and predicts
``bus.y[:, context_len:, [VM_H, VA_H]]`` for the future horizon. Other
graph elements (gen, branch, edge_index) are not consumed by the
baselines — the comparison is intentionally a non-graph,
non-foundation reference.

Usage::

    python examples/scripts/train_baseline.py \\
        --config /path/to/baseline_case118.yaml \\
        --baseline lstm \\
        --seed 0 \\
        --data-path /mnt/lustre/suny/gsapkota/data \\
        --output /path/to/metrics.json \\
        [--checkpoint-out /path/to/checkpoint.pt]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from gridfm_graphkit.datasets.globals import VA_H, VM_H
from gridfm_graphkit.datasets.hetero_powergrid_datamodule import (
    LitGridHeteroDataModule,
)
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.io.registries import MODELS_REGISTRY


_BASELINE_TYPE_TO_REGISTRY_NAME = {
    "linear": "LinearForecaster",
    "mlp": "MLPForecaster",
    "lstm": "LSTMForecaster",
}


class _DummyTrainer:
    is_global_zero = True
    logger = None


def _build_datamodule(cfg: dict, data_path: str):
    args = NestedNamespace(**cfg)
    dm = LitGridHeteroDataModule(args, data_dir=data_path)
    dm.trainer = _DummyTrainer()
    dm.setup("fit")
    return dm, args


def _select_device(force_cpu: bool) -> torch.device:
    if force_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_model(baseline_type: str, args, device: torch.device):
    registry_name = _BASELINE_TYPE_TO_REGISTRY_NAME[baseline_type]
    args.model.type = registry_name
    model = MODELS_REGISTRY.create(registry_name, args)
    return model.to(device)


def _train_loop(model, dm, args, device, max_epochs: int):
    """Plain PyTorch training loop. No Lightning, no MLflow — keeps the
    baseline path simple and explicit. The resulting checkpoint is
    a vanilla PyTorch state_dict."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.optimizer.learning_rate),
        betas=(float(args.optimizer.beta1), float(args.optimizer.beta2)),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(args.optimizer.lr_decay),
        patience=int(args.optimizer.lr_patience),
    )
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    context_len = model.context_len

    best_val = float("inf")
    history = []
    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for batch in train_loader:
            bus_x = batch["bus"].x.to(device)  # [B*N, T, F]
            bus_y = batch["bus"].y.to(device)  # [B*N, T, 5]
            target = bus_y[:, context_len:, [VM_H, VA_H]]  # [B*N, horizon, 2]

            pred = model(bus_x)
            loss = F.mse_loss(pred, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        train_loss = sum(train_losses) / max(len(train_losses), 1)

        # Validation.
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                bus_x = batch["bus"].x.to(device)
                bus_y = batch["bus"].y.to(device)
                target = bus_y[:, context_len:, [VM_H, VA_H]]
                pred = model(bus_x)
                val_losses.append(F.mse_loss(pred, target).item())
        val_loss = sum(val_losses) / max(len(val_losses), 1)

        scheduler.step(val_loss)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        print(
            f"  epoch {epoch:3d}  train_loss={train_loss:.6f}  "
            f"val_loss={val_loss:.6f}",
        )
        if val_loss < best_val:
            best_val = val_loss

    return history, best_val


def _evaluate(model, test_loader, device):
    """Compute MAE / RMSE / NRMSE on Vm and Va at the future positions
    of every test window. Mirrors the metric definitions in
    forecasting_eval.py for the foundation models so the resulting
    JSONs are directly comparable."""
    model.eval()
    context_len = model.context_len

    vm_err: list[torch.Tensor] = []
    va_err: list[torch.Tensor] = []
    vm_target: list[torch.Tensor] = []
    va_target: list[torch.Tensor] = []
    vm_err_persistence: list[torch.Tensor] = []
    va_err_persistence: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in test_loader:
            bus_x = batch["bus"].x.to(device)
            bus_y = batch["bus"].y.to(device)
            pred = model(bus_x)  # [B*N, horizon, 2]

            true_vm = bus_y[:, context_len:, VM_H]  # [B*N, horizon]
            true_va = bus_y[:, context_len:, VA_H]
            pred_vm = pred[..., 0]
            pred_va = pred[..., 1]

            # Persistence: future = last observed value, broadcast.
            # The "last observed" value comes from bus_y at position
            # context_len - 1 (the final unmasked position).
            last_vm = bus_y[:, context_len - 1, VM_H : VM_H + 1].expand_as(true_vm)
            last_va = bus_y[:, context_len - 1, VA_H : VA_H + 1].expand_as(true_va)

            vm_err.append((pred_vm - true_vm).flatten().cpu())
            va_err.append((pred_va - true_va).flatten().cpu())
            vm_target.append(true_vm.flatten().cpu())
            va_target.append(true_va.flatten().cpu())
            vm_err_persistence.append((last_vm - true_vm).flatten().cpu())
            va_err_persistence.append((last_va - true_va).flatten().cpu())

    return {
        "vm_err": torch.cat(vm_err),
        "va_err": torch.cat(va_err),
        "vm_target": torch.cat(vm_target),
        "va_target": torch.cat(va_target),
        "vm_err_persistence": torch.cat(vm_err_persistence),
        "va_err_persistence": torch.cat(va_err_persistence),
    }


def _summarize(errs: dict) -> dict:
    def _stat(err: torch.Tensor, target: torch.Tensor) -> dict:
        if err.numel() == 0:
            return {"mae": None, "rmse": None, "nrmse": None, "n": 0}
        mae = err.abs().mean().item()
        rmse = err.pow(2).mean().sqrt().item()
        target_std = target.std(unbiased=False).item()
        nrmse = rmse / target_std if target_std > 0 else None
        return {"mae": mae, "rmse": rmse, "nrmse": nrmse, "n": int(err.numel())}

    return {
        "model": {
            "vm": _stat(errs["vm_err"], errs["vm_target"]),
            "va": _stat(errs["va_err"], errs["va_target"]),
        },
        "persistence": {
            "vm": _stat(errs["vm_err_persistence"], errs["vm_target"]),
            "va": _stat(errs["va_err_persistence"], errs["va_target"]),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--baseline",
        choices=list(_BASELINE_TYPE_TO_REGISTRY_NAME.keys()),
        required=True,
    )
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("/mnt/lustre/suny/gsapkota/data"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-out",
        type=Path,
        default=None,
        help="Optional: write the trained state_dict to this path.",
    )
    parser.add_argument(
        "--cpu",
        action="store_true",
        help="Force CPU even if CUDA is available.",
    )
    args = parser.parse_args()

    # Stamp the seed into the config dict so the datamodule split is
    # reproducible per run.
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    cfg["seed"] = int(args.seed)

    torch.manual_seed(args.seed)

    print(f"Loading datamodule from {args.config} (seed={args.seed})...")
    dm, ns_args = _build_datamodule(cfg, str(args.data_path))

    device = _select_device(args.cpu)
    print(f"Using device: {device}")

    model = _make_model(args.baseline, ns_args, device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {model.__class__.__name__} with {n_params} trainable params")

    print("Training...")
    max_epochs = int(ns_args.training.epochs)
    history, best_val = _train_loop(model, dm, ns_args, device, max_epochs)

    print("Evaluating on test set (and computing persistence baseline)...")
    test_loaders = dm.test_dataloader()
    if not test_loaders:
        raise SystemExit("Empty test dataloader list — no networks?")
    errs = _evaluate(model, test_loaders[0], device)
    metrics = _summarize(errs)

    record = {
        "config": str(args.config),
        "baseline": args.baseline,
        "model_type": _BASELINE_TYPE_TO_REGISTRY_NAME[args.baseline],
        "seed": int(args.seed),
        "n_params": int(n_params),
        "best_val_loss": float(best_val),
        "n_masked_positions": int(errs["vm_err"].numel()),
        "metrics": metrics,
        "history": history,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(record, f, indent=2)
    print(f"Wrote metrics: {args.output}")
    print(json.dumps(metrics, indent=2))

    if args.checkpoint_out is not None:
        args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), args.checkpoint_out)
        print(f"Wrote checkpoint: {args.checkpoint_out}")


if __name__ == "__main__":
    main()
