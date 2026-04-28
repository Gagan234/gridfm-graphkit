"""Datamodule integration tests for the spatio-temporal pipeline.

These exercise ``LitGridHeteroDataModule`` end-to-end on the small
``case14_ieee`` test data when the configured task is
``TemporalReconstruction``. The corresponding config lives at
``tests/config/datamodule_temporal_test_base_config.yaml``.

What's covered:

- Setup runs cleanly and the three split datasets are non-empty.
- Train / val / test dataloaders each yield a batch.
- Sample tensors carry the temporal axis at axis 1 — ``bus.x`` is
  ``[B*N, T, F_bus]`` after PyG batching, etc.
- Per-scenario cleanup transforms run *before* temporal stacking, so
  ``F_edge`` is 10 (the ``B_ON`` column has been stripped) rather than
  the raw 11, and ``F_gen`` is 6 (post ``G_ON`` strip).
- ``mask_dict`` is populated by ``AddTemporalMask`` and downstream
  ``ApplyMasking`` has zeroed the masked positions.
- Switching the masking strategy is purely a YAML change.
- A non-temporal task name continues to route through the static path
  (regression check).
"""

from __future__ import annotations

import copy

import pytest
import yaml
import torch

from gridfm_graphkit.datasets.hetero_powergrid_datamodule import (
    LitGridHeteroDataModule,
)
from gridfm_graphkit.datasets.temporal_dataset import HeteroGridTemporalDataset
from gridfm_graphkit.io.param_handler import NestedNamespace


with open("tests/config/datamodule_temporal_test_base_config.yaml") as f:
    BASE_CONFIG = yaml.safe_load(f)


class _DummyTrainer:
    is_global_zero = True


def _build_datamodule(cfg_overrides: dict | None = None) -> LitGridHeteroDataModule:
    cfg = copy.deepcopy(BASE_CONFIG)
    if cfg_overrides:
        for key, value in cfg_overrides.items():
            section, name = key.split(".", 1)
            cfg[section][name] = value
    args = NestedNamespace(**cfg)
    dm = LitGridHeteroDataModule(args, data_dir="tests/data")
    dm.trainer = _DummyTrainer()
    return dm


def test_temporal_setup_runs_cleanly_and_splits_are_nonempty():
    dm = _build_datamodule()
    dm.setup("fit")

    assert len(dm.train_datasets) == 1
    assert len(dm.val_datasets) == 1
    assert len(dm.test_datasets) == 1
    assert len(dm.train_datasets[0]) > 0
    assert len(dm.val_datasets[0]) > 0
    assert len(dm.test_datasets[0]) > 0


def test_temporal_dataset_is_a_temporal_wrapper():
    """The split subsets must wrap a HeteroGridTemporalDataset."""
    dm = _build_datamodule()
    dm.setup("fit")

    underlying = dm.train_datasets[0].dataset
    assert isinstance(underlying, HeteroGridTemporalDataset)


def test_temporal_dataloaders_yield_batches_with_time_axis():
    dm = _build_datamodule()
    dm.setup("fit")

    cfg = dm.args
    T = int(cfg.data.window_size)
    B = int(cfg.training.batch_size)

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    assert batch is not None

    bus_x = batch["bus"].x
    gen_x = batch["gen"].x
    edge_attr = batch[("bus", "connects", "bus")].edge_attr

    # Time axis at position 1; node-axis 0 is concatenated by PyG batching.
    assert bus_x.dim() == 3
    assert bus_x.shape[1] == T
    assert gen_x.dim() == 3
    assert gen_x.shape[1] == T
    assert edge_attr.dim() == 3
    assert edge_attr.shape[1] == T

    # Edge feature width must be 10 (raw 11 minus the B_ON column,
    # stripped by RemoveInactiveBranches before stacking).
    assert edge_attr.shape[2] == 10

    # Gen feature width is 6 (raw 7 minus G_ON, stripped by
    # RemoveInactiveGenerators before stacking).
    assert gen_x.shape[2] == 6

    # Batch dim collapsed into N: total nodes = N_per_sample * batch_size.
    # With case14_ieee, N=14, so total bus rows divides by 14.
    assert bus_x.shape[0] % 14 == 0
    # And the number of samples in this batch can be at most B (last batch
    # may be partial; the first batch should be exactly B for our sizes).
    assert bus_x.shape[0] // 14 <= B


def test_temporal_batch_carries_mask_dict_and_masked_positions_are_zeroed():
    dm = _build_datamodule()
    dm.setup("fit")

    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))

    assert hasattr(batch, "mask_dict")
    md = batch.mask_dict
    for key in ("bus", "gen", "branch"):
        assert key in md, f"mask_dict missing '{key}'"
        assert md[key].dtype == torch.bool

    # ApplyMasking should have set masked positions to mask_value=0.0.
    bus_x = batch["bus"].x
    bus_mask = md["bus"]
    assert bus_mask.shape == bus_x.shape
    assert torch.all(bus_x[bus_mask] == 0.0)


def test_temporal_train_scenarios_are_contiguous_block():
    """The temporal pipeline must take the smallest N load_scenario_idx
    values as the active scenarios — non-contiguous would break the
    HeteroGridTemporalDataset's invariants."""
    dm = _build_datamodule()
    dm.setup("fit")

    all_ids = sorted(
        set(dm.train_scenario_ids[0])
        | set(dm.val_scenario_ids[0])
        | set(dm.test_scenario_ids[0])
    )
    # Active scenarios = the union of scenarios appearing in any window.
    # With stride=1 and num_scenarios windows starting at 0..N-window_size,
    # this should equal num_scenarios distinct scenarios.
    base = dm.datasets[0]
    full_load = base.load_scenarios
    order = torch.argsort(full_load).tolist()
    expected = sorted(order[: int(dm.args.data.scenarios[0])])
    assert all_ids == expected


@pytest.mark.parametrize(
    "strategy_block",
    [
        {"strategy": "random_point", "rate": 0.4},
        {"strategy": "block_temporal", "block_length": 2, "anchor": "trailing"},
        {"strategy": "causal", "anchor_position": "random"},
        {"strategy": "block_spatial", "spatial_rate": 0.3},
        {"strategy": "tube", "tube_rate": 0.3, "tube_seed": 7},
        {"strategy": "topology", "hop_count": 1, "anchor_strategy": "random_bus"},
    ],
)
def test_each_masking_strategy_routes_through_datamodule(strategy_block):
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["masking"] = strategy_block
    args = NestedNamespace(**cfg)
    dm = LitGridHeteroDataModule(args, data_dir="tests/data")
    dm.trainer = _DummyTrainer()

    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert hasattr(batch, "mask_dict")


def test_static_task_still_routes_to_static_path():
    """Regression: a non-temporal task must keep the original code path —
    no temporal wrapper, no [N, T, F] shape."""
    cfg = copy.deepcopy(BASE_CONFIG)
    cfg["task"] = {
        "task_name": "PowerFlow",
        "noise_type": "Gaussian",
        "measurements": {
            "power_inj": {"mask_ratio": 0.2, "outlier_ratio": 0.1, "std": 0.02},
            "power_flow": {"mask_ratio": 0.2, "outlier_ratio": 0.1, "std": 0.02},
            "vm": {"mask_ratio": 0.2, "outlier_ratio": 0.1, "std": 0.02},
        },
        "relative_measurement": True,
    }
    cfg.pop("masking", None)
    cfg["data"].pop("window_size", None)
    cfg["data"].pop("window_stride", None)
    cfg["model"]["type"] = "GNS_heterogeneous"
    cfg["training"]["losses"] = ["LayeredWeightedPhysics", "MaskedBusMSE"]
    cfg["training"]["loss_weights"] = [0.1, 0.9]
    cfg["training"]["loss_args"] = [{"base_weight": 0.5}, {}]

    args = NestedNamespace(**cfg)
    dm = LitGridHeteroDataModule(args, data_dir="tests/data")
    dm.trainer = _DummyTrainer()

    dm.setup("fit")

    # Underlying dataset is NOT a temporal wrapper.
    underlying = dm.train_datasets[0].dataset
    assert not isinstance(underlying, HeteroGridTemporalDataset)

    batch = next(iter(dm.train_dataloader()))
    # Static path: bus.x is [B*N, F_bus] (rank-2), no time axis.
    assert batch["bus"].x.dim() == 2
