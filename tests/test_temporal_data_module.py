"""Datamodule integration tests for the spatio-temporal pipeline.

These exercise ``LitGridHeteroDataModule`` end-to-end on a synthetic
QSTS dataset built fresh in a tmp_path each test session — fixed
topology, contiguous ``load_scenario_idx`` 0..N-1, realistic feature
values for the normalizer to fit on. The synthetic data is the *right*
test fixture for the temporal pipeline because the upstream
``case14_ieee`` test data is single-time-point topology-perturbation
data, not a real time series.

What's covered:

- Setup runs cleanly and the three split datasets are non-empty.
- Train / val / test dataloaders each yield a batch.
- Sample tensors carry the temporal axis at axis 1 — ``bus.x`` is
  ``[B*N, T, F_bus]`` after PyG batching, etc.
- Per-scenario cleanup transforms run *before* temporal stacking, so
  ``F_edge`` is 10 (the ``B_ON`` column has been stripped) and
  ``F_gen`` is 6 (post ``G_ON`` strip).
- ``mask_dict`` is populated by ``AddTemporalMask`` and downstream
  ``ApplyMasking`` has zeroed the masked positions.
- Switching the masking strategy is purely a config change.
- A non-temporal task name continues to route through the static path
  (regression check using the upstream case14_ieee fixture).
- Direct unit tests for ``_pick_contiguous_temporal_block`` covering
  clean input, dedup, gap-aware run selection, and edge cases.
"""

from __future__ import annotations

import copy
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from gridfm_graphkit.datasets.hetero_powergrid_datamodule import (
    LitGridHeteroDataModule,
)
from gridfm_graphkit.datasets.temporal_dataset import HeteroGridTemporalDataset
from gridfm_graphkit.io.param_handler import NestedNamespace


class _DummyTrainer:
    is_global_zero = True


# ---------------------------------------------------------------------------
# Synthetic QSTS test data
# ---------------------------------------------------------------------------


def _build_synthetic_qsts_data(out_dir, n_scenarios: int = 30) -> str:
    """Materialize a small fixed-topology QSTS dataset on disk.

    Produces ``out_dir/synth_qsts/raw/{bus,gen,branch}_data.parquet``
    matching the schema that ``HeteroGridDatasetDisk`` and
    ``HeteroDataMVANormalizer`` expect:
    contiguous ``load_scenario_idx`` 0..N-1, one scenario per time
    step, all branches active (``br_status=1``) and all generators on
    (``in_service=1``) so topology is invariant across the window. Bus
    types: bus 0 is REF, gen-buses 1/2/5/7 are PV, the rest are PQ.

    Returns the network name; the caller already knows ``out_dir``.
    """
    n_buses = 14
    # Ring + chord — small but connected and non-trivial.
    base_branches = [(i, (i + 1) % n_buses) for i in range(n_buses)] + [
        (0, 7),
    ]
    gen_buses = [0, 1, 2, 5, 7]

    rng = np.random.default_rng(42)

    bus_rows = []
    gen_rows = []
    branch_rows = []

    for s in range(n_scenarios):
        # Slowly varying load profile (sinusoidal + bounded noise).
        load_scale = 1.0 + 0.3 * np.sin(2 * np.pi * s / max(n_scenarios, 1))

        for b in range(n_buses):
            is_ref = 1 if b == 0 else 0
            is_pv = 1 if b in gen_buses[1:] else 0
            is_pq = 1 - is_ref - is_pv

            bus_rows.append(
                {
                    "scenario": s,
                    "bus": b,
                    "load_scenario_idx": s,
                    "Pd": float(load_scale * (1.0 + 0.5 * rng.random()) * 10.0),
                    "Qd": float(load_scale * (0.5 + 0.3 * rng.random()) * 5.0),
                    "Qg": 0.0,
                    "Vm": float(1.0 + 0.05 * rng.standard_normal()),
                    "Va": float(5.0 * rng.standard_normal()),
                    "PQ": is_pq,
                    "PV": is_pv,
                    "REF": is_ref,
                    "min_vm_pu": 0.9,
                    "max_vm_pu": 1.1,
                    "GS": 0.0,
                    "BS": 0.0,
                    "vn_kv": 138.0,
                },
            )

        for b in gen_buses:
            gen_rows.append(
                {
                    "scenario": s,
                    "bus": b,
                    "p_mw": float(load_scale * 50.0 * (1 + 0.2 * rng.random())),
                    "min_p_mw": 0.0,
                    "max_p_mw": 200.0,
                    "cp0_eur": 100.0,
                    "cp1_eur_per_mw": 20.0,
                    "cp2_eur_per_mw2": 0.05,
                    "in_service": 1,
                    "min_q_mvar": -50.0,
                    "max_q_mvar": 50.0,
                },
            )

        for from_b, to_b in base_branches:
            branch_rows.append(
                {
                    "scenario": s,
                    "from_bus": from_b,
                    "to_bus": to_b,
                    "pf": float(10.0 * rng.standard_normal()),
                    "qf": float(5.0 * rng.standard_normal()),
                    "pt": float(10.0 * rng.standard_normal()),
                    "qt": float(5.0 * rng.standard_normal()),
                    "Yff_r": 1.5,
                    "Yff_i": -10.0,
                    "Yft_r": -1.5,
                    "Yft_i": 10.0,
                    "Ytt_r": 1.5,
                    "Ytt_i": -10.0,
                    "Ytf_r": -1.5,
                    "Ytf_i": 10.0,
                    "tap": 1.0,
                    "ang_min": -30.0,
                    "ang_max": 30.0,
                    "rate_a": 100.0,
                    "br_status": 1,
                },
            )

    network = "synth_qsts"
    raw_dir = out_dir / network / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(bus_rows).to_parquet(raw_dir / "bus_data.parquet")
    pd.DataFrame(gen_rows).to_parquet(raw_dir / "gen_data.parquet")
    pd.DataFrame(branch_rows).to_parquet(raw_dir / "branch_data.parquet")

    return network


@pytest.fixture(scope="module")
def synth_qsts_data(tmp_path_factory) -> Tuple[str, str]:
    """Session-shared synthetic QSTS dataset.

    Returns ``(data_dir, network)`` such that
    ``HeteroGridDatasetDisk(data_dir/network)`` finds parquet files
    matching the gridfm-datakit schema.
    """
    data_dir = tmp_path_factory.mktemp("data")
    network = _build_synthetic_qsts_data(data_dir, n_scenarios=30)
    return str(data_dir), network


def _temporal_config_dict(
    network: str, n_scenarios: int = 30, masking: dict | None = None,
) -> dict:
    """Build a TemporalReconstruction config dict for the given network."""
    return {
        "callbacks": {"patience": 100, "tol": 0},
        "task": {"task_name": "TemporalReconstruction"},
        "masking": masking or {"strategy": "random_point", "rate": 0.5},
        "data": {
            "baseMVA": 100,
            "mask_value": 0.0,
            "normalization": "HeteroDataMVANormalizer",
            "networks": [network],
            "scenarios": [n_scenarios],
            "test_ratio": 0.15,
            "val_ratio": 0.15,
            "workers": 0,
            "split_by_load_scenario_idx": True,
            "window_size": 6,
            "window_stride": 1,
        },
        "model": {
            "type": "TemporalGNS_heterogeneous",
            "attention_head": 2,
            "edge_dim": 10,
            "hidden_size": 16,
            "input_bus_dim": 15,
            "input_gen_dim": 6,
            "output_bus_dim": 2,
            "output_gen_dim": 1,
            "num_layers": 2,
        },
        "optimizer": {
            "beta1": 0.9,
            "beta2": 0.999,
            "learning_rate": 0.0005,
            "lr_decay": 0.7,
            "lr_patience": 5,
        },
        "seed": 0,
        "training": {
            "batch_size": 4,
            "epochs": 1,
            "loss_weights": [1.0],
            "losses": ["MaskedBusMSE"],
            "loss_args": [{}],
            "accelerator": "auto",
            "devices": "auto",
            "strategy": "auto",
        },
        "verbose": True,
    }


def _build_datamodule(
    data_dir: str, cfg: dict,
) -> LitGridHeteroDataModule:
    args = NestedNamespace(**cfg)
    dm = LitGridHeteroDataModule(args, data_dir=data_dir)
    dm.trainer = _DummyTrainer()
    return dm


# ---------------------------------------------------------------------------
# Integration tests on the synthetic QSTS data
# ---------------------------------------------------------------------------


def test_temporal_setup_runs_cleanly_and_splits_are_nonempty(synth_qsts_data):
    data_dir, network = synth_qsts_data
    dm = _build_datamodule(data_dir, _temporal_config_dict(network))
    dm.setup("fit")

    assert len(dm.train_datasets) == 1
    assert len(dm.val_datasets) == 1
    assert len(dm.test_datasets) == 1
    assert len(dm.train_datasets[0]) > 0
    assert len(dm.val_datasets[0]) > 0
    assert len(dm.test_datasets[0]) > 0


def test_temporal_dataset_is_a_temporal_wrapper(synth_qsts_data):
    """The split subsets must wrap a HeteroGridTemporalDataset."""
    data_dir, network = synth_qsts_data
    dm = _build_datamodule(data_dir, _temporal_config_dict(network))
    dm.setup("fit")

    underlying = dm.train_datasets[0].dataset
    assert isinstance(underlying, HeteroGridTemporalDataset)


def test_temporal_dataloaders_yield_batches_with_time_axis(synth_qsts_data):
    data_dir, network = synth_qsts_data
    cfg = _temporal_config_dict(network)
    dm = _build_datamodule(data_dir, cfg)
    dm.setup("fit")

    T = int(cfg["data"]["window_size"])
    B = int(cfg["training"]["batch_size"])
    n_buses = 14

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

    # Edge feature width = 10 (raw 11 minus B_ON, stripped by
    # RemoveInactiveBranches before stacking).
    assert edge_attr.shape[2] == 10
    # Gen feature width = 6 (raw 7 minus G_ON, stripped by
    # RemoveInactiveGenerators before stacking).
    assert gen_x.shape[2] == 6

    # Total bus rows in batch divides evenly by N_buses.
    assert bus_x.shape[0] % n_buses == 0
    # Number of samples in the batch is at most B.
    assert bus_x.shape[0] // n_buses <= B


def test_temporal_batch_carries_mask_dict_and_masked_positions_are_zeroed(
    synth_qsts_data,
):
    data_dir, network = synth_qsts_data
    dm = _build_datamodule(data_dir, _temporal_config_dict(network))
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


def test_temporal_splits_cover_a_contiguous_run_of_load_scenario_idx(
    synth_qsts_data,
):
    """Union of train/val/test scenario IDs must map to a contiguous run
    of ``load_scenario_idx`` values, with one scenario per time step.

    With overlapping sliding windows (stride < window_size) every chosen
    scenario appears in at least one window, so the union of split
    scenarios equals the contiguous block selected by
    ``_pick_contiguous_block_grouped_by_topology``.
    """
    data_dir, network = synth_qsts_data
    dm = _build_datamodule(data_dir, _temporal_config_dict(network))
    dm.setup("fit")

    all_ids = sorted(
        set(dm.train_scenario_ids[0])
        | set(dm.val_scenario_ids[0])
        | set(dm.test_scenario_ids[0])
    )
    base = dm.datasets[0]
    chosen_loads = sorted(
        {int(base.load_scenarios[i].item()) for i in all_ids},
    )
    assert chosen_loads == list(
        range(chosen_loads[0], chosen_loads[-1] + 1),
    )
    assert len(chosen_loads) == len(all_ids)


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
def test_each_masking_strategy_routes_through_datamodule(
    synth_qsts_data, strategy_block,
):
    data_dir, network = synth_qsts_data
    cfg = _temporal_config_dict(network, masking=strategy_block)
    dm = _build_datamodule(data_dir, cfg)
    dm.setup("fit")
    batch = next(iter(dm.train_dataloader()))
    assert hasattr(batch, "mask_dict")


# ---------------------------------------------------------------------------
# Static-path regression check uses the upstream case14_ieee fixture,
# which is genuine static (non-temporal) data — exactly what the static
# path is designed for.
# ---------------------------------------------------------------------------


with open("tests/config/datamodule_test_base_config.yaml") as f:
    _STATIC_BASE_CONFIG = yaml.safe_load(f)


def test_static_task_still_routes_to_static_path():
    """Regression: a non-temporal task must keep the original code path —
    no temporal wrapper, no [N, T, F] shape."""
    cfg = copy.deepcopy(_STATIC_BASE_CONFIG)
    args = NestedNamespace(**cfg)
    dm = LitGridHeteroDataModule(args, data_dir="tests/data")
    dm.trainer = _DummyTrainer()

    dm.setup("fit")

    underlying = dm.train_datasets[0].dataset
    assert not isinstance(underlying, HeteroGridTemporalDataset)

    batch = next(iter(dm.train_dataloader()))
    # Static path: bus.x is [B*N, F_bus] (rank-2), no time axis.
    assert batch["bus"].x.dim() == 2


# ---------------------------------------------------------------------------
# Direct unit tests for the contiguous-run picker. Synthetic input,
# zero disk IO — guaranteed signal independent of the on-disk fixture.
# ---------------------------------------------------------------------------


def test_pick_contiguous_block_clean_input():
    ls = torch.tensor([3, 1, 4, 0, 2], dtype=torch.long)
    indices, loads = LitGridHeteroDataModule._pick_contiguous_temporal_block(
        ls, num_scenarios=5,
    )
    assert loads.tolist() == [0, 1, 2, 3, 4]
    assert indices == [3, 1, 4, 0, 2]


def test_pick_contiguous_block_dedupes_keeping_first_per_load_idx():
    # Three topology variants per time step.
    ls = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long)
    indices, loads = LitGridHeteroDataModule._pick_contiguous_temporal_block(
        ls, num_scenarios=10,
    )
    assert loads.tolist() == [0, 1, 2]
    assert indices == [0, 3, 6]


def test_pick_contiguous_block_picks_longest_run_with_gap():
    # Two runs: [0,1,2] of length 3, then [10,11,12,13] of length 4.
    ls = torch.tensor([0, 1, 2, 10, 11, 12, 13], dtype=torch.long)
    indices, loads = LitGridHeteroDataModule._pick_contiguous_temporal_block(
        ls, num_scenarios=10,
    )
    assert loads.tolist() == [10, 11, 12, 13]
    assert indices == [3, 4, 5, 6]


def test_pick_contiguous_block_caps_at_num_scenarios():
    ls = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7], dtype=torch.long)
    indices, loads = LitGridHeteroDataModule._pick_contiguous_temporal_block(
        ls, num_scenarios=3,
    )
    assert loads.tolist() == [0, 1, 2]
    assert indices == [0, 1, 2]


def test_pick_contiguous_block_combines_dedup_and_gap_handling():
    # Run A dedupes to [0, 1] (length 2). Run B dedupes to [5, 6, 7, 8]
    # (length 4). Picker chooses B.
    ls = torch.tensor([0, 0, 1, 1, 5, 5, 6, 7, 8], dtype=torch.long)
    indices, loads = LitGridHeteroDataModule._pick_contiguous_temporal_block(
        ls, num_scenarios=100,
    )
    assert loads.tolist() == [5, 6, 7, 8]
    assert indices == [4, 6, 7, 8]


def test_pick_contiguous_block_singleton_input():
    indices, loads = LitGridHeteroDataModule._pick_contiguous_temporal_block(
        torch.tensor([42], dtype=torch.long), num_scenarios=5,
    )
    assert loads.tolist() == [42]
    assert indices == [0]


def test_pick_contiguous_block_empty_input():
    indices, loads = LitGridHeteroDataModule._pick_contiguous_temporal_block(
        torch.empty(0, dtype=torch.long), num_scenarios=5,
    )
    assert indices == []
    assert len(loads) == 0
