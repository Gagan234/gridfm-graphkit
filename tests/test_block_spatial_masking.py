"""Unit tests for the block-spatial masking strategy."""

from __future__ import annotations

from typing import Dict

import pytest
import torch

from gridfm_graphkit.datasets.temporal_masking import BlockSpatialMasking
from gridfm_graphkit.io.registries import MASKING_STRATEGY_REGISTRY


def _make_tensors(
    N: int = 10,
    T: int = 6,
    G: int = 4,
    E: int = 12,
    F_bus: int = 3,
    F_gen: int = 2,
    F_edge: int = 4,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    return {
        "x_bus": torch.randn(N, T, F_bus, generator=g),
        "x_gen": torch.randn(G, T, F_gen, generator=g),
        "x_edge": torch.randn(E, T, F_edge, generator=g),
        "edge_index": torch.zeros(2, E, dtype=torch.long),
    }


def _build(strategy, tensors, seed: int = 0) -> Dict[str, torch.Tensor]:
    rng = torch.Generator().manual_seed(seed)
    return strategy.build_masks(
        tensors["x_bus"],
        tensors["x_gen"],
        tensors["x_edge"],
        tensors["edge_index"],
        rng,
    )


def test_strategy_is_registered() -> None:
    assert "block_spatial" in MASKING_STRATEGY_REGISTRY
    s = MASKING_STRATEGY_REGISTRY.create("block_spatial", spatial_rate=0.5)
    assert isinstance(s, BlockSpatialMasking)


def test_mask_shape_matches_feature_shape() -> None:
    t = _make_tensors()
    s = BlockSpatialMasking(spatial_rate=0.4)
    m = _build(s, t)
    assert m["bus"].shape == t["x_bus"].shape
    assert m["gen"].shape == t["x_gen"].shape
    assert m["branch"].shape == t["x_edge"].shape
    assert m["bus"].dtype == torch.bool


def test_only_buses_are_masked() -> None:
    """gen and branch masks are all-False; only bus mask carries any truth."""
    t = _make_tensors()
    s = BlockSpatialMasking(spatial_rate=0.5)
    m = _build(s, t)
    assert not m["gen"].any()
    assert not m["branch"].any()


def test_n_masked_matches_floor_of_rate_times_N() -> None:
    """Exactly floor(spatial_rate * N) buses are masked."""
    t = _make_tensors(N=10, T=6, F_bus=4)
    s = BlockSpatialMasking(spatial_rate=0.3)
    m = _build(s, t)

    # The 1-D bus mask is the projection along (T, F).
    bus_mask = m["bus"][:, 0, 0]
    assert bus_mask.sum().item() == int(0.3 * 10)  # = 3


def test_masked_buses_are_consistent_across_T_and_F() -> None:
    """Every (t, f) shows the same set of masked bus indices."""
    t = _make_tensors(N=10, T=6, F_bus=3)
    s = BlockSpatialMasking(spatial_rate=0.4)
    m = _build(s, t)

    reference = m["bus"][:, 0, 0]
    for ti in range(6):
        for f in range(3):
            assert torch.equal(m["bus"][:, ti, f], reference)


def test_different_seeds_give_different_subsets() -> None:
    """Across different RNG seeds, the masked-bus subset varies."""
    t = _make_tensors(N=10)
    s = BlockSpatialMasking(spatial_rate=0.3)
    m1 = _build(s, t, seed=0)
    m2 = _build(s, t, seed=1)
    # Same shape, bool, same N masked, but the actual subset differs.
    assert m1["bus"].sum() == m2["bus"].sum()
    assert not torch.equal(m1["bus"][:, 0, 0], m2["bus"][:, 0, 0])


def test_determinism_with_fixed_seed() -> None:
    t = _make_tensors()
    s = BlockSpatialMasking(spatial_rate=0.5)
    m1 = _build(s, t, seed=42)
    m2 = _build(s, t, seed=42)
    assert torch.equal(m1["bus"], m2["bus"])


def test_rate_zero_masks_no_buses() -> None:
    t = _make_tensors()
    s = BlockSpatialMasking(spatial_rate=0.0)
    m = _build(s, t)
    assert not m["bus"].any()


def test_rate_one_masks_all_buses() -> None:
    t = _make_tensors()
    s = BlockSpatialMasking(spatial_rate=1.0)
    m = _build(s, t)
    assert m["bus"].all()


def test_invalid_spatial_rate_raises() -> None:
    with pytest.raises(ValueError, match="spatial_rate"):
        BlockSpatialMasking(spatial_rate=-0.1)
    with pytest.raises(ValueError, match="spatial_rate"):
        BlockSpatialMasking(spatial_rate=1.5)
