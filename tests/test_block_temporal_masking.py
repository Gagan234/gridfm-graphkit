"""Unit tests for the block-temporal masking strategy."""

from __future__ import annotations

from typing import Dict

import pytest
import torch

from gridfm_graphkit.datasets.temporal_masking import BlockTemporalMasking
from gridfm_graphkit.io.registries import MASKING_STRATEGY_REGISTRY


def _make_tensors(
    N: int = 4,
    T: int = 12,
    G: int = 3,
    E: int = 8,
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


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_strategy_is_registered() -> None:
    assert "block_temporal" in MASKING_STRATEGY_REGISTRY
    s = MASKING_STRATEGY_REGISTRY.create(
        "block_temporal", block_length=4, anchor="trailing",
    )
    assert isinstance(s, BlockTemporalMasking)


# ---------------------------------------------------------------------------
# Shape & dtype
# ---------------------------------------------------------------------------


def test_mask_shape_matches_feature_shape() -> None:
    t = _make_tensors()
    s = BlockTemporalMasking(block_length=4)
    m = _build(s, t)
    assert m["bus"].shape == t["x_bus"].shape
    assert m["gen"].shape == t["x_gen"].shape
    assert m["branch"].shape == t["x_edge"].shape
    assert m["bus"].dtype == torch.bool


# ---------------------------------------------------------------------------
# Block-shape invariants
# ---------------------------------------------------------------------------


def test_block_is_contiguous_in_time() -> None:
    """The True positions in the time axis form a contiguous run of length L."""
    t = _make_tensors(T=20)
    s = BlockTemporalMasking(block_length=5)
    m = _build(s, t, seed=42)

    # Project mask to a 1-D time profile (any (n, f) gives the same).
    time_profile = m["bus"][0, :, 0]
    assert time_profile.sum().item() == 5  # exactly L masked time steps

    # Find the True run; verify it's a single contiguous block of length 5.
    indices = torch.where(time_profile)[0]
    assert torch.equal(indices, torch.arange(indices[0], indices[0] + 5))


def test_block_is_replicated_across_all_entities_and_features() -> None:
    """Every (n, f) (and (g, f), (e, f)) sees the *same* time-axis pattern."""
    t = _make_tensors(N=4, T=10, G=3, E=8, F_bus=3, F_gen=2, F_edge=4)
    s = BlockTemporalMasking(block_length=3)
    m = _build(s, t, seed=11)

    time_profile = m["bus"][0, :, 0]
    # Every bus, every feature: same time pattern
    for n in range(4):
        for f in range(3):
            assert torch.equal(m["bus"][n, :, f], time_profile)
    # Generators and edges: same pattern too
    assert torch.equal(m["gen"][0, :, 0], time_profile)
    assert torch.equal(m["branch"][0, :, 0], time_profile)


def test_anchor_trailing_places_block_at_end() -> None:
    """With anchor='trailing', the block always sits at the last L time steps."""
    t = _make_tensors(T=10)
    s = BlockTemporalMasking(block_length=4, anchor="trailing")
    m = _build(s, t, seed=0)

    time_profile = m["bus"][0, :, 0]
    expected = torch.tensor([False] * 6 + [True] * 4)
    assert torch.equal(time_profile, expected)


def test_anchor_random_covers_valid_range() -> None:
    """Across many seeds, the random anchor lands at every valid t0 at least once."""
    T = 8
    L = 3
    t = _make_tensors(T=T)
    s = BlockTemporalMasking(block_length=L, anchor="random")

    seen_t0 = set()
    for seed in range(1000):
        m = _build(s, t, seed=seed)
        time_profile = m["bus"][0, :, 0]
        t0 = int(torch.where(time_profile)[0][0].item())
        seen_t0.add(t0)

    # Valid t0 range is [0, T - L] inclusive = [0, 5] for T=8, L=3.
    assert seen_t0 == set(range(T - L + 1))


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_block_length_equal_to_T_masks_everything() -> None:
    t = _make_tensors(T=5)
    s = BlockTemporalMasking(block_length=5)
    m = _build(s, t)
    assert m["bus"].all()
    assert m["gen"].all()
    assert m["branch"].all()


def test_block_length_one_masks_one_step() -> None:
    t = _make_tensors(T=5)
    s = BlockTemporalMasking(block_length=1)
    m = _build(s, t, seed=0)
    time_profile = m["bus"][0, :, 0]
    assert time_profile.sum().item() == 1


def test_block_length_too_large_raises() -> None:
    t = _make_tensors(T=5)
    s = BlockTemporalMasking(block_length=10)
    with pytest.raises(ValueError, match="exceeds sample T"):
        _build(s, t)


def test_invalid_block_length_raises() -> None:
    with pytest.raises(ValueError, match="block_length"):
        BlockTemporalMasking(block_length=0)
    with pytest.raises(ValueError, match="block_length"):
        BlockTemporalMasking(block_length=-3)


def test_invalid_anchor_raises() -> None:
    with pytest.raises(ValueError, match="anchor"):
        BlockTemporalMasking(block_length=3, anchor="middle")


def test_determinism_with_fixed_seed() -> None:
    t = _make_tensors()
    s = BlockTemporalMasking(block_length=4)
    m1 = _build(s, t, seed=42)
    m2 = _build(s, t, seed=42)
    assert torch.equal(m1["bus"], m2["bus"])
