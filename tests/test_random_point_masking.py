"""Unit tests for the random-point masking strategy."""

from __future__ import annotations

from typing import Dict

import pytest
import torch

from gridfm_graphkit.datasets.temporal_masking import (
    MaskingStrategy,
    RandomPointMasking,
)
from gridfm_graphkit.io.registries import MASKING_STRATEGY_REGISTRY


def _make_tensors(
    N: int = 4,
    T: int = 6,
    G: int = 2,
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
# Registration & abstract-class behavior
# ---------------------------------------------------------------------------


def test_strategy_is_registered() -> None:
    assert "random_point" in MASKING_STRATEGY_REGISTRY


def test_registry_create_returns_instance() -> None:
    """The registry's `create()` factory yields a usable strategy instance."""
    s = MASKING_STRATEGY_REGISTRY.create("random_point", rate=0.3)
    assert isinstance(s, RandomPointMasking)
    assert s.rate == 0.3


def test_abstract_base_cannot_be_instantiated() -> None:
    with pytest.raises(TypeError):
        MaskingStrategy()  # type: ignore[abstract]


# ---------------------------------------------------------------------------
# Shape and dtype invariants
# ---------------------------------------------------------------------------


def test_mask_shape_matches_feature_shape() -> None:
    t = _make_tensors()
    s = RandomPointMasking(rate=0.5)
    m = _build(s, t)
    assert m["bus"].shape == t["x_bus"].shape
    assert m["gen"].shape == t["x_gen"].shape
    assert m["branch"].shape == t["x_edge"].shape


def test_mask_dtype_is_bool() -> None:
    t = _make_tensors()
    s = RandomPointMasking(rate=0.5)
    m = _build(s, t)
    assert m["bus"].dtype == torch.bool
    assert m["gen"].dtype == torch.bool
    assert m["branch"].dtype == torch.bool


# ---------------------------------------------------------------------------
# Statistical correctness
# ---------------------------------------------------------------------------


def test_empirical_rate_matches_target_rate() -> None:
    """Over a large tensor, the empirical mask population matches the target rate."""
    t = _make_tensors(N=64, T=24, G=16, E=128, F_bus=8, F_gen=4, F_edge=6)
    s = RandomPointMasking(rate=0.4)
    m = _build(s, t, seed=123)

    n_total = m["bus"].numel() + m["gen"].numel() + m["branch"].numel()
    n_masked = int(m["bus"].sum() + m["gen"].sum() + m["branch"].sum())
    empirical = n_masked / n_total
    # ~32k positions; std of proportion << 0.01 at p=0.4. 0.02 is comfortable.
    assert abs(empirical - 0.4) < 0.02


def test_per_entity_rates_apply_independently() -> None:
    """When entity_rates is provided, each entity's rate is honored separately."""
    t = _make_tensors(N=64, T=24, G=64, E=64, F_bus=4, F_gen=4, F_edge=4)
    s = RandomPointMasking(
        rate=0.5,  # this is the fallback; entity_rates should override
        entity_rates={"bus": 0.1, "gen": 0.9, "branch": 0.5},
    )
    m = _build(s, t, seed=7)

    bus_rate = m["bus"].float().mean().item()
    gen_rate = m["gen"].float().mean().item()
    branch_rate = m["branch"].float().mean().item()

    assert abs(bus_rate - 0.1) < 0.05
    assert abs(gen_rate - 0.9) < 0.05
    assert abs(branch_rate - 0.5) < 0.05


def test_missing_entity_falls_back_to_global_rate() -> None:
    """Entities absent from entity_rates use the global `rate`."""
    t = _make_tensors(N=64, T=24, G=64, E=64, F_bus=4, F_gen=4, F_edge=4)
    s = RandomPointMasking(rate=0.7, entity_rates={"bus": 0.1})
    m = _build(s, t, seed=11)

    bus_rate = m["bus"].float().mean().item()
    gen_rate = m["gen"].float().mean().item()
    branch_rate = m["branch"].float().mean().item()

    assert abs(bus_rate - 0.1) < 0.05
    assert abs(gen_rate - 0.7) < 0.05
    assert abs(branch_rate - 0.7) < 0.05


# ---------------------------------------------------------------------------
# Determinism and seed-dependence
# ---------------------------------------------------------------------------


def test_determinism_with_fixed_seed() -> None:
    """Two calls with identical seeds produce identical masks."""
    t = _make_tensors()
    s = RandomPointMasking(rate=0.5)
    m1 = _build(s, t, seed=42)
    m2 = _build(s, t, seed=42)
    assert torch.equal(m1["bus"], m2["bus"])
    assert torch.equal(m1["gen"], m2["gen"])
    assert torch.equal(m1["branch"], m2["branch"])


def test_different_seeds_give_different_masks() -> None:
    """Two calls with different seeds produce at least one differing mask."""
    t = _make_tensors()
    s = RandomPointMasking(rate=0.5)
    m1 = _build(s, t, seed=42)
    m2 = _build(s, t, seed=43)
    all_equal = (
        torch.equal(m1["bus"], m2["bus"])
        and torch.equal(m1["gen"], m2["gen"])
        and torch.equal(m1["branch"], m2["branch"])
    )
    assert not all_equal


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_rate_zero_produces_all_false_mask() -> None:
    """With rate=0, no positions are masked."""
    t = _make_tensors()
    s = RandomPointMasking(rate=0.0)
    m = _build(s, t, seed=0)
    assert not m["bus"].any()
    assert not m["gen"].any()
    assert not m["branch"].any()


def test_rate_one_produces_all_true_mask() -> None:
    """With rate=1, every position is masked.

    `torch.rand` samples in [0, 1), so `< 1.0` is True everywhere.
    """
    t = _make_tensors()
    s = RandomPointMasking(rate=1.0)
    m = _build(s, t, seed=0)
    assert m["bus"].all()
    assert m["gen"].all()
    assert m["branch"].all()


def test_invalid_rate_raises() -> None:
    with pytest.raises(ValueError, match="rate"):
        RandomPointMasking(rate=-0.1)
    with pytest.raises(ValueError, match="rate"):
        RandomPointMasking(rate=1.5)


def test_invalid_entity_rate_raises() -> None:
    with pytest.raises(ValueError, match="entity_rates"):
        RandomPointMasking(rate=0.5, entity_rates={"bus": 1.5})
    with pytest.raises(ValueError, match="entity_rates"):
        RandomPointMasking(rate=0.5, entity_rates={"gen": -0.2})
