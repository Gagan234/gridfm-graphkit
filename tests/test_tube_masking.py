"""Unit tests for the tube masking strategy (persistent bus subset)."""

from __future__ import annotations

from typing import Dict

import pytest
import torch

from gridfm_graphkit.datasets.temporal_masking import TubeMasking
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
    assert "tube" in MASKING_STRATEGY_REGISTRY
    s = MASKING_STRATEGY_REGISTRY.create("tube", tube_rate=0.2, tube_seed=7)
    assert isinstance(s, TubeMasking)


def test_mask_shape_and_dtype() -> None:
    t = _make_tensors()
    s = TubeMasking(tube_rate=0.4, tube_seed=0)
    m = _build(s, t)
    assert m["bus"].shape == t["x_bus"].shape
    assert m["gen"].shape == t["x_gen"].shape
    assert m["branch"].shape == t["x_edge"].shape
    assert m["bus"].dtype == torch.bool


def test_only_buses_are_masked() -> None:
    t = _make_tensors()
    s = TubeMasking(tube_rate=0.5, tube_seed=0)
    m = _build(s, t)
    assert not m["gen"].any()
    assert not m["branch"].any()


def test_n_masked_matches_floor_of_rate_times_N() -> None:
    t = _make_tensors(N=10)
    s = TubeMasking(tube_rate=0.3, tube_seed=0)
    m = _build(s, t)
    bus_mask = m["bus"][:, 0, 0]
    assert bus_mask.sum().item() == int(0.3 * 10)


def test_subset_is_persistent_across_calls() -> None:
    """The defining property of tube masking: same subset every call.

    We give different `rng` seeds at each call; the tube subset must not
    change because tube uses an internal fixed seed independent of the
    caller's rng.
    """
    t = _make_tensors(N=20)
    s = TubeMasking(tube_rate=0.4, tube_seed=42)

    m1 = _build(s, t, seed=0)
    m2 = _build(s, t, seed=1)
    m3 = _build(s, t, seed=2)

    bus_mask_1 = m1["bus"][:, 0, 0]
    bus_mask_2 = m2["bus"][:, 0, 0]
    bus_mask_3 = m3["bus"][:, 0, 0]

    assert torch.equal(bus_mask_1, bus_mask_2)
    assert torch.equal(bus_mask_2, bus_mask_3)


def test_different_tube_seeds_give_different_subsets() -> None:
    """Different `tube_seed` values produce different (but each persistent) subsets."""
    t = _make_tensors(N=20)
    s_a = TubeMasking(tube_rate=0.4, tube_seed=0)
    s_b = TubeMasking(tube_rate=0.4, tube_seed=1)

    m_a = _build(s_a, t)
    m_b = _build(s_b, t)

    bus_a = m_a["bus"][:, 0, 0]
    bus_b = m_b["bus"][:, 0, 0]
    # Same count, different specific subset.
    assert bus_a.sum() == bus_b.sum()
    assert not torch.equal(bus_a, bus_b)


def test_caller_rng_is_irrelevant_to_subset() -> None:
    """Tube ignores the caller's rng for spatial subset selection."""
    t = _make_tensors(N=20)
    s = TubeMasking(tube_rate=0.3, tube_seed=99)

    rng_a = torch.Generator().manual_seed(0)
    rng_b = torch.Generator().manual_seed(99999)

    m_a = s.build_masks(
        t["x_bus"], t["x_gen"], t["x_edge"], t["edge_index"], rng_a,
    )
    m_b = s.build_masks(
        t["x_bus"], t["x_gen"], t["x_edge"], t["edge_index"], rng_b,
    )
    assert torch.equal(m_a["bus"], m_b["bus"])


def test_subset_replicated_across_T_and_F() -> None:
    t = _make_tensors(N=10, T=6, F_bus=3)
    s = TubeMasking(tube_rate=0.4, tube_seed=0)
    m = _build(s, t)
    reference = m["bus"][:, 0, 0]
    for ti in range(6):
        for f in range(3):
            assert torch.equal(m["bus"][:, ti, f], reference)


def test_rate_zero_masks_nothing() -> None:
    t = _make_tensors()
    s = TubeMasking(tube_rate=0.0, tube_seed=0)
    m = _build(s, t)
    assert not m["bus"].any()


def test_rate_one_masks_all_buses() -> None:
    t = _make_tensors()
    s = TubeMasking(tube_rate=1.0, tube_seed=0)
    m = _build(s, t)
    assert m["bus"].all()


def test_invalid_tube_rate_raises() -> None:
    with pytest.raises(ValueError, match="tube_rate"):
        TubeMasking(tube_rate=-0.1, tube_seed=0)
    with pytest.raises(ValueError, match="tube_rate"):
        TubeMasking(tube_rate=1.5, tube_seed=0)
