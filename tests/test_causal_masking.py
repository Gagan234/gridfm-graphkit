"""Unit tests for the causal masking strategy."""

from __future__ import annotations

from typing import Dict

import pytest
import torch

from gridfm_graphkit.datasets.temporal_masking import CausalMasking
from gridfm_graphkit.io.registries import MASKING_STRATEGY_REGISTRY


def _make_tensors(
    N: int = 4,
    T: int = 10,
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
    assert "causal" in MASKING_STRATEGY_REGISTRY
    s = MASKING_STRATEGY_REGISTRY.create("causal", anchor_position=4)
    assert isinstance(s, CausalMasking)


# ---------------------------------------------------------------------------
# Causal monotonicity (the defining invariant)
# ---------------------------------------------------------------------------


def test_causal_monotonicity_with_int_anchor() -> None:
    """If position t is masked then every position t' > t is also masked."""
    T = 12
    t = _make_tensors(T=T)
    s = CausalMasking(anchor_position=5)
    m = _build(s, t)

    time_profile = m["bus"][0, :, 0]
    # First True must be followed by all True; first False must not have any True after the next True
    cumulative_true = torch.cumsum(time_profile.int(), dim=0)
    # For monotonic causal masks: once we see True (cumsum > 0), all subsequent are True.
    # Equivalently, the True positions are an upper-tail interval [t_star+1, T).
    first_true_idx = (
        int(torch.where(time_profile)[0][0].item())
        if time_profile.any()
        else T
    )
    expected = torch.tensor(
        [False] * first_true_idx + [True] * (T - first_true_idx),
    )
    assert torch.equal(time_profile, expected)


def test_int_anchor_at_k_masks_positions_after_k() -> None:
    """anchor_position=k → exactly positions t > k are masked."""
    T = 10
    t = _make_tensors(T=T)
    for k in range(T):
        s = CausalMasking(anchor_position=k)
        m = _build(s, t)
        time_profile = m["bus"][0, :, 0]
        expected = torch.tensor([False] * (k + 1) + [True] * (T - k - 1))
        assert torch.equal(time_profile, expected), f"k={k}"


def test_anchor_at_T_minus_one_masks_nothing() -> None:
    """anchor_position=T-1 → nothing is masked (no positions strictly after T-1)."""
    T = 8
    t = _make_tensors(T=T)
    s = CausalMasking(anchor_position=T - 1)
    m = _build(s, t)
    assert not m["bus"].any()
    assert not m["gen"].any()
    assert not m["branch"].any()


def test_anchor_at_zero_masks_all_after_first() -> None:
    """anchor_position=0 → only t=0 is unmasked, all others masked."""
    T = 8
    t = _make_tensors(T=T)
    s = CausalMasking(anchor_position=0)
    m = _build(s, t)
    time_profile = m["bus"][0, :, 0]
    expected = torch.tensor([False] + [True] * (T - 1))
    assert torch.equal(time_profile, expected)


# ---------------------------------------------------------------------------
# Random anchor
# ---------------------------------------------------------------------------


def test_random_anchor_covers_valid_range() -> None:
    """Across many seeds, the random anchor lands at every t in [0, T) at least once."""
    T = 6
    t = _make_tensors(T=T)
    s = CausalMasking(anchor_position="random")
    seen = set()
    for seed in range(500):
        m = _build(s, t, seed=seed)
        time_profile = m["bus"][0, :, 0]
        if time_profile.any():
            t_star = int(torch.where(time_profile)[0][0].item()) - 1
        else:
            t_star = T - 1  # nothing masked => anchor was at T-1
        seen.add(t_star)
    assert seen == set(range(T))


def test_random_anchor_determinism_with_fixed_seed() -> None:
    t = _make_tensors()
    s = CausalMasking(anchor_position="random")
    m1 = _build(s, t, seed=42)
    m2 = _build(s, t, seed=42)
    assert torch.equal(m1["bus"], m2["bus"])


# ---------------------------------------------------------------------------
# Replication & shape
# ---------------------------------------------------------------------------


def test_mask_replicated_across_all_entities_and_features() -> None:
    """Every (n, f), (g, f), (e, f) sees the same time-axis profile."""
    t = _make_tensors(N=4, T=8, G=3, E=6, F_bus=3, F_gen=2, F_edge=4)
    s = CausalMasking(anchor_position=3)
    m = _build(s, t)

    time_profile = m["bus"][0, :, 0]
    for n in range(4):
        for f in range(3):
            assert torch.equal(m["bus"][n, :, f], time_profile)
    assert torch.equal(m["gen"][0, :, 0], time_profile)
    assert torch.equal(m["branch"][0, :, 0], time_profile)


# ---------------------------------------------------------------------------
# Edge cases / validation
# ---------------------------------------------------------------------------


def test_anchor_position_out_of_range_raises() -> None:
    t = _make_tensors(T=5)
    s = CausalMasking(anchor_position=10)
    with pytest.raises(ValueError, match="anchor_position"):
        _build(s, t)


def test_invalid_anchor_string_raises() -> None:
    with pytest.raises(ValueError, match="anchor_position"):
        CausalMasking(anchor_position="middle")


def test_invalid_anchor_type_raises() -> None:
    with pytest.raises(ValueError, match="anchor_position"):
        CausalMasking(anchor_position=1.5)  # type: ignore[arg-type]


def test_negative_anchor_raises() -> None:
    with pytest.raises(ValueError, match="anchor_position"):
        CausalMasking(anchor_position=-1)
