"""Unit tests for the topology (k-hop) masking strategy."""

from __future__ import annotations

from typing import Dict, Tuple

import pytest
import torch

from gridfm_graphkit.datasets.temporal_masking import TopologyMasking
from gridfm_graphkit.io.registries import MASKING_STRATEGY_REGISTRY


# ---------------------------------------------------------------------------
# Fixed test topology
# ---------------------------------------------------------------------------
#
# We test against a small known graph so the BFS results are easy to verify
# by hand. Bus connectivity (undirected, both directions present in
# edge_index per gridfm-graphkit's data convention):
#
#         0 ── 1 ── 2 ── 3
#              │
#              4 ── 5
#
# Adjacency:
#   0: {1}
#   1: {0, 2, 4}
#   2: {1, 3}
#   3: {2}
#   4: {1, 5}
#   5: {4}
#
# k-hop neighborhoods from anchor=1:
#   k=0: {1}
#   k=1: {0, 1, 2, 4}
#   k=2: {0, 1, 2, 3, 4, 5}
#   k>=2: same (graph diameter from bus 1 is 2)
#
# k-hop neighborhoods from anchor=3:
#   k=0: {3}
#   k=1: {2, 3}
#   k=2: {1, 2, 3}
#   k=3: {0, 1, 2, 3, 4}
#   k=4: {0, 1, 2, 3, 4, 5}
# ---------------------------------------------------------------------------


def _make_test_graph_tensors(
    T: int = 4,
    F_bus: int = 3,
    F_gen: int = 2,
    F_edge: int = 4,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Build a minimal HeteroData-shaped tensor dict over the 6-bus test graph."""
    g = torch.Generator().manual_seed(seed)
    N = 6
    G = 2
    # Both directions of each undirected edge:
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 2, 3, 1, 4, 4, 5],
            [1, 0, 2, 1, 3, 2, 4, 1, 5, 4],
        ],
        dtype=torch.long,
    )
    E = edge_index.shape[1]
    return {
        "x_bus": torch.randn(N, T, F_bus, generator=g),
        "x_gen": torch.randn(G, T, F_gen, generator=g),
        "x_edge": torch.randn(E, T, F_edge, generator=g),
        "edge_index": edge_index,
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


def _masked_bus_set(mask_dict: Dict[str, torch.Tensor]) -> set:
    """Extract the set of masked bus indices from a mask dict."""
    bus_mask = mask_dict["bus"][:, 0, 0]
    return set(torch.where(bus_mask)[0].tolist())


# ---------------------------------------------------------------------------
# Registration & basic structure
# ---------------------------------------------------------------------------


def test_strategy_is_registered() -> None:
    assert "topology" in MASKING_STRATEGY_REGISTRY
    s = MASKING_STRATEGY_REGISTRY.create("topology", hop_count=2, anchor_bus=0)
    assert isinstance(s, TopologyMasking)


def test_mask_shape_and_dtype() -> None:
    t = _make_test_graph_tensors()
    s = TopologyMasking(hop_count=1, anchor_bus=1)
    m = _build(s, t)
    assert m["bus"].shape == t["x_bus"].shape
    assert m["gen"].shape == t["x_gen"].shape
    assert m["branch"].shape == t["x_edge"].shape
    assert m["bus"].dtype == torch.bool


def test_only_buses_are_masked() -> None:
    t = _make_test_graph_tensors()
    s = TopologyMasking(hop_count=2, anchor_bus=0)
    m = _build(s, t)
    assert not m["gen"].any()
    assert not m["branch"].any()


# ---------------------------------------------------------------------------
# BFS correctness against the hand-computed reference table
# ---------------------------------------------------------------------------


def test_hop_zero_masks_only_anchor() -> None:
    t = _make_test_graph_tensors()
    for anchor in range(6):
        s = TopologyMasking(hop_count=0, anchor_bus=anchor)
        m = _build(s, t)
        assert _masked_bus_set(m) == {anchor}, (
            f"k=0 anchor={anchor} should mask only itself"
        )


def test_hop_one_from_bus_1() -> None:
    t = _make_test_graph_tensors()
    s = TopologyMasking(hop_count=1, anchor_bus=1)
    m = _build(s, t)
    assert _masked_bus_set(m) == {0, 1, 2, 4}


def test_hop_two_from_bus_1_covers_whole_graph() -> None:
    t = _make_test_graph_tensors()
    s = TopologyMasking(hop_count=2, anchor_bus=1)
    m = _build(s, t)
    assert _masked_bus_set(m) == {0, 1, 2, 3, 4, 5}


def test_hop_three_from_bus_1_does_not_grow() -> None:
    """Beyond the graph's eccentricity from the anchor, larger k adds nothing."""
    t = _make_test_graph_tensors()
    s2 = TopologyMasking(hop_count=2, anchor_bus=1)
    s10 = TopologyMasking(hop_count=10, anchor_bus=1)
    m2 = _build(s2, t)
    m10 = _build(s10, t)
    assert _masked_bus_set(m2) == _masked_bus_set(m10)


def test_hop_one_from_bus_3_isolated_branch() -> None:
    """From bus 3, k=1 only reaches bus 2."""
    t = _make_test_graph_tensors()
    s = TopologyMasking(hop_count=1, anchor_bus=3)
    m = _build(s, t)
    assert _masked_bus_set(m) == {2, 3}


def test_hop_two_from_bus_3_extends_via_bus_2() -> None:
    """From bus 3, k=2 reaches buses {1, 2, 3}."""
    t = _make_test_graph_tensors()
    s = TopologyMasking(hop_count=2, anchor_bus=3)
    m = _build(s, t)
    assert _masked_bus_set(m) == {1, 2, 3}


# ---------------------------------------------------------------------------
# Connectivity invariant (the property that distinguishes topology masking
# from random spatial masking)
# ---------------------------------------------------------------------------


def _is_connected_subgraph(
    nodes: set, edge_pairs: set, anchor: int,
) -> bool:
    """Verify every node in `nodes` is reachable from `anchor` along edges
    whose endpoints both lie in `nodes`."""
    if anchor not in nodes:
        return False
    seen = {anchor}
    frontier = {anchor}
    while frontier:
        next_frontier = set()
        for u in frontier:
            for v in nodes:
                if v == u or v in seen:
                    continue
                if (u, v) in edge_pairs or (v, u) in edge_pairs:
                    next_frontier.add(v)
        seen |= next_frontier
        frontier = next_frontier
    return seen == nodes


def test_masked_region_is_connected_for_every_anchor_and_k() -> None:
    t = _make_test_graph_tensors()
    edge_pairs = set()
    e = t["edge_index"]
    for i in range(e.shape[1]):
        edge_pairs.add((int(e[0, i]), int(e[1, i])))

    for anchor in range(6):
        for k in range(5):
            s = TopologyMasking(hop_count=k, anchor_bus=anchor)
            m = _build(s, t)
            masked = _masked_bus_set(m)
            assert _is_connected_subgraph(masked, edge_pairs, anchor), (
                f"k={k} anchor={anchor}: masked subgraph {masked} is "
                f"not connected to anchor"
            )


# ---------------------------------------------------------------------------
# Random anchor behavior
# ---------------------------------------------------------------------------


def test_random_anchor_visits_every_bus_eventually() -> None:
    t = _make_test_graph_tensors()
    s = TopologyMasking(hop_count=0, anchor_strategy="random_bus")
    seen_anchors = set()
    for seed in range(200):
        m = _build(s, t, seed=seed)
        seen_anchors |= _masked_bus_set(m)  # k=0 → mask = {anchor}
    assert seen_anchors == set(range(6))


def test_random_anchor_determinism_with_fixed_seed() -> None:
    t = _make_test_graph_tensors()
    s = TopologyMasking(hop_count=2, anchor_strategy="random_bus")
    m1 = _build(s, t, seed=42)
    m2 = _build(s, t, seed=42)
    assert torch.equal(m1["bus"], m2["bus"])


# ---------------------------------------------------------------------------
# Replication & shape
# ---------------------------------------------------------------------------


def test_mask_replicated_across_T_and_F() -> None:
    t = _make_test_graph_tensors(T=4, F_bus=3)
    s = TopologyMasking(hop_count=1, anchor_bus=1)
    m = _build(s, t)
    reference = m["bus"][:, 0, 0]
    for ti in range(4):
        for f in range(3):
            assert torch.equal(m["bus"][:, ti, f], reference)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_invalid_hop_count_raises() -> None:
    with pytest.raises(ValueError, match="hop_count"):
        TopologyMasking(hop_count=-1)


def test_invalid_anchor_strategy_raises() -> None:
    with pytest.raises(ValueError, match="anchor_strategy"):
        TopologyMasking(hop_count=2, anchor_strategy="something_else")


def test_anchor_bus_out_of_range_raises() -> None:
    t = _make_test_graph_tensors()
    s = TopologyMasking(hop_count=1, anchor_bus=100)
    with pytest.raises(ValueError, match="anchor_bus"):
        _build(s, t)
