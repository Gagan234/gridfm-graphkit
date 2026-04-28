"""Integration tests for AddTemporalMask + TemporalReconstructionTransforms."""

from __future__ import annotations

from typing import Any, Dict

import pytest
import torch
from torch_geometric.data import HeteroData

from gridfm_graphkit.datasets.task_transforms import (
    TemporalReconstructionTransforms,
)
from gridfm_graphkit.datasets.temporal_masking import AddTemporalMask
from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.io.registries import TRANSFORM_REGISTRY


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_temporal_sample(
    N: int = 4,
    T: int = 6,
    G: int = 2,
    E: int = 6,
    F_bus: int = 3,
    F_gen: int = 2,
    F_edge: int = 4,
    seed: int = 0,
) -> HeteroData:
    """Build a single temporal HeteroData sample matching the wrapper's shape contract."""
    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["bus"].x = torch.randn(N, T, F_bus, generator=g)
    data["gen"].x = torch.randn(G, T, F_gen, generator=g)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    bus_bus = ("bus", "connects", "bus")
    data[bus_bus].edge_index = edge_index
    data[bus_bus].edge_attr = torch.randn(E, T, F_edge, generator=g)
    return data


def _make_args(strategy: str, mask_value: float = 0.0, **strategy_kwargs: Any):
    """Build a NestedNamespace mimicking the YAML-loaded args structure."""
    return NestedNamespace(
        masking={"strategy": strategy, **strategy_kwargs},
        data={"mask_value": mask_value},
    )


# ---------------------------------------------------------------------------
# Compose-chain registration & smoke test
# ---------------------------------------------------------------------------


def test_temporal_reconstruction_compose_is_registered() -> None:
    assert "TemporalReconstruction" in TRANSFORM_REGISTRY
    args = _make_args("random_point", rate=0.3)
    chain = TRANSFORM_REGISTRY.create("TemporalReconstruction", args)
    assert isinstance(chain, TemporalReconstructionTransforms)


# ---------------------------------------------------------------------------
# AddTemporalMask: mask construction
# ---------------------------------------------------------------------------


def test_add_temporal_mask_attaches_mask_dict() -> None:
    data = _make_temporal_sample()
    args = _make_args("random_point", rate=0.5, seed=42)
    transform = AddTemporalMask(args=args)
    out = transform(data)

    assert hasattr(out, "mask_dict")
    assert set(out.mask_dict.keys()) == {"bus", "gen", "branch"}
    assert out.mask_dict["bus"].shape == data["bus"].x.shape
    assert out.mask_dict["gen"].shape == data["gen"].x.shape
    assert out.mask_dict["branch"].shape == data[
        ("bus", "connects", "bus")
    ].edge_attr.shape


@pytest.mark.parametrize(
    "strategy_name,kwargs",
    [
        ("random_point", {"rate": 0.5}),
        ("block_temporal", {"block_length": 3, "anchor": "trailing"}),
        ("causal", {"anchor_position": 2}),
        ("block_spatial", {"spatial_rate": 0.5}),
        ("tube", {"tube_rate": 0.25, "tube_seed": 7}),
        ("topology", {"hop_count": 1, "anchor_bus": 1}),
    ],
)
def test_add_temporal_mask_each_strategy(
    strategy_name: str, kwargs: Dict[str, Any],
) -> None:
    """All six registered strategies are reachable through AddTemporalMask."""
    data = _make_temporal_sample(N=4, T=6)
    args = _make_args(strategy_name, seed=0, **kwargs)
    transform = AddTemporalMask(args=args)
    out = transform(data)

    assert hasattr(out, "mask_dict")
    # Every entity gets a bool mask of matching shape.
    assert out.mask_dict["bus"].dtype == torch.bool
    assert out.mask_dict["gen"].dtype == torch.bool
    assert out.mask_dict["branch"].dtype == torch.bool


def test_add_temporal_mask_seed_is_deterministic() -> None:
    """When `seed` is set in args.masking, two calls produce identical masks."""
    args = _make_args("random_point", rate=0.5, seed=123)
    transform = AddTemporalMask(args=args)

    d1 = _make_temporal_sample(seed=0)
    d2 = _make_temporal_sample(seed=0)
    o1 = transform(d1)
    o2 = transform(d2)

    assert torch.equal(o1.mask_dict["bus"], o2.mask_dict["bus"])
    assert torch.equal(o1.mask_dict["gen"], o2.mask_dict["gen"])
    assert torch.equal(o1.mask_dict["branch"], o2.mask_dict["branch"])


# ---------------------------------------------------------------------------
# Full Compose chain: mask + apply masking (zero out positions)
# ---------------------------------------------------------------------------


def test_full_compose_zeros_masked_positions() -> None:
    """After the chain, every masked position has value `mask_value` (= 0)."""
    data = _make_temporal_sample(seed=0)
    args = _make_args("random_point", rate=0.5, seed=42, mask_value=0.0)

    chain = TemporalReconstructionTransforms(args=args)
    out = chain(data)

    # x_dict positions where mask is True must equal mask_value (0).
    bus_x = out["bus"].x
    bus_m = out.mask_dict["bus"]
    assert torch.all(bus_x[bus_m] == 0.0)

    gen_x = out["gen"].x
    gen_m = out.mask_dict["gen"]
    assert torch.all(gen_x[gen_m] == 0.0)

    edge_x = out[("bus", "connects", "bus")].edge_attr
    edge_m = out.mask_dict["branch"]
    assert torch.all(edge_x[edge_m] == 0.0)


def test_full_compose_with_topology_strategy() -> None:
    """Sanity check that the topology strategy works through the full chain."""
    data = _make_temporal_sample(seed=0)
    args = _make_args(
        "topology", hop_count=1, anchor_bus=1, seed=0, mask_value=0.0,
    )

    chain = TemporalReconstructionTransforms(args=args)
    out = chain(data)

    # With hop_count=1 from anchor=1 on the path graph 0-1-2-3,
    # masked buses should be {0, 1, 2} and gens/edges unmasked.
    masked = set(torch.where(out.mask_dict["bus"][:, 0, 0])[0].tolist())
    assert masked == {0, 1, 2}
    # All bus.x positions for those buses are zeroed.
    for n in masked:
        assert torch.all(out["bus"].x[n] == 0.0)
    # Other buses retain their original (nonzero) values somewhere.
    other_bus_x = out["bus"].x[3]
    assert other_bus_x.abs().sum() > 0


# ---------------------------------------------------------------------------
# Validation: bad config
# ---------------------------------------------------------------------------


def test_unknown_strategy_raises_keyerror() -> None:
    args = _make_args("nonexistent_strategy")
    with pytest.raises(KeyError):
        AddTemporalMask(args=args)


def test_missing_strategy_field_raises() -> None:
    """If `args.masking.strategy` is absent, AddTemporalMask raises."""
    args = NestedNamespace(
        masking={"rate": 0.5},  # no `strategy` key
        data={"mask_value": 0.0},
    )
    with pytest.raises(ValueError, match="strategy"):
        AddTemporalMask(args=args)


def test_unrecognized_kwargs_are_filtered_out() -> None:
    """Unknown kwargs in args.masking are silently ignored, not passed to the strategy."""
    args = _make_args(
        "random_point",
        rate=0.5,
        block_length=99,  # belongs to block_temporal, not random_point — must be filtered
        seed=0,
    )
    transform = AddTemporalMask(args=args)
    data = _make_temporal_sample()
    out = transform(data)
    assert hasattr(out, "mask_dict")  # no TypeError raised
