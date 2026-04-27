"""Unit tests for HeteroGridTemporalDataset."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData

from gridfm_graphkit.datasets.temporal_dataset import HeteroGridTemporalDataset


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


class _StubBaseDataset(Dataset):
    """Tiny in-memory base dataset standing in for HeteroGridDatasetDisk.

    Each "scenario" is a HeteroData with a fixed topology and randomly-filled
    bus / gen / edge feature tensors. Used by the unit tests to verify the
    wrapper's behavior in isolation, without depending on the disk-backed
    real dataset.
    """

    def __init__(
        self,
        n_scenarios: int = 10,
        n_buses: int = 4,
        n_gens: int = 2,
        n_edges: int = 6,
        f_bus: int = 3,
        f_gen: int = 2,
        f_edge: int = 4,
        seed: int = 0,
    ) -> None:
        gen_seed = torch.Generator().manual_seed(seed)
        self.scenarios = []

        # Fixed bus-bus topology (one undirected ring with a chord).
        edge_index = torch.tensor(
            [[0, 1, 2, 3, 0, 2], [1, 0, 3, 2, 2, 0]],
            dtype=torch.long,
        )
        gb_edge_index = torch.tensor([[0, 1], [0, 2]], dtype=torch.long)
        bg_edge_index = gb_edge_index.flip(0)

        for i in range(n_scenarios):
            data = HeteroData()
            data["bus"].x = torch.randn(n_buses, f_bus, generator=gen_seed)
            data["bus"].y = data["bus"].x.clone()
            data["gen"].x = torch.randn(n_gens, f_gen, generator=gen_seed)
            data["gen"].y = data["gen"].x.clone()
            data[("bus", "connects", "bus")].edge_index = edge_index
            data[("bus", "connects", "bus")].edge_attr = torch.randn(
                n_edges, f_edge, generator=gen_seed,
            )
            data[("bus", "connects", "bus")].y = torch.randn(
                n_edges, 2, generator=gen_seed,
            )
            data[("gen", "connected_to", "bus")].edge_index = gb_edge_index
            data[("bus", "connected_to", "gen")].edge_index = bg_edge_index
            data["scenario_id"] = torch.tensor([i], dtype=torch.long)
            self.scenarios.append(data)

    def __len__(self) -> int:
        return len(self.scenarios)

    def __getitem__(self, i: int) -> HeteroData:
        return self.scenarios[i]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_window_shape_basic() -> None:
    """Each sample's feature tensors carry an extra time dim at axis 1."""
    base = _StubBaseDataset(n_scenarios=10, n_buses=4, f_bus=3)
    load_idx = torch.arange(10, dtype=torch.long)
    ds = HeteroGridTemporalDataset(base, load_idx, window_size=4)

    assert len(ds) == 7  # 10 - 4 + 1 = 7 windows with stride 1

    sample = ds[0]
    assert sample["bus"].x.shape == (4, 4, 3)  # [N=4, T=4, F=3]
    assert sample["bus"].y.shape == (4, 4, 3)
    assert sample["gen"].x.shape == (2, 4, 2)
    assert sample[("bus", "connects", "bus")].edge_attr.shape == (6, 4, 4)


def test_topology_is_static_in_window() -> None:
    """edge_index has no time dimension (single static topology per window)."""
    base = _StubBaseDataset(n_scenarios=5)
    load_idx = torch.arange(5, dtype=torch.long)
    ds = HeteroGridTemporalDataset(base, load_idx, window_size=3)

    sample = ds[0]
    et = ("bus", "connects", "bus")
    assert sample[et].edge_index.shape == (2, 6)  # NOT [2, T, E]


def test_temporal_ordering_when_load_idx_is_reversed() -> None:
    """Window scenarios are ordered by load_scenario_idx, not by base index."""
    base = _StubBaseDataset(n_scenarios=8)
    # Reverse mapping: base scenario 0 has load_scenario_idx=7, base 1 has 6, etc.
    load_idx = torch.arange(7, -1, -1, dtype=torch.long)
    ds = HeteroGridTemporalDataset(base, load_idx, window_size=3)

    sample = ds[0]
    # The first window after sort should pick the scenarios with the
    # smallest load_scenario_idx values (0, 1, 2), which in the base dataset
    # are at base indices 7, 6, 5 respectively.
    assert sample["window_base_scenario_ids"].tolist() == [7, 6, 5]


def test_window_count_with_stride() -> None:
    base = _StubBaseDataset(n_scenarios=10)
    load_idx = torch.arange(10, dtype=torch.long)
    ds = HeteroGridTemporalDataset(base, load_idx, window_size=4, stride=2)
    # n=10, T=4, stride=2: starts at 0, 2, 4, 6 → 4 windows
    assert len(ds) == 4

    # Disjoint windows when stride == window_size:
    ds_disjoint = HeteroGridTemporalDataset(base, load_idx, window_size=4, stride=4)
    # n=10, T=4, stride=4: starts at 0, 4 → 2 windows (start=8 would not fit)
    assert len(ds_disjoint) == 2


def test_disallows_non_contiguous_load_idx() -> None:
    """If the time indices have a gap, the wrapper refuses to construct."""
    base = _StubBaseDataset(n_scenarios=5)
    load_idx = torch.tensor([0, 1, 3, 4, 5], dtype=torch.long)  # gap at 2
    with pytest.raises(ValueError, match="contiguous"):
        HeteroGridTemporalDataset(base, load_idx, window_size=3)


def test_disallows_too_short_dataset() -> None:
    base = _StubBaseDataset(n_scenarios=2)
    load_idx = torch.tensor([0, 1], dtype=torch.long)
    with pytest.raises(ValueError, match="at least window_size"):
        HeteroGridTemporalDataset(base, load_idx, window_size=3)


def test_index_out_of_range_raises() -> None:
    base = _StubBaseDataset(n_scenarios=5)
    load_idx = torch.arange(5, dtype=torch.long)
    ds = HeteroGridTemporalDataset(base, load_idx, window_size=3)
    with pytest.raises(IndexError):
        _ = ds[len(ds)]
    with pytest.raises(IndexError):
        _ = ds[-1]  # we don't support negative indexing


def test_invalid_window_size_or_stride() -> None:
    base = _StubBaseDataset(n_scenarios=5)
    load_idx = torch.arange(5, dtype=torch.long)
    with pytest.raises(ValueError, match="window_size"):
        HeteroGridTemporalDataset(base, load_idx, window_size=0)
    with pytest.raises(ValueError, match="stride"):
        HeteroGridTemporalDataset(base, load_idx, window_size=2, stride=0)


def test_load_idx_length_mismatch() -> None:
    base = _StubBaseDataset(n_scenarios=5)
    load_idx = torch.arange(4, dtype=torch.long)  # one short
    with pytest.raises(ValueError, match="must match"):
        HeteroGridTemporalDataset(base, load_idx, window_size=2)


def test_topology_drift_raises() -> None:
    """If two scenarios in a window have different topology, the wrapper refuses."""
    base = _StubBaseDataset(n_scenarios=5)
    # Mutate scenario 2's edge_index to break the static-topology invariant.
    et = ("bus", "connects", "bus")
    base.scenarios[2][et].edge_index = base.scenarios[2][et].edge_index.flip(1)
    load_idx = torch.arange(5, dtype=torch.long)
    ds = HeteroGridTemporalDataset(base, load_idx, window_size=3)
    with pytest.raises(ValueError, match="Topology drift"):
        _ = ds[0]


def test_feature_stacking_preserves_per_step_values() -> None:
    """The k-th time slice of the stacked sample equals the k-th underlying scenario."""
    base = _StubBaseDataset(n_scenarios=6)
    load_idx = torch.arange(6, dtype=torch.long)
    ds = HeteroGridTemporalDataset(base, load_idx, window_size=4)

    sample = ds[0]
    for t in range(4):
        # base[t]["bus"].x is [N, F], sample["bus"].x[:, t, :] is also [N, F]
        assert torch.equal(sample["bus"].x[:, t, :], base[t]["bus"].x)
        assert torch.equal(sample["gen"].x[:, t, :], base[t]["gen"].x)
        assert torch.equal(
            sample[("bus", "connects", "bus")].edge_attr[:, t, :],
            base[t][("bus", "connects", "bus")].edge_attr,
        )
