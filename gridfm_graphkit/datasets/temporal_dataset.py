"""Sequence-aware wrapper turning a per-scenario static dataset into temporal windows.

Stacks `window_size` consecutive QSTS scenarios into a single sample whose
feature tensors carry an extra time dimension. Topology is treated as fixed
within a window, consistent with the QSTS assumption used throughout this
project: load varies temporally but the network does not.

Tensor shape convention (the time dimension is inserted at axis 1, *after* the
node/edge dimension, so PyG's default batching — which concatenates along axis
0 — works without a custom collator):

    bus.x          [N, T, F_bus]
    gen.x          [G, T, F_gen]
    edge_attr      [E, T, F_edge]
    edge_index     [2, E]                     (single, shared across the T steps)
    bus.y          [N, T, F_bus_y]            (same convention for targets)
    gen.y          [G, T, F_gen_y]
    edge.y         [E, T, F_edge_y]

The window is selected in `load_scenario_idx` order — i.e., the T scenarios
in any one sample are temporally adjacent in the original load profile from
which the dataset was generated.
"""

from __future__ import annotations

from typing import List

import torch
from torch.utils.data import Dataset
from torch_geometric.data import HeteroData


class HeteroGridTemporalDataset(Dataset):
    """Temporal wrapper turning a per-scenario static dataset into windowed sequences.

    Args:
        base_dataset: an indexable dataset returning one ``HeteroData`` per
            scenario. Each scenario must have feature tensors with shape
            ``[N, F]`` (bus), ``[G, F]`` (gen), ``[E, F]`` (edge), and a
            topology (``edge_index``) that is invariant across the window.
        load_scenario_idx: 1-D long tensor mapping ``base_dataset`` index ->
            time index. The wrapper sorts samples by this and asserts that
            the resulting time indices are contiguous integers (no gaps).
            This corresponds to the ``load_scenario_idx`` column produced by
            ``gridfm-datakit``'s ``agg_load_profile`` generator.
        window_size: number of consecutive time steps per sample (T).
        stride: step between window starts (default 1). With stride 1 the
            windows overlap maximally; with stride T they are disjoint.

    Raises:
        ValueError: if the inputs are inconsistent (length mismatch,
            non-contiguous time indices, dataset shorter than the window,
            or invalid window/stride values).
    """

    def __init__(
        self,
        base_dataset: Dataset,
        load_scenario_idx: torch.Tensor,
        window_size: int,
        stride: int = 1,
    ) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        if stride < 1:
            raise ValueError(f"stride must be >= 1, got {stride}")
        if len(load_scenario_idx) != len(base_dataset):
            raise ValueError(
                "load_scenario_idx has length "
                f"{len(load_scenario_idx)} but base_dataset has length "
                f"{len(base_dataset)}; they must match.",
            )

        self.base_dataset = base_dataset
        self.window_size = int(window_size)
        self.stride = int(stride)

        # Sort base scenarios by load_scenario_idx (temporal order).
        # `_sorted_scenario_ids[k]` is the index into `base_dataset` of the
        # scenario whose `load_scenario_idx` is the k-th smallest.
        order = torch.argsort(load_scenario_idx)
        self._sorted_scenario_ids: List[int] = order.tolist()

        # Sanity: time indices must form a contiguous run when sorted.
        sorted_idx = load_scenario_idx[order]
        if not torch.equal(
            sorted_idx,
            torch.arange(
                int(sorted_idx[0].item()),
                int(sorted_idx[-1].item()) + 1,
                dtype=sorted_idx.dtype,
            ),
        ):
            raise ValueError(
                "load_scenario_idx is not contiguous after sort; the wrapper "
                "requires a fully temporally-contiguous dataset (no gaps).",
            )

        n = len(self._sorted_scenario_ids)
        if n < self.window_size:
            raise ValueError(
                f"base_dataset has only {n} scenarios; need at least "
                f"window_size={self.window_size}",
            )

        # Number of valid windows.
        self._num_windows = 1 + (n - self.window_size) // self.stride

    def __len__(self) -> int:
        return self._num_windows

    def __getitem__(self, idx: int) -> HeteroData:
        if idx < 0 or idx >= self._num_windows:
            raise IndexError(
                f"index {idx} out of range for {self._num_windows} windows",
            )

        start = idx * self.stride
        scenario_ids = self._sorted_scenario_ids[start : start + self.window_size]

        # Load each scenario from the base dataset (already normalized and
        # transformed by whatever pre-existing pipeline the base has).
        scenarios = [self.base_dataset[sid] for sid in scenario_ids]
        first = scenarios[0]

        sample = HeteroData()

        # --- topology: copied once, sanity-checked across the window ---
        for et in first.edge_types:
            sample[et].edge_index = first[et].edge_index
            for other in scenarios[1:]:
                if not torch.equal(other[et].edge_index, first[et].edge_index):
                    raise ValueError(
                        "Topology drift across the window: edge_index for "
                        f"{et} differs between scenarios in this window. The "
                        "QSTS assumption (fixed topology within a sample) is "
                        "violated.",
                    )

        # --- node features: stack along a new time dim at axis 1 ---
        for nt in first.node_types:
            for key in ("x", "y"):
                if key in first[nt]:
                    sample[nt][key] = torch.stack(
                        [s[nt][key] for s in scenarios],
                        dim=1,
                    )

        # --- edge features: stack along a new time dim at axis 1 ---
        for et in first.edge_types:
            for key in ("edge_attr", "y"):
                if key in first[et]:
                    sample[et][key] = torch.stack(
                        [s[et][key] for s in scenarios],
                        dim=1,
                    )

        # --- carry the per-step scenario identifier through for traceability ---
        # Useful for logging "which time steps did this batch see?" and for
        # joining back to the underlying parquet files at analysis time.
        sample["window_base_scenario_ids"] = torch.tensor(
            scenario_ids,
            dtype=torch.long,
        )

        return sample
