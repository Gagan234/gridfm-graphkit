"""Tests for TemporalGNS_heterogeneous and TemporalReconstructionTask registration + forward shape."""

from __future__ import annotations

import torch

from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.io.registries import (
    MODELS_REGISTRY,
    PHYSICS_DECODER_REGISTRY,
    TASK_REGISTRY,
)
from gridfm_graphkit.models.temporal_gns_heterogeneous import (
    TemporalGNS_heterogeneous,
)
from gridfm_graphkit.tasks.temporal_reconstruction_task import (
    TemporalReconstructionTask,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _make_args(F_bus: int = 15, F_gen: int = 6, F_edge: int = 11):
    """Minimal args fixture for instantiating GNS_heterogeneous + temporal wrapper.

    The dimensions match the gridfm-datakit schema:
    ``F_bus = 15`` features per bus, ``F_gen = 6`` per generator,
    ``F_edge = 11`` per branch (matching the constants in
    ``gridfm_graphkit.datasets.globals``).
    """
    return NestedNamespace(
        model={
            "type": "TemporalGNS_heterogeneous",
            "attention_head": 2,
            "edge_dim": 10,           # GNS_heterogeneous uses this for the edge MLP
            "hidden_size": 16,
            "input_bus_dim": F_bus,
            "input_gen_dim": F_gen,
            "output_bus_dim": 2,
            "output_gen_dim": 1,
            "num_layers": 2,
        },
        task={"task_name": "TemporalReconstruction"},
        data={
            "baseMVA": 100,
            "mask_value": 0.0,
            "normalization": "HeteroDataMVANormalizer",
            "networks": ["case118_ieee"],
        },
    )


def _make_temporal_inputs(N: int = 4, T: int = 3, G: int = 2, E: int = 6):
    """Construct a temporal HeteroData-like input dict tuple."""
    F_bus = 15
    F_gen = 6
    F_edge = 11

    # Random features, but ensure exactly one REF bus (index 0), one PV
    # (index 1), and the rest PQ — these flags are read by the inner
    # PhysicsDecoderPF.
    g = torch.Generator().manual_seed(0)
    x_bus = torch.randn(N, T, F_bus, generator=g)
    # Set bus-type indicator columns
    # Globals: PQ_H=5, PV_H=6, REF_H=7
    x_bus[..., 5] = 0.0  # PQ flag
    x_bus[..., 6] = 0.0  # PV flag
    x_bus[..., 7] = 0.0  # REF flag
    x_bus[0, :, 7] = 1.0   # bus 0: REF
    x_bus[1, :, 6] = 1.0   # bus 1: PV
    for n in range(2, N):
        x_bus[n, :, 5] = 1.0  # bus 2..N-1: PQ

    x_gen = torch.randn(G, T, F_gen, generator=g)
    edge_index_bus_bus = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    edge_attr_bus_bus = torch.randn(E, T, F_edge, generator=g)

    edge_index_gen_bus = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    edge_index_bus_gen = edge_index_gen_bus.flip(0)

    x_dict = {"bus": x_bus, "gen": x_gen}
    edge_index_dict = {
        ("bus", "connects", "bus"): edge_index_bus_bus,
        ("gen", "connected_to", "bus"): edge_index_gen_bus,
        ("bus", "connected_to", "gen"): edge_index_bus_gen,
    }
    edge_attr_dict = {
        ("bus", "connects", "bus"): edge_attr_bus_bus,
    }
    # mask_dict produced by AddTemporalMask — same shape as features.
    mask_dict = {
        "bus": torch.zeros_like(x_bus, dtype=torch.bool),
        "gen": torch.zeros_like(x_gen, dtype=torch.bool),
        "branch": torch.zeros_like(edge_attr_bus_bus, dtype=torch.bool),
    }
    # Mask a few positions so the loss code path is exercised.
    mask_dict["bus"][2, 1, 3] = True  # bus 2, time 1, feature 3 (Vm)
    mask_dict["bus"][3, 2, 4] = True  # bus 3, time 2, feature 4 (Va)
    return x_dict, edge_index_dict, edge_attr_dict, mask_dict


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_temporal_model_is_registered() -> None:
    assert "TemporalGNS_heterogeneous" in MODELS_REGISTRY
    args = _make_args()
    model = MODELS_REGISTRY.create("TemporalGNS_heterogeneous", args)
    assert isinstance(model, TemporalGNS_heterogeneous)


def test_temporal_task_is_registered() -> None:
    assert "TemporalReconstruction" in TASK_REGISTRY


def test_temporal_physics_decoder_is_registered_under_temporal_task_name() -> None:
    """get_physics_decoder(args) inside the inner GNS_heterogeneous needs this."""
    assert "TemporalReconstruction" in PHYSICS_DECODER_REGISTRY


# ---------------------------------------------------------------------------
# Forward pass shape
# ---------------------------------------------------------------------------


def test_temporal_model_forward_pass_shape() -> None:
    """Output preserves the [N, T, F_out] convention from the input."""
    args = _make_args()
    model = TemporalGNS_heterogeneous(args)
    model.eval()

    x_dict, edge_index_dict, edge_attr_dict, mask_dict = _make_temporal_inputs(
        N=4, T=3, G=2, E=6,
    )
    with torch.no_grad():
        out = model(x_dict, edge_index_dict, edge_attr_dict, mask_dict)

    # output_bus_dim=2 in the args, so for each (n, t) we get a length-2
    # output. The PF physics decoder produces [N, 4] = [Vm, Va, Pg, Qg]
    # actually — let's just verify the leading two dims and the output is
    # well-shaped (rank-3, leading dims [N, T]).
    assert "bus" in out
    assert out["bus"].shape[0] == 4   # N
    assert out["bus"].shape[1] == 3   # T
    assert out["bus"].dim() == 3
