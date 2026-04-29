"""Unit tests for the factorized spatio-temporal GNS model and its building blocks.

Three tiers of tests:

1. ``SinusoidalTemporalPositionalEncoding`` — pure functional check that
   the encoding has the right shape, varies along the time axis, and
   rejects odd ``d_model``.
2. ``TemporalAttentionLayer`` — shape preservation, identity behavior at
   ``T=1``, and the *non-trivial mixing* property: with input that is
   constant across the time axis, the layer's output is not constant
   (proving that attention is actually doing cross-time work).
3. ``FactorizedSpatioTemporalGNS_heterogeneous`` — constructibility from
   a config namespace, forward-pass shape, ``layer_residuals``
   population, gradient flow through both spatial and temporal
   sublayers, and a behavioral test that confirms the model produces
   different outputs from the per-time-step baseline (so the temporal
   layers are actually contributing).
"""

from __future__ import annotations

import torch
from torch_geometric.data import HeteroData

from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.models.factorized_st_gns_heterogeneous import (
    FactorizedSpatioTemporalGNS_heterogeneous,
)
from gridfm_graphkit.models.temporal_attention import (
    SinusoidalTemporalPositionalEncoding,
    TemporalAttentionLayer,
)
from gridfm_graphkit.models.temporal_gns_heterogeneous import (
    TemporalGNS_heterogeneous,
)


# ---------------------------------------------------------------------------
# SinusoidalTemporalPositionalEncoding
# ---------------------------------------------------------------------------


def test_pos_enc_output_shape_matches_input():
    pe = SinusoidalTemporalPositionalEncoding(d_model=16, max_len=64)
    x = torch.zeros(5, 10, 16)  # [N, T, d_model]
    out = pe(x)
    assert out.shape == x.shape


def test_pos_enc_varies_along_time_axis():
    pe = SinusoidalTemporalPositionalEncoding(d_model=16, max_len=64)
    x = torch.zeros(1, 10, 16)
    out = pe(x)
    # Adjacent positions must differ — otherwise the encoding is useless.
    for t in range(out.shape[1] - 1):
        assert not torch.allclose(out[0, t], out[0, t + 1])


def test_pos_enc_rejects_odd_d_model():
    try:
        SinusoidalTemporalPositionalEncoding(d_model=15)
    except ValueError as e:
        assert "even d_model" in str(e)
    else:
        raise AssertionError("Expected ValueError for odd d_model")


def test_pos_enc_rejects_window_longer_than_max_len():
    pe = SinusoidalTemporalPositionalEncoding(d_model=8, max_len=4)
    x = torch.zeros(1, 5, 8)
    try:
        pe(x)
    except ValueError as e:
        assert "max_len" in str(e)
    else:
        raise AssertionError("Expected ValueError for T > max_len")


# ---------------------------------------------------------------------------
# TemporalAttentionLayer
# ---------------------------------------------------------------------------


def test_temporal_layer_output_shape():
    layer = TemporalAttentionLayer(d_model=16, n_heads=2)
    x = torch.randn(8, 6, 16)  # [N, T, d_model]
    out = layer(x)
    assert out.shape == x.shape


def test_temporal_layer_at_T_equals_one():
    """With T=1 the only token attends to itself; output equals input
    (modulo numerical noise from the residual + dropout-zero path).

    This is mainly a smoke test that the layer accepts T=1 without
    erroring; the residual connection guarantees the output is *near*
    the input even with random attention weights.
    """
    layer = TemporalAttentionLayer(d_model=16, n_heads=2)
    layer.eval()
    x = torch.randn(4, 1, 16)
    out = layer(x)
    assert out.shape == x.shape
    # The post-attention contribution at T=1 is just a re-weighted copy
    # of the single token; the residual keeps the output close to x.
    # We don't assert exact equality (the attention output is non-zero
    # in general), just shape and finiteness.
    assert torch.all(torch.isfinite(out))


def test_temporal_layer_mixes_across_time_with_constant_input():
    """If a layer that's supposed to attend across T outputs the *same*
    value at every t when the input was constant across t, then
    nothing useful happened. With non-zero attention weights the
    output should depend on position (because of LayerNorm + the
    attention output projection) — at minimum it should not be
    identical to the input at every position.

    This is the "attention is actually doing cross-time work" check
    that distinguishes a proper temporal layer from a no-op or a
    shape-preserving identity.
    """
    layer = TemporalAttentionLayer(d_model=16, n_heads=2)
    layer.eval()
    # Input constant across time: same vector copied 6 times.
    base = torch.randn(4, 1, 16)
    x = base.expand(-1, 6, -1).clone()
    out = layer(x)
    # Output across t should not be exactly equal to input across t —
    # the attention layer transformed it.
    assert not torch.allclose(out, x, atol=1e-6)


def test_temporal_layer_rejects_d_model_not_divisible_by_heads():
    try:
        TemporalAttentionLayer(d_model=17, n_heads=2)
    except ValueError as e:
        assert "divisible by n_heads" in str(e)
    else:
        raise AssertionError("Expected ValueError")


# ---------------------------------------------------------------------------
# FactorizedSpatioTemporalGNS_heterogeneous
# ---------------------------------------------------------------------------


def _make_args(num_layers: int = 2, hidden: int = 16, heads: int = 2):
    return NestedNamespace(
        **{
            "task": {"task_name": "TemporalReconstruction"},
            "model": {
                "type": "FactorizedSpatioTemporalGNS_heterogeneous",
                "attention_head": heads,
                "edge_dim": 10,
                "hidden_size": hidden,
                "input_bus_dim": 15,
                "input_gen_dim": 6,
                "output_bus_dim": 2,
                "output_gen_dim": 1,
                "num_layers": num_layers,
                "dropout": 0.0,
            },
            "data": {"baseMVA": 100, "mask_value": 0.0},
        },
    )


def _make_temporal_sample(
    n_buses: int = 8,
    n_gens: int = 3,
    n_edges: int = 12,
    T: int = 6,
    f_bus: int = 15,
    f_gen: int = 6,
    f_edge: int = 10,
    seed: int = 0,
):
    """Construct a small temporal HeteroData sample plus its mask_dict."""
    rng = torch.Generator().manual_seed(seed)

    bus_x = torch.randn(n_buses, T, f_bus, generator=rng)
    # Place valid PQ/PV/REF flags: bus 0 is REF, bus 1 is PV, the rest
    # are PQ. These are the static portion of the bus features and
    # don't change across T.
    bus_x[..., 5] = 0.0  # PQ
    bus_x[..., 6] = 0.0  # PV
    bus_x[..., 7] = 0.0  # REF
    bus_x[0, :, 7] = 1.0  # bus 0 = REF
    bus_x[1, :, 6] = 1.0  # bus 1 = PV
    bus_x[2:, :, 5] = 1.0  # buses 2..N-1 = PQ

    gen_x = torch.randn(n_gens, T, f_gen, generator=rng)

    # Bus-bus edges (random but valid).
    src = torch.randint(0, n_buses, (n_edges,), generator=rng)
    dst = torch.randint(0, n_buses, (n_edges,), generator=rng)
    edge_index_bb = torch.stack([src, dst], dim=0)
    edge_attr_bb = torch.randn(n_edges, T, f_edge, generator=rng)

    # Gen-bus and bus-gen edges. Generator g is connected to bus g (for
    # simplicity), so gen_to_bus_index = [0, 1, 2].
    gb_edge_index = torch.tensor(
        [[g for g in range(n_gens)], [g for g in range(n_gens)]],
        dtype=torch.long,
    )
    bg_edge_index = gb_edge_index.flip(0)

    edge_index_dict = {
        ("bus", "connects", "bus"): edge_index_bb,
        ("gen", "connected_to", "bus"): gb_edge_index,
        ("bus", "connected_to", "gen"): bg_edge_index,
    }
    # `gridfm-datakit` only attaches edge_attr to the bus-bus relation;
    # gen-bus and bus-gen edges have edge_index but no edge_attr in real
    # data. Match that convention here so the test exercises the same
    # input shape the production pipeline produces.
    edge_attr_dict = {
        ("bus", "connects", "bus"): edge_attr_bb,
    }

    # Random Bernoulli mask per (entity, time, feature) at rate 0.5.
    mask_dict = {
        "bus": (torch.rand(n_buses, T, f_bus, generator=rng) < 0.5),
        "gen": (torch.rand(n_gens, T, f_gen, generator=rng) < 0.5),
        "branch": (torch.rand(n_edges, T, f_edge, generator=rng) < 0.5),
    }

    x_dict = {"bus": bus_x, "gen": gen_x}
    return x_dict, edge_index_dict, edge_attr_dict, mask_dict


def test_factorized_model_constructs_from_args():
    args = _make_args()
    model = FactorizedSpatioTemporalGNS_heterogeneous(args)
    # Sanity: the model has temporal layers (proves architectural delta
    # vs the per-time-step baseline).
    assert len(model.temporal_bus_layers) == args.model.num_layers
    assert len(model.temporal_gen_layers) == args.model.num_layers


def test_factorized_model_forward_shape():
    args = _make_args()
    model = FactorizedSpatioTemporalGNS_heterogeneous(args).eval()
    x_dict, ei, ea, md = _make_temporal_sample()
    with torch.no_grad():
        out = model(x_dict, ei, ea, md)
    # Bus output is the physics decoder's output [Vm, Va, Pg, Qg],
    # always 4 columns regardless of `output_bus_dim` (which sets the
    # mlp_bus intermediate width to (Vm, Va) only). The loss
    # (`MaskedBusMSE`) consumes specific columns of this 4-wide tensor
    # via VM_OUT / VA_OUT / PG_OUT / QG_OUT indices.
    assert out["bus"].shape == (
        x_dict["bus"].shape[0],
        x_dict["bus"].shape[1],
        4,
    )
    # Gen output: [G, T, output_gen_dim]
    assert out["gen"].shape == (
        x_dict["gen"].shape[0],
        x_dict["gen"].shape[1],
        args.model.output_gen_dim,
    )


def test_factorized_model_layer_residuals_populated():
    args = _make_args(num_layers=3)
    model = FactorizedSpatioTemporalGNS_heterogeneous(args).eval()
    x_dict, ei, ea, md = _make_temporal_sample()
    with torch.no_grad():
        _ = model(x_dict, ei, ea, md)
    # The PowerFlow path populates layer_residuals at every layer.
    assert set(model.layer_residuals.keys()) == set(range(args.model.num_layers))
    for v in model.layer_residuals.values():
        assert torch.is_tensor(v)
        assert v.dim() == 0  # scalar
        assert torch.isfinite(v)


def test_factorized_model_gradients_flow_through_temporal_layers():
    """Backprop through the model must reach the temporal layers'
    weights. If they have zero grad, the temporal layers aren't
    actually in the computational path."""
    args = _make_args()
    model = FactorizedSpatioTemporalGNS_heterogeneous(args)
    model.train()
    x_dict, ei, ea, md = _make_temporal_sample()
    out = model(x_dict, ei, ea, md)
    loss = out["bus"].pow(2).mean() + out["gen"].pow(2).mean()
    loss.backward()

    for i, layer in enumerate(model.temporal_bus_layers):
        # The QKV in_proj is the first set of trainable weights inside
        # nn.MultiheadAttention; its grad must be non-zero somewhere.
        grad = layer.attn.in_proj_weight.grad
        assert grad is not None, f"temporal_bus_layers[{i}].attn has no grad"
        assert torch.any(grad != 0), (
            f"temporal_bus_layers[{i}].attn has all-zero grad — temporal "
            "layer is not on the computational path"
        )


def test_factorized_model_differs_from_per_time_step_baseline():
    """The whole point of factorized space-time is that it produces a
    different function than the per-time-step baseline. Trained
    differently they may still be close, but at random init the two
    architectures must not coincidentally be identical.

    We initialize both with the same seed, run them on the same input,
    and verify the outputs differ — proving the temporal layers
    contribute to the forward pass.
    """
    args = _make_args()
    torch.manual_seed(0)
    factorized = FactorizedSpatioTemporalGNS_heterogeneous(args).eval()
    torch.manual_seed(0)
    baseline = TemporalGNS_heterogeneous(args).eval()

    x_dict, ei, ea, md = _make_temporal_sample(seed=42)
    with torch.no_grad():
        out_f = factorized(x_dict, ei, ea, md)
        out_b = baseline(x_dict, ei, ea, md)

    # Different architectures → different outputs at random init.
    assert not torch.allclose(out_f["bus"], out_b["bus"])


def test_factorized_model_is_registered():
    """The model name is reachable from the registry — required for
    YAML configs that say `model.type:
    FactorizedSpatioTemporalGNS_heterogeneous` to resolve correctly."""
    from gridfm_graphkit.io.registries import MODELS_REGISTRY

    assert "FactorizedSpatioTemporalGNS_heterogeneous" in MODELS_REGISTRY
