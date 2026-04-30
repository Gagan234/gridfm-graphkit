"""Unit tests for the non-foundation forecasting baselines.

Three model classes (LinearForecaster, MLPForecaster, LSTMForecaster)
share the same input/output contract: read a per-bus context window
of bus features, predict (Vm, Va) for the trailing horizon. The tests
verify shapes, gradient flow, and that each model produces different
output (so they're not accidentally collapsed into the same function).
"""

from __future__ import annotations

import torch

from gridfm_graphkit.io.param_handler import NestedNamespace
from gridfm_graphkit.io.registries import MODELS_REGISTRY
from gridfm_graphkit.models.baselines import (
    LinearForecaster,
    LSTMForecaster,
    MLPForecaster,
)


def _make_args(model_type: str = "LinearForecaster"):
    return NestedNamespace(
        **{
            "data": {"window_size": 12},
            "model": {
                "type": model_type,
                "horizon": 4,
                "input_bus_dim": 15,
                "output_features": 2,
                "hidden_size": 64,
                "num_layers": 2,
                "dropout": 0.0,
            },
        },
    )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_all_three_baselines_are_registered():
    assert "LinearForecaster" in MODELS_REGISTRY
    assert "MLPForecaster" in MODELS_REGISTRY
    assert "LSTMForecaster" in MODELS_REGISTRY


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_linear_forecaster_forward_shape():
    args = _make_args("LinearForecaster")
    model = LinearForecaster(args).eval()
    bus_x = torch.randn(8, 12, 15)  # [N=8, T=12, F=15]
    with torch.no_grad():
        out = model(bus_x)
    assert out.shape == (8, 4, 2)


def test_mlp_forecaster_forward_shape():
    args = _make_args("MLPForecaster")
    model = MLPForecaster(args).eval()
    bus_x = torch.randn(8, 12, 15)
    with torch.no_grad():
        out = model(bus_x)
    assert out.shape == (8, 4, 2)


def test_lstm_forecaster_forward_shape():
    args = _make_args("LSTMForecaster")
    model = LSTMForecaster(args).eval()
    bus_x = torch.randn(8, 12, 15)
    with torch.no_grad():
        out = model(bus_x)
    assert out.shape == (8, 4, 2)


# ---------------------------------------------------------------------------
# Param counts (sanity that the baselines have parameters of the expected
# orders of magnitude — useful for the thesis comparison column on
# parameter count).
# ---------------------------------------------------------------------------


def test_linear_forecaster_param_count():
    args = _make_args("LinearForecaster")
    model = LinearForecaster(args)
    n = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # context_len * F * horizon * 2 + horizon * 2 = 8*15*8 + 8 = 968
    assert 900 < n < 1100


def test_mlp_forecaster_has_more_params_than_linear():
    args = _make_args()
    n_linear = sum(p.numel() for p in LinearForecaster(args).parameters())
    n_mlp = sum(p.numel() for p in MLPForecaster(args).parameters())
    assert n_mlp > n_linear


def test_lstm_forecaster_has_more_params_than_linear():
    args = _make_args()
    n_linear = sum(p.numel() for p in LinearForecaster(args).parameters())
    n_lstm = sum(p.numel() for p in LSTMForecaster(args).parameters())
    assert n_lstm > n_linear


# ---------------------------------------------------------------------------
# Behavioral sanity: trailing-block masking should not affect the
# baselines' output, since they read only the first context_len steps.
# ---------------------------------------------------------------------------


def test_baselines_ignore_masked_positions():
    """The trailing horizon time steps may be zeroed by an upstream
    block_temporal trailing mask. The baselines should produce the
    same output regardless of what's in those positions."""
    args = _make_args()
    for cls in (LinearForecaster, MLPForecaster, LSTMForecaster):
        model = cls(args).eval()
        # Two inputs that differ only in the trailing horizon.
        bus_x_a = torch.randn(4, 12, 15)
        bus_x_b = bus_x_a.clone()
        bus_x_b[:, 8:, :] = torch.randn(4, 4, 15)  # randomize trailing block
        with torch.no_grad():
            out_a = model(bus_x_a)
            out_b = model(bus_x_b)
        # Same context → same prediction.
        assert torch.allclose(out_a, out_b), (
            f"{cls.__name__} output depends on the trailing horizon — "
            "but the baseline should only read the context"
        )


# ---------------------------------------------------------------------------
# Gradient flow
# ---------------------------------------------------------------------------


def test_baselines_are_differentiable():
    args = _make_args()
    bus_x = torch.randn(4, 12, 15, requires_grad=True)
    target = torch.randn(4, 4, 2)
    for cls in (LinearForecaster, MLPForecaster, LSTMForecaster):
        model = cls(args).train()
        out = model(bus_x)
        loss = (out - target).pow(2).mean()
        loss.backward()
        # At least one trainable parameter should have a non-zero grad.
        nonzero_grad = any(
            p.grad is not None and torch.any(p.grad != 0)
            for p in model.parameters()
            if p.requires_grad
        )
        assert nonzero_grad, f"{cls.__name__} has no gradient flowing"


# ---------------------------------------------------------------------------
# Three baselines should not produce identical outputs at random init
# ---------------------------------------------------------------------------


def test_three_baselines_produce_different_outputs():
    args = _make_args()
    bus_x = torch.randn(4, 12, 15)
    torch.manual_seed(0)
    out_linear = LinearForecaster(args).eval()(bus_x)
    torch.manual_seed(0)
    out_mlp = MLPForecaster(args).eval()(bus_x)
    torch.manual_seed(0)
    out_lstm = LSTMForecaster(args).eval()(bus_x)
    # Three different architectures must produce three different outputs.
    assert not torch.allclose(out_linear, out_mlp)
    assert not torch.allclose(out_linear, out_lstm)
    assert not torch.allclose(out_mlp, out_lstm)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_linear_forecaster_rejects_too_short_input():
    args = _make_args()
    model = LinearForecaster(args).eval()
    bus_x = torch.randn(4, 5, 15)  # T=5 < context_len=8
    try:
        with torch.no_grad():
            model(bus_x)
    except ValueError as e:
        assert "context_len" in str(e)
    else:
        raise AssertionError("Expected ValueError on too-short input")


def test_baselines_rejects_zero_context():
    """If horizon == window_size, context_len = 0 — should fail at construction."""
    args = NestedNamespace(
        **{
            "data": {"window_size": 4},
            "model": {
                "type": "LinearForecaster",
                "horizon": 4,
                "input_bus_dim": 15,
                "output_features": 2,
            },
        },
    )
    try:
        LinearForecaster(args)
    except ValueError as e:
        assert "context_len" in str(e)
    else:
        raise AssertionError("Expected ValueError on zero context_len")
