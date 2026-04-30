"""Non-foundation baseline forecasters for the thesis comparison table.

Three reference models — Linear regression, an MLP, and an LSTM — that
forecast bus voltage magnitude (Vm) and angle (Va) from the past
context window. Unlike the foundation models, these are trained
*directly* on the forecasting objective (no masked-reconstruction
pretraining); they are the trained-but-non-foundation reference
points the thesis compares against.

All three share the same input/output contract:

- Input: ``bus_x`` of shape ``[N, T, F_bus]`` — the same per-window
  bus features the foundation pipeline produces. Only the first
  ``context_len = T - horizon`` time steps are used; the trailing
  ``horizon`` time steps are ignored (they're typically zeroed by
  the trailing-block forecasting mask anyway, but the baselines
  don't depend on the masking).
- Output: predicted ``[Vm, Va]`` for the trailing ``horizon`` time
  steps, shape ``[N, horizon, 2]``.

Weights are shared across all buses — each baseline is the same
function applied to every bus's individual time series. Buses do not
exchange information through the baselines; this is intentional, so
the baseline comparison isolates the value of the *graph-aware* and
*foundation-model* contributions of the thesis.

All three are registered in ``MODELS_REGISTRY`` so the existing CLI /
config plumbing works without modification.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from gridfm_graphkit.io.registries import MODELS_REGISTRY


def _resolve_dims(args) -> tuple[int, int, int, int]:
    """Pull (window_size, horizon, in_features, out_features) from a config namespace.

    Defaults for the production case118 ablation are:
    window_size=12, horizon=4, in_features=15 (post-RemoveInactiveGenerators
    bus feature width), out_features=2 (Vm, Va).
    """
    window_size = int(getattr(args.data, "window_size", 12))
    horizon = int(getattr(args.model, "horizon", 4))
    in_features = int(getattr(args.model, "input_bus_dim", 15))
    out_features = int(getattr(args.model, "output_features", 2))
    return window_size, horizon, in_features, out_features


@MODELS_REGISTRY.register("LinearForecaster")
class LinearForecaster(nn.Module):
    """Per-bus linear regression: past context → future Vm, Va.

    The simplest non-foundation baseline. A single linear layer maps a
    flattened per-bus context vector of shape ``(context_len * F_bus,)``
    to a flattened per-bus future vector of shape ``(horizon * 2,)``.
    Weights are shared across the N buses; each bus is independently
    transformed by the same linear function.

    Parameters: ``context_len * F_bus * horizon * 2 + horizon * 2``.
    For our production config (context_len=8, F_bus=15, horizon=4):
    ~960 trainable parameters.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.window_size, self.horizon, self.in_features, self.out_features = (
            _resolve_dims(args)
        )
        self.context_len = self.window_size - self.horizon
        if self.context_len <= 0:
            raise ValueError(
                f"context_len = window_size({self.window_size}) - "
                f"horizon({self.horizon}) must be > 0",
            )
        self.linear = nn.Linear(
            self.context_len * self.in_features,
            self.horizon * self.out_features,
        )

    def forward(self, bus_x: torch.Tensor) -> torch.Tensor:
        """Forecast Vm, Va for the trailing ``horizon`` time steps.

        Args:
            bus_x: ``[N, T, F_bus]``. The trailing ``horizon`` time
                steps may be zeroed by an upstream mask; we ignore them.

        Returns:
            ``[N, horizon, 2]`` predicted (Vm, Va) per future time step.
        """
        N, T, F = bus_x.shape
        if T < self.context_len:
            raise ValueError(
                f"input has T={T} but context_len={self.context_len} required",
            )
        ctx = bus_x[:, : self.context_len, :].reshape(N, -1)
        out = self.linear(ctx)
        return out.reshape(N, self.horizon, self.out_features)


@MODELS_REGISTRY.register("MLPForecaster")
class MLPForecaster(nn.Module):
    """Per-bus multi-layer perceptron: past context → future Vm, Va.

    A modest MLP with two hidden layers and LeakyReLU activations.
    Same per-bus weight-sharing as :class:`LinearForecaster` — the
    network sees one bus's time series at a time, with no
    inter-bus interaction at the model level.

    Hyperparameters (all from ``args.model``):
    - ``hidden_size`` (default 64): width of the hidden layers.
    - ``dropout`` (default 0.1): dropout between layers.

    Parameters at the production config (context=120, hidden=64,
    output=8): ~16K trainable parameters.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.window_size, self.horizon, self.in_features, self.out_features = (
            _resolve_dims(args)
        )
        self.context_len = self.window_size - self.horizon
        hidden = int(getattr(args.model, "hidden_size", 64))
        dropout = float(getattr(args.model, "dropout", 0.1))

        in_dim = self.context_len * self.in_features
        out_dim = self.horizon * self.out_features
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.LeakyReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, bus_x: torch.Tensor) -> torch.Tensor:
        N, T, F = bus_x.shape
        ctx = bus_x[:, : self.context_len, :].reshape(N, -1)
        out = self.mlp(ctx)
        return out.reshape(N, self.horizon, self.out_features)


@MODELS_REGISTRY.register("LSTMForecaster")
class LSTMForecaster(nn.Module):
    """Per-bus LSTM encoder-decoder for temporal forecasting.

    The classical recurrent baseline. An LSTM encoder reads the
    context window step-by-step, producing a final hidden state.
    A linear decoder then unrolls that hidden state into ``horizon``
    future predictions of (Vm, Va) per bus.

    Two architectural choices:
    1. **Single-shot decoding** (used here, not iterative): the final
       encoder hidden state is mapped directly to the full
       ``horizon * 2`` future tensor by a linear layer. This avoids
       autoregressive error compounding at the cost of less
       expressive multi-step structure. A natural extension is an
       autoregressive decoder; we kept the simpler design to keep the
       baseline straightforward and parameter-comparable to the MLP.
    2. **Per-bus independence**: the LSTM is applied to each bus's
       time series with the same weights (treating N as the batch
       dimension), so the baseline cannot exploit graph topology —
       this isolates the foundation-model + graph contributions in
       the thesis comparison.

    Hyperparameters:
    - ``hidden_size`` (default 64): LSTM hidden width.
    - ``num_layers`` (default 2): stacked LSTM layers.
    - ``dropout`` (default 0.1): inter-layer dropout (only effective
      with num_layers > 1).
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.window_size, self.horizon, self.in_features, self.out_features = (
            _resolve_dims(args)
        )
        self.context_len = self.window_size - self.horizon
        hidden = int(getattr(args.model, "hidden_size", 64))
        num_layers = int(getattr(args.model, "num_layers", 2))
        dropout = float(getattr(args.model, "dropout", 0.1))

        self.encoder = nn.LSTM(
            input_size=self.in_features,
            hidden_size=hidden,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.decoder = nn.Linear(hidden, self.horizon * self.out_features)

    def forward(self, bus_x: torch.Tensor) -> torch.Tensor:
        N, T, F = bus_x.shape
        ctx = bus_x[:, : self.context_len, :]  # [N, context_len, F]
        # nn.LSTM with batch_first=True takes [batch, seq, features];
        # we treat N as the batch dim so each bus is its own sequence.
        outputs, (h_n, c_n) = self.encoder(ctx)
        # h_n: [num_layers, N, hidden]; we use the final layer's
        # final-time-step state.
        last_h = h_n[-1]  # [N, hidden]
        out = self.decoder(last_h)  # [N, horizon * 2]
        return out.reshape(N, self.horizon, self.out_features)
