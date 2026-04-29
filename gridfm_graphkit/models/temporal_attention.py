"""Reusable temporal-attention building blocks for spatio-temporal GridFM.

Provides:

- :class:`SinusoidalTemporalPositionalEncoding`: fixed (non-learned) sin/cos
  positional encoding along the time axis. Position-aware without adding
  trainable parameters; generalizes to longer windows than seen at
  training time.
- :class:`TemporalAttentionLayer`: pre-norm multi-head self-attention along
  the time dimension, applied independently per node. Each node's time
  series is attended to as if it were a sequence; nodes do not attend to
  each other inside this layer (cross-node mixing is the spatial layer's
  job).

Together they implement the "temporal" half of factorized space-time
attention: alternate spatial graph convolution (existing
:class:`HeteroConv` + :class:`TransformerConv`) with these temporal layers
to obtain a model that mixes information along both the graph topology
and the time axis. This factorization follows STD-MAE (Gao et al.,
IJCAI 2024) and GeoMAE (2025).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalTemporalPositionalEncoding(nn.Module):
    """Fixed sin/cos positional encoding for the time axis.

    Adds a position-dependent offset to the embedding of each time step
    so the temporal attention layer can distinguish positions. Buffer is
    pre-computed up to ``max_len`` and indexed at forward time, so no
    trainable parameters are added.

    Args:
        d_model: embedding dimension. The encoding has the same width.
        max_len: largest window length supported. ``max_len=512`` is
            ample for the QSTS windows in this project (typical T = 6 to
            96); pre-allocating once is cheap.
    """

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(
                "SinusoidalTemporalPositionalEncoding requires an even "
                f"d_model, got {d_model}",
            )
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model),
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as a buffer so .to(device) and state_dict capture it
        # without exposing it as a parameter.
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add sin/cos positional encoding along the time axis.

        Args:
            x: ``[N, T, d_model]``.

        Returns:
            ``x + pe[:T]``, broadcast over the leading ``N`` dimension.
        """
        T = x.shape[1]
        if T > self.pe.shape[0]:
            raise ValueError(
                f"Window length T={T} exceeds positional encoding "
                f"max_len={self.pe.shape[0]}",
            )
        return x + self.pe[:T].unsqueeze(0)


class TemporalAttentionLayer(nn.Module):
    """Pre-norm multi-head self-attention along the time dimension.

    Operates on an input of shape ``[N, T, d_model]`` where ``N`` is
    treated as a batch dimension — each of the ``N`` nodes has its own
    time series that is attended to independently. This is the
    "temporal" factor in factorized space-time attention; the spatial
    factor (cross-node mixing) is handled separately by the graph
    convolution layer.

    Architecture per call:

    1. ``y = LayerNorm(x)``
    2. ``y = MultiheadAttention(y, y, y)``
    3. ``return x + Dropout(y)``

    No feed-forward MLP is included by default — the spatial graph
    layer (``HeteroConv`` with ``TransformerConv``) provides the
    nonlinearity in the factorized stack. Adding an FFN here is a
    natural extension if expressivity becomes a bottleneck.

    Args:
        d_model: feature dimension. Must be divisible by ``n_heads``.
        n_heads: number of attention heads.
        dropout: dropout probability applied to attention weights and
            the output projection. ``0.0`` by default to match the rest
            of the gridfm-graphkit defaults.
    """

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by n_heads={n_heads}",
            )
        self.norm = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply temporal self-attention.

        Args:
            x: ``[N, T, d_model]``.

        Returns:
            ``[N, T, d_model]`` — same shape, with each time step
            updated by attending across the ``T`` axis.
        """
        h = self.norm(x)
        # `nn.MultiheadAttention` with batch_first=True takes
        # [batch, seq, embed]; we set batch=N (one node per batch row),
        # seq=T (the time axis), embed=d_model.
        out, _ = self.attn(h, h, h, need_weights=False)
        return x + self.dropout(out)
