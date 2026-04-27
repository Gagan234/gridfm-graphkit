"""Masking strategies for spatio-temporal pretraining of GridFM.

Each ``MaskingStrategy`` subclass produces, for a single temporal sample,
boolean mask tensors with shape matching the corresponding feature tensors:

    m_bus      [N, T, F_bus]
    m_gen      [G, T, F_gen]
    m_branch   [E, T, F_edge]

``True`` means "masked / unknown / target of reconstruction loss." The
strategies differ only in the rule that places ``True``s; the integration
with the rest of the pipeline (the way the mask is consumed by ``ApplyMasking``
to zero out the input, by the model as side-channel input, and by the loss
to score only the masked positions) is identical across strategies.

Strategies are registered in ``MASKING_STRATEGY_REGISTRY``. New strategies
register themselves with the ``@MASKING_STRATEGY_REGISTRY.register("name")``
decorator and become selectable from a YAML config via ``masking.strategy: name``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Optional, Union

import torch

from gridfm_graphkit.io.registries import MASKING_STRATEGY_REGISTRY


def _broadcast_time_mask_to_entity_shapes(
    time_mask: torch.Tensor,
    x_bus: torch.Tensor,
    x_gen: torch.Tensor,
    x_edge: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Replicate a 1-D ``[T]`` time mask across every entity tensor.

    For node/edge feature tensors with shape ``[N, T, F]`` (or ``[G, T, F]``,
    ``[E, T, F]``), inserts singleton dims at axes 0 and 2 of the time mask
    and broadcasts it to match. Returns owned (cloned) tensors so downstream
    callers may write to them without aliasing the source.
    """
    if time_mask.dtype != torch.bool:
        time_mask = time_mask.bool()
    T = time_mask.shape[0]
    return {
        "bus": time_mask.view(1, T, 1).expand_as(x_bus).clone(),
        "gen": time_mask.view(1, T, 1).expand_as(x_gen).clone(),
        "branch": time_mask.view(1, T, 1).expand_as(x_edge).clone(),
    }


class MaskingStrategy(ABC):
    """Abstract base for spatio-temporal masking strategies.

    Subclasses implement ``build_masks`` to return a dict with boolean
    mask tensors for the three entity types (``bus``, ``gen``, ``branch``).
    The constructor receives only the strategy's own hyperparameters; the
    YAML→constructor wiring is handled at the registry-factory level.
    """

    @abstractmethod
    def build_masks(
        self,
        x_bus: torch.Tensor,
        x_gen: torch.Tensor,
        x_edge: torch.Tensor,
        edge_index: torch.Tensor,
        rng: torch.Generator,
    ) -> Dict[str, torch.Tensor]:
        """Build the mask dict for a temporal sample.

        Args:
            x_bus: bus features ``[N, T, F_bus]``.
            x_gen: generator features ``[G, T, F_gen]``.
            x_edge: branch features ``[E, T, F_edge]``.
            edge_index: bus-to-bus topology ``[2, E]``. Unused by strategies
                that don't need topology (e.g. random point), but accepted
                uniformly so the call signature is identical for all strategies.
            rng: a ``torch.Generator`` whose state is advanced during sampling.
                Pass a freshly-seeded generator for deterministic output;
                pass an unseeded ``torch.Generator()`` for nondeterministic.

        Returns:
            ``{"bus": m_bus, "gen": m_gen, "branch": m_branch}`` with each
            tensor a ``torch.bool`` of the same shape as its corresponding
            input. ``True`` indicates a masked position.
        """
        ...


@MASKING_STRATEGY_REGISTRY.register("random_point")
class RandomPointMasking(MaskingStrategy):
    """Independent Bernoulli mask per ``(node, time, feature)`` position.

    Each scalar element of each feature tensor is masked independently with
    the configured rate. By default all three entity types share a single
    rate; per-entity overrides are supported through ``entity_rates``.

    This strategy is the closest direct analogue of GraphMAE's node-feature
    masking on static graphs (Hou et al., KDD 2022) generalized over time.
    It is the most uninformed, "no structure" baseline; use it to establish
    how much the model can learn from a fully agnostic pretraining signal,
    against which the other (structurally aware) strategies are compared.

    Args:
        rate: global mask rate, in ``[0, 1]``. Default ``0.5``.
        entity_rates: optional per-entity overrides, e.g.
            ``{"bus": 0.5, "gen": 0.3, "branch": 0.0}``. Missing entities
            fall back to ``rate``.

    Raises:
        ValueError: if ``rate`` is outside ``[0, 1]``, or if any value in
            ``entity_rates`` is outside ``[0, 1]``.
    """

    def __init__(
        self,
        rate: float = 0.5,
        entity_rates: Optional[Dict[str, float]] = None,
    ) -> None:
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"rate must be in [0, 1], got {rate}")
        if entity_rates is not None:
            for ent, r in entity_rates.items():
                if not 0.0 <= r <= 1.0:
                    raise ValueError(
                        f"entity_rates[{ent!r}] must be in [0, 1], got {r}",
                    )
        self.rate = float(rate)
        self.entity_rates = dict(entity_rates) if entity_rates else {}

    def _rate_for(self, entity: str) -> float:
        return self.entity_rates.get(entity, self.rate)

    def build_masks(
        self,
        x_bus: torch.Tensor,
        x_gen: torch.Tensor,
        x_edge: torch.Tensor,
        edge_index: torch.Tensor,
        rng: torch.Generator,
    ) -> Dict[str, torch.Tensor]:
        # `torch.rand` produces values in [0, 1). Comparing with `< rate`
        # gives a Bernoulli sample where True = "masked".
        return {
            "bus": torch.rand(x_bus.shape, generator=rng) < self._rate_for("bus"),
            "gen": torch.rand(x_gen.shape, generator=rng) < self._rate_for("gen"),
            "branch": torch.rand(x_edge.shape, generator=rng) < self._rate_for("branch"),
        }


@MASKING_STRATEGY_REGISTRY.register("block_temporal")
class BlockTemporalMasking(MaskingStrategy):
    """Mask a contiguous block of length ``block_length`` along the time axis.

    Pretrains the model for *forecasting*: given the un-masked time steps
    as context, the model must reconstruct the values inside the masked
    block. This is the headline downstream task of the thesis.

    The mask is a single 1-D pattern over time that is broadcast across all
    nodes / edges and all features — every entity has the same block of
    time masked. (For per-entity time masks, compose with another strategy
    or extend this class.)

    Args:
        block_length: length L of the masked block. Must satisfy ``1 <= L <= T``
            where T is the sample's time dimension; checked at ``build_masks``.
        anchor: how the block's start position ``t0`` is chosen.
            - ``"random"``: ``t0`` is sampled uniformly from ``[0, T - L]``.
            - ``"trailing"``: ``t0 = T - L`` always. The block sits at the
              end of the window — this is the "predict the next L steps from
              the past" pattern used at evaluation time for forecasting.

    Raises:
        ValueError: if ``block_length < 1`` or ``anchor`` is unrecognized.
    """

    def __init__(self, block_length: int, anchor: str = "random") -> None:
        if block_length < 1:
            raise ValueError(f"block_length must be >= 1, got {block_length}")
        if anchor not in ("random", "trailing"):
            raise ValueError(
                f"anchor must be 'random' or 'trailing', got {anchor!r}",
            )
        self.block_length = int(block_length)
        self.anchor = anchor

    def build_masks(
        self,
        x_bus: torch.Tensor,
        x_gen: torch.Tensor,
        x_edge: torch.Tensor,
        edge_index: torch.Tensor,
        rng: torch.Generator,
    ) -> Dict[str, torch.Tensor]:
        T = x_bus.shape[1]
        if self.block_length > T:
            raise ValueError(
                f"block_length={self.block_length} exceeds sample T={T}",
            )

        if self.anchor == "trailing":
            t0 = T - self.block_length
        else:
            # `randint(low, high, ...)` is exclusive on `high`. With L<=T,
            # the maximum valid t0 is T-L, hence high=T-L+1.
            t0 = int(
                torch.randint(
                    0, T - self.block_length + 1, (1,), generator=rng,
                ).item(),
            )

        time_mask = torch.zeros(T, dtype=torch.bool)
        time_mask[t0 : t0 + self.block_length] = True

        return _broadcast_time_mask_to_entity_shapes(
            time_mask, x_bus, x_gen, x_edge,
        )


@MASKING_STRATEGY_REGISTRY.register("causal")
class CausalMasking(MaskingStrategy):
    """Mask every time step strictly after an anchor ``t*``.

    The mask satisfies the **causal monotonicity** invariant:
    ``m[..., t, ...]`` implies ``m[..., t', ...]`` for all ``t' > t``.
    Pretrains an autoregressive forecasting objective — at training time
    the model only sees positions ``t <= t*``, never the future.

    Args:
        anchor_position: where to place the boundary.
            - ``int``: deterministic anchor; positions ``t > anchor`` are masked.
              Must satisfy ``0 <= anchor < T`` at ``build_masks`` time.
            - ``"random"``: anchor sampled uniformly from ``[0, T)``.

    Raises:
        ValueError: on negative integer anchors or unrecognized strings.
    """

    def __init__(
        self,
        anchor_position: Union[int, str] = "random",
    ) -> None:
        if isinstance(anchor_position, str):
            if anchor_position != "random":
                raise ValueError(
                    "anchor_position string must be 'random', "
                    f"got {anchor_position!r}",
                )
        elif isinstance(anchor_position, int) and not isinstance(
            anchor_position, bool,
        ):
            if anchor_position < 0:
                raise ValueError(
                    f"anchor_position must be >= 0, got {anchor_position}",
                )
        else:
            raise ValueError(
                "anchor_position must be int or 'random', got "
                f"{type(anchor_position).__name__}",
            )
        self.anchor_position = anchor_position

    def build_masks(
        self,
        x_bus: torch.Tensor,
        x_gen: torch.Tensor,
        x_edge: torch.Tensor,
        edge_index: torch.Tensor,
        rng: torch.Generator,
    ) -> Dict[str, torch.Tensor]:
        T = x_bus.shape[1]

        if isinstance(self.anchor_position, str):
            t_star = int(torch.randint(0, T, (1,), generator=rng).item())
        else:
            t_star = int(self.anchor_position)
            if t_star >= T:
                raise ValueError(
                    f"anchor_position={t_star} >= sample T={T}",
                )

        time_mask = torch.zeros(T, dtype=torch.bool)
        time_mask[t_star + 1 :] = True

        return _broadcast_time_mask_to_entity_shapes(
            time_mask, x_bus, x_gen, x_edge,
        )
