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
from typing import Dict, Optional

import torch

from gridfm_graphkit.io.registries import MASKING_STRATEGY_REGISTRY


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
