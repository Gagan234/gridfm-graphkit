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
from torch_geometric.utils import k_hop_subgraph

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


def _bus_mask_only(
    bus_mask_1d: torch.Tensor,
    x_bus: torch.Tensor,
    x_gen: torch.Tensor,
    x_edge: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Build a mask dict from a 1-D ``[N]`` bus mask — bus-only masking.

    The 1-D bus mask is broadcast across all time steps and all features of
    the bus tensor. The generator and branch tensors are returned with
    all-False masks (no propagation to connected entities at this layer —
    Phase 4 strategies are deliberately bus-scoped; gen/edge masking is a
    separate concern handled by future strategy compositions).
    """
    if bus_mask_1d.dtype != torch.bool:
        bus_mask_1d = bus_mask_1d.bool()
    N = bus_mask_1d.shape[0]
    return {
        "bus": bus_mask_1d.view(N, 1, 1).expand_as(x_bus).clone(),
        "gen": torch.zeros_like(x_gen, dtype=torch.bool),
        "branch": torch.zeros_like(x_edge, dtype=torch.bool),
    }


def _select_bus_subset(N: int, rate: float, rng: torch.Generator) -> torch.Tensor:
    """Pick ``floor(rate * N)`` distinct bus indices uniformly at random.

    Uses ``torch.randperm`` for sampling without replacement. Returns a
    ``[k]`` long tensor of bus indices.
    """
    n_masked = int(rate * N)
    if n_masked < 0:
        n_masked = 0
    if n_masked > N:
        n_masked = N
    perm = torch.randperm(N, generator=rng)
    return perm[:n_masked]


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


@MASKING_STRATEGY_REGISTRY.register("block_spatial")
class BlockSpatialMasking(MaskingStrategy):
    """Mask a *fresh* random subset of buses across all time steps.

    Each call to ``build_masks`` picks a new random subset
    ``S ⊂ [0, N)`` with ``|S| = floor(spatial_rate * N)`` using the
    caller-supplied RNG. The masked subset varies between samples in a
    batch — i.e., the model sees different "missing regions" each step.

    Pretrains the model for *state estimation under regional measurement
    loss*: a contiguous spatial subset of bus measurements is missing,
    and the model must infer those values from the rest of the grid.

    Args:
        spatial_rate: fraction of buses to mask, in ``[0, 1]``. The
            actual count is ``floor(spatial_rate * N)``, so very small
            rates with small ``N`` may round to zero buses.

    Raises:
        ValueError: on out-of-range ``spatial_rate``.
    """

    def __init__(self, spatial_rate: float = 0.3) -> None:
        if not 0.0 <= spatial_rate <= 1.0:
            raise ValueError(
                f"spatial_rate must be in [0, 1], got {spatial_rate}",
            )
        self.spatial_rate = float(spatial_rate)

    def build_masks(
        self,
        x_bus: torch.Tensor,
        x_gen: torch.Tensor,
        x_edge: torch.Tensor,
        edge_index: torch.Tensor,
        rng: torch.Generator,
    ) -> Dict[str, torch.Tensor]:
        N = x_bus.shape[0]
        masked_buses = _select_bus_subset(N, self.spatial_rate, rng)

        bus_mask_1d = torch.zeros(N, dtype=torch.bool)
        bus_mask_1d[masked_buses] = True

        return _bus_mask_only(bus_mask_1d, x_bus, x_gen, x_edge)


@MASKING_STRATEGY_REGISTRY.register("tube")
class TubeMasking(MaskingStrategy):
    """Mask the *same* random subset of buses on every call (persistent failure).

    Identical structure to :class:`BlockSpatialMasking`, but the masked
    subset ``S`` is fully determined by ``tube_seed`` and is therefore
    constant across all calls. This models a *sustained sensor failure*
    or a permanently-uninstrumented region: the same buses are absent in
    every training sample, so the model never gets to "see" them.

    The caller-supplied ``rng`` is **ignored** for spatial subset
    selection. (It is part of the signature for API uniformity across all
    strategies, and may be used by future extensions of ``TubeMasking``
    that add other random elements.)

    Args:
        tube_rate: fraction of buses included in the persistent missing
            region, in ``[0, 1]``.
        tube_seed: seed for the deterministic subset selection. Different
            seeds yield different (but each individually persistent)
            masked subsets — useful for ablation runs across multiple
            seeds.

    Raises:
        ValueError: on out-of-range ``tube_rate``.
    """

    def __init__(self, tube_rate: float = 0.3, tube_seed: int = 0) -> None:
        if not 0.0 <= tube_rate <= 1.0:
            raise ValueError(
                f"tube_rate must be in [0, 1], got {tube_rate}",
            )
        self.tube_rate = float(tube_rate)
        self.tube_seed = int(tube_seed)

    def build_masks(
        self,
        x_bus: torch.Tensor,
        x_gen: torch.Tensor,
        x_edge: torch.Tensor,
        edge_index: torch.Tensor,
        rng: torch.Generator,
    ) -> Dict[str, torch.Tensor]:
        N = x_bus.shape[0]

        # Use a fresh, fixed-seed generator independent of the caller's
        # `rng`. This is what makes the masked subset persistent across
        # calls regardless of how `rng` is seeded.
        tube_rng = torch.Generator().manual_seed(self.tube_seed)
        masked_buses = _select_bus_subset(N, self.tube_rate, tube_rng)

        bus_mask_1d = torch.zeros(N, dtype=torch.bool)
        bus_mask_1d[masked_buses] = True

        return _bus_mask_only(bus_mask_1d, x_bus, x_gen, x_edge)


@MASKING_STRATEGY_REGISTRY.register("topology")
class TopologyMasking(MaskingStrategy):
    """Mask the ``k``-hop graph neighborhood of an anchor bus.

    Picks an anchor bus, then runs breadth-first search on the bus-to-bus
    connectivity graph (provided as ``edge_index``) to find every bus
    within graph distance ``≤ hop_count`` of the anchor. The set of
    masked buses is therefore always a connected sub-graph (in the bus
    connectivity), which mirrors the way real grid failures propagate —
    a substation outage takes down all connected buses, a line
    contingency disconnects an entire downstream radial section, etc.

    This is the strategy with no direct precedent in the spatio-temporal
    MAE literature: prior work either masks at random or uses
    rectangular/tube regions in a regular grid (traffic, weather), but
    none consume the topological connectivity of an irregular graph the
    way a power network requires. Pretrains the model for downstream
    tasks where missingness is *coherent* rather than i.i.d. — primarily
    contingency analysis and security-state estimation.

    Args:
        hop_count: BFS radius ``k``. ``k = 0`` masks only the anchor
            itself; ``k = 1`` masks the anchor and its immediate
            neighbors; etc. Larger ``k`` → larger masked sub-graph.
        anchor_strategy: how to pick the anchor.
            - ``"random_bus"``: uniform over ``[0, N)``.
        anchor_bus: optional deterministic anchor index. If provided,
            overrides ``anchor_strategy``. Useful for unit tests and for
            reproducing specific contingency scenarios.

    Raises:
        ValueError: on negative ``hop_count``, unrecognized
            ``anchor_strategy``, or out-of-range explicit ``anchor_bus``.
    """

    def __init__(
        self,
        hop_count: int = 2,
        anchor_strategy: str = "random_bus",
        anchor_bus: Optional[int] = None,
    ) -> None:
        if hop_count < 0:
            raise ValueError(f"hop_count must be >= 0, got {hop_count}")
        if anchor_strategy not in ("random_bus",):
            raise ValueError(
                f"anchor_strategy must be 'random_bus', got {anchor_strategy!r}",
            )
        self.hop_count = int(hop_count)
        self.anchor_strategy = anchor_strategy
        self.anchor_bus = anchor_bus

    def build_masks(
        self,
        x_bus: torch.Tensor,
        x_gen: torch.Tensor,
        x_edge: torch.Tensor,
        edge_index: torch.Tensor,
        rng: torch.Generator,
    ) -> Dict[str, torch.Tensor]:
        N = x_bus.shape[0]

        if self.anchor_bus is not None:
            anchor = int(self.anchor_bus)
            if anchor < 0 or anchor >= N:
                raise ValueError(
                    f"anchor_bus={anchor} out of range [0, {N})",
                )
        else:
            anchor = int(torch.randint(0, N, (1,), generator=rng).item())

        # `k_hop_subgraph` returns (node_subset, ...). The subset includes
        # the anchor itself when k >= 0.
        node_subset, _, _, _ = k_hop_subgraph(
            [anchor],
            num_hops=self.hop_count,
            edge_index=edge_index,
            num_nodes=N,
            relabel_nodes=False,
        )

        bus_mask_1d = torch.zeros(N, dtype=torch.bool)
        bus_mask_1d[node_subset] = True

        return _bus_mask_only(bus_mask_1d, x_bus, x_gen, x_edge)
