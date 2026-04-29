"""Factorized spatio-temporal GNS for the QSTS masked-reconstruction task.

This is the principal architectural contribution of the thesis. It
extends :class:`GNS_heterogeneous` from a static (single-time-point)
spatial graph transformer into a factorized spatio-temporal model
that alternates **spatial** graph convolution with **temporal**
self-attention along the time axis. The factorization follows the
STD-MAE (Gao et al., IJCAI 2024) and GeoMAE (2025) lineage, adapted
for power grids with heterogeneous bus / generator nodes and a
physics-informed iterative-refinement loop preserved from the static
GridFM architecture.

Architecture (per layer ``i``, repeated ``num_layers`` times):

1. **Spatial step (per time step):** the same :class:`HeteroConv` of
   :class:`TransformerConv` blocks used by ``GNS_heterogeneous``, run
   independently on the bus/gen latents at each time step ``t``. Edge
   attributes are projected once at the start of forward and used for
   every time step (edges are inputs, not iteratively-refined node
   states). This step is *spatial only* — no information flows along
   the time axis.
2. **Temporal step:** :class:`TemporalAttentionLayer` is applied to
   the bus and generator latents along the time dimension. Each node
   attends across its own ``T`` time steps; nodes do not attend to
   each other in this layer (cross-node mixing is the spatial step's
   job). This step is the *temporal-only* factor; together with step
   1 it constitutes the factorized space-time attention.
3. **Decode + physics (per time step):** the per-time-step decode
   (``mlp_bus``, ``mlp_gen``), masking restoration, branch-flow /
   node-injection / physics-decoder, and residual computation are
   identical to the static ``GNS_heterogeneous`` per-layer logic.
   Layer residuals (mean L2 norm of bus power-balance residuals) are
   averaged across time and fed back into ``h_bus`` via
   ``physics_mlp`` so subsequent layers can refine the prediction.

This model is registered under ``"FactorizedSpatioTemporalGNS_heterogeneous"``
in ``MODELS_REGISTRY``. The physics decoder is registered under
``"TemporalReconstruction"`` (idempotent — same registration as
``TemporalGNS_heterogeneous`` so this module can be imported alongside
the baseline).

Comparison with the per-time-step baseline:

- ``TemporalGNS_heterogeneous`` (baseline): wraps the entire
  ``GNS_heterogeneous`` and runs it once per time step with no
  cross-time interaction. Same params as the static model.
- ``FactorizedSpatioTemporalGNS_heterogeneous`` (this): adds
  ``num_layers`` temporal attention blocks (~``num_layers * (4 *
  d_model^2 + 4 * d_model)`` params per attention head, plus
  layer-norm) interleaved with the spatial graph layers. Strictly
  contains the baseline's representational capacity (set the temporal
  blocks to identity and the model is the baseline) and adds
  cross-time information flow.
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from torch_geometric.nn import HeteroConv, TransformerConv

from gridfm_graphkit.datasets.globals import (
    BS,
    GS,
    MAX_PG,
    MAX_VM_H,
    MIN_PG,
    MIN_VM_H,
    PG_H,
    PG_OUT_GEN,
    PQ_H,
    PV_H,
    REF_H,
    VA_H,
    VM_H,
    VM_OUT,
)
from gridfm_graphkit.io.param_handler import get_physics_decoder
from gridfm_graphkit.io.registries import (
    MODELS_REGISTRY,
    PHYSICS_DECODER_REGISTRY,
)
from gridfm_graphkit.models.temporal_attention import (
    SinusoidalTemporalPositionalEncoding,
    TemporalAttentionLayer,
)
from gridfm_graphkit.models.utils import (
    ComputeBranchFlow,
    ComputeNodeInjection,
    ComputeNodeResiduals,
    bound_with_sigmoid,
)
from torch_scatter import scatter_add


# Same idempotent registration used by ``TemporalGNS_heterogeneous`` —
# both temporal models share the PowerFlow physics decoder under the
# ``TemporalReconstruction`` task name.
if "TemporalReconstruction" not in PHYSICS_DECODER_REGISTRY:
    from gridfm_graphkit.models.utils import PhysicsDecoderPF

    PHYSICS_DECODER_REGISTRY.register("TemporalReconstruction")(
        PhysicsDecoderPF,
    )


@MODELS_REGISTRY.register("FactorizedSpatioTemporalGNS_heterogeneous")
class FactorizedSpatioTemporalGNS_heterogeneous(nn.Module):
    """Factorized space-time GNS for QSTS masked reconstruction.

    Args:
        args: experiment configuration. Inherits the shape used by
            :class:`GNS_heterogeneous` plus the temporal-attention
            knobs:

            - ``args.model.num_layers``: number of (spatial + temporal)
              layers.
            - ``args.model.hidden_size``: latent dimension.
            - ``args.model.attention_head``: number of heads — used
              both by ``TransformerConv`` (spatial) and by
              :class:`TemporalAttentionLayer` (temporal).
            - ``args.model.input_bus_dim``, ``input_gen_dim``,
              ``edge_dim``, ``output_bus_dim``, ``output_gen_dim``:
              same meaning as for ``GNS_heterogeneous``.
            - ``args.model.dropout`` (default 0.0): applied in both the
              spatial and temporal layers.
            - ``args.task.task_name`` is expected to be
              ``"TemporalReconstruction"``.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.num_layers = int(args.model.num_layers)
        self.hidden_dim = int(args.model.hidden_size)
        self.input_bus_dim = int(args.model.input_bus_dim)
        self.input_gen_dim = int(args.model.input_gen_dim)
        self.output_bus_dim = int(args.model.output_bus_dim)
        self.output_gen_dim = int(args.model.output_gen_dim)
        self.edge_dim = int(args.model.edge_dim)
        self.heads = int(args.model.attention_head)
        self.task = args.task.task_name
        self.dropout = float(getattr(args.model, "dropout", 0.0))

        # ---- Embeddings ---------------------------------------------------

        self.input_proj_bus = nn.Sequential(
            nn.Linear(self.input_bus_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.input_proj_gen = nn.Sequential(
            nn.Linear(self.input_gen_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )
        self.input_proj_edge = nn.Sequential(
            nn.Linear(self.edge_dim, self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
        )

        # Positional encoding for the time axis. Applied to bus and
        # generator latents at the embedding stage so all subsequent
        # spatial / temporal layers see position-aware features.
        self.bus_pos_enc = SinusoidalTemporalPositionalEncoding(
            d_model=self.hidden_dim,
        )
        self.gen_pos_enc = SinusoidalTemporalPositionalEncoding(
            d_model=self.hidden_dim,
        )

        # Physics-mlp turns (P, Q) residuals back into latent-space updates.
        self.physics_mlp = nn.Sequential(
            nn.Linear(2, self.hidden_dim * self.heads),
            nn.LeakyReLU(),
        )

        # ---- Spatial + temporal layers -----------------------------------

        self.spatial_layers = nn.ModuleList()
        self.norms_bus = nn.ModuleList()
        self.norms_gen = nn.ModuleList()
        self.temporal_bus_layers = nn.ModuleList()
        self.temporal_gen_layers = nn.ModuleList()

        for i in range(self.num_layers):
            in_bus = (
                self.hidden_dim if i == 0 else self.hidden_dim * self.heads
            )
            in_gen = (
                self.hidden_dim if i == 0 else self.hidden_dim * self.heads
            )
            out_dim = self.hidden_dim

            conv_dict = {
                ("bus", "connects", "bus"): TransformerConv(
                    in_bus,
                    out_dim,
                    heads=self.heads,
                    edge_dim=self.hidden_dim,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("gen", "connected_to", "bus"): TransformerConv(
                    in_gen,
                    out_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=True,
                ),
                ("bus", "connected_to", "gen"): TransformerConv(
                    in_bus,
                    out_dim,
                    heads=self.heads,
                    dropout=self.dropout,
                    beta=True,
                ),
            }
            self.spatial_layers.append(HeteroConv(conv_dict, aggr="sum"))

            self.norms_bus.append(nn.LayerNorm(out_dim * self.heads))
            self.norms_gen.append(nn.LayerNorm(out_dim * self.heads))

            # Temporal attention dimension matches the post-spatial latent
            # width (hidden_dim * heads). The temporal layer uses
            # `self.heads` heads for symmetry with the spatial transformer
            # — both knobs share the same `attention_head` config field.
            self.temporal_bus_layers.append(
                TemporalAttentionLayer(
                    d_model=out_dim * self.heads,
                    n_heads=self.heads,
                    dropout=self.dropout,
                ),
            )
            self.temporal_gen_layers.append(
                TemporalAttentionLayer(
                    d_model=out_dim * self.heads,
                    n_heads=self.heads,
                    dropout=self.dropout,
                ),
            )

        # ---- Decode heads ------------------------------------------------

        self.mlp_bus = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_bus_dim),
        )
        self.mlp_gen = nn.Sequential(
            nn.Linear(self.hidden_dim * self.heads, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_dim, self.output_gen_dim),
        )

        self.activation = nn.LeakyReLU()
        self.branch_flow_layer = ComputeBranchFlow()
        self.node_injection_layer = ComputeNodeInjection()
        self.node_residuals_layer = ComputeNodeResiduals()
        self.physics_decoder = get_physics_decoder(args)

        # Mean per-layer residual norm, averaged across time. Same
        # reading interface as ``GNS_heterogeneous`` so
        # ``LayeredWeightedPhysicsLoss`` works without changes.
        self.layer_residuals: Dict[int, torch.Tensor] = {}

    # ----------------------------------------------------------------------
    # forward
    # ----------------------------------------------------------------------

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict,
        edge_attr_dict: Dict,
        mask_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Forward pass on a temporal sample.

        Args:
            x_dict: ``{"bus": [N, T, F_bus], "gen": [G, T, F_gen]}``.
            edge_index_dict: standard hetero edge index dict; topology
                is invariant across the window (QSTS assumption).
            edge_attr_dict: ``{("bus","connects","bus"): [E, T, F_edge],
                ...}``.
            mask_dict: ``{"bus": [N, T, F_bus], "gen": [G, T, F_gen],
                "branch": [E, T, F_edge]}`` boolean masks; ``True`` =
                masked (zeroed in input).

        Returns:
            ``{"bus": [N, T, output_bus_dim], "gen": [G, T,
            output_gen_dim]}``.
        """
        self.layer_residuals = {}

        T = x_dict["bus"].shape[1]
        num_bus = x_dict["bus"].size(0)
        bus_edge_index = edge_index_dict[("bus", "connects", "bus")]
        bus_edge_attr_full = edge_attr_dict[("bus", "connects", "bus")]
        _, gen_to_bus_index = edge_index_dict[("gen", "connected_to", "bus")]

        # ---- Embed inputs ------------------------------------------------

        h_bus = self.input_proj_bus(x_dict["bus"])  # [N, T, hidden]
        h_gen = self.input_proj_gen(x_dict["gen"])  # [G, T, hidden]

        # Position-aware: add sin/cos PE along time. The downstream
        # temporal attention layer is the consumer; the spatial layer
        # also sees PE-augmented features but treats them as plain
        # node features (it doesn't know about time).
        h_bus = self.bus_pos_enc(h_bus)
        h_gen = self.gen_pos_enc(h_gen)

        # Project edge attributes once; reused across time and layers.
        edge_attr_proj_dict: Dict = {}
        for key, edge_attr in edge_attr_dict.items():
            if edge_attr is not None:
                edge_attr_proj_dict[key] = self.input_proj_edge(edge_attr)
            else:
                edge_attr_proj_dict[key] = None

        # ---- Standing references for the per-time-step decode -----------

        # bus_mask, gen_mask, bus_fixed, gen_fixed have a time axis at
        # position 1; we slice along that axis at decode time.
        bus_mask = mask_dict["bus"][..., VM_H : VA_H + 1]  # [N, T, 2]
        gen_mask = mask_dict["gen"][..., : (PG_H + 1)]     # [G, T, 1]
        bus_fixed = x_dict["bus"][..., VM_H : VA_H + 1]    # [N, T, 2]
        gen_fixed = x_dict["gen"][..., : (PG_H + 1)]       # [G, T, 1]

        # ---- Layer loop --------------------------------------------------

        # Output containers: lists of per-time-step decoded outputs from
        # the FINAL layer (matches GNS_heterogeneous's "return last
        # layer's prediction").
        final_bus_outputs = None
        final_gen_outputs = None

        for i in range(self.num_layers):
            # 1) Spatial step: per-time-step HeteroConv.
            out_bus_per_t = []
            out_gen_per_t = []
            for t in range(T):
                edge_attr_t = {
                    k: (v[:, t, :] if v is not None else None)
                    for k, v in edge_attr_proj_dict.items()
                }
                conv_out = self.spatial_layers[i](
                    {"bus": h_bus[:, t, :], "gen": h_gen[:, t, :]},
                    edge_index_dict,
                    edge_attr_t,
                )
                out_bus_per_t.append(conv_out["bus"])
                out_gen_per_t.append(conv_out["gen"])

            out_bus = torch.stack(out_bus_per_t, dim=1)
            out_gen = torch.stack(out_gen_per_t, dim=1)
            out_bus = self.activation(self.norms_bus[i](out_bus))
            out_gen = self.activation(self.norms_gen[i](out_gen))

            # Skip connection from the layer's input.
            h_bus = h_bus + out_bus if out_bus.shape == h_bus.shape else out_bus
            h_gen = h_gen + out_gen if out_gen.shape == h_gen.shape else out_gen

            # 2) Temporal step: self-attention along T per node. This is
            #    the new computation relative to the per-time-step
            #    baseline.
            h_bus = self.temporal_bus_layers[i](h_bus)
            h_gen = self.temporal_gen_layers[i](h_gen)

            # 3) Decode + physics (per time step). Same structure as
            #    GNS_heterogeneous, but rolled over T explicitly.
            bus_out_per_t = []
            gen_out_per_t = []
            residuals_per_t = []

            for t in range(T):
                bus_temp = self.mlp_bus(h_bus[:, t, :])  # [N, output_bus_dim]
                gen_temp = self.mlp_gen(h_gen[:, t, :])  # [G, output_gen_dim]

                if self.task == "StateEstimation":
                    if i == self.num_layers - 1:
                        Pft, Qft = self.branch_flow_layer(
                            bus_temp,
                            bus_edge_index,
                            bus_edge_attr_full[:, t, :],
                        )
                        P_in, Q_in = self.node_injection_layer(
                            Pft, Qft, bus_edge_index, num_bus,
                        )
                        bus_out = self.physics_decoder(
                            P_in,
                            Q_in,
                            bus_temp,
                            x_dict["bus"][:, t, :],
                            None,
                            None,
                        )
                        bus_out_per_t.append(bus_out)
                        gen_out_per_t.append(gen_temp)
                else:
                    bus_temp = torch.where(
                        bus_mask[:, t, :], bus_temp, bus_fixed[:, t, :],
                    )
                    gen_temp = torch.where(
                        gen_mask[:, t, :], gen_temp, gen_fixed[:, t, :],
                    )

                    if self.task == "OptimalPowerFlow":
                        bus_temp[:, VM_OUT] = bound_with_sigmoid(
                            bus_temp[:, VM_OUT],
                            x_dict["bus"][:, t, MIN_VM_H],
                            x_dict["bus"][:, t, MAX_VM_H],
                        )
                        gen_temp[:, PG_OUT_GEN] = bound_with_sigmoid(
                            gen_temp[:, PG_OUT_GEN],
                            x_dict["gen"][:, t, MIN_PG],
                            x_dict["gen"][:, t, MAX_PG],
                        )

                    Pft, Qft = self.branch_flow_layer(
                        bus_temp,
                        bus_edge_index,
                        bus_edge_attr_full[:, t, :],
                    )
                    P_in, Q_in = self.node_injection_layer(
                        Pft, Qft, bus_edge_index, num_bus,
                    )
                    agg_bus = scatter_add(
                        gen_temp.squeeze(-1),
                        gen_to_bus_index,
                        dim=0,
                        dim_size=num_bus,
                    )
                    # The per-time-step physics_decoder needs a
                    # per-time mask_dict (slice of the temporal one).
                    # Bus-type flags PQ/PV/REF come from the static
                    # bus features, so they're invariant across T;
                    # but to keep the contract consistent with
                    # ``TemporalGNS_heterogeneous`` we recompute them
                    # from the current slice.
                    bus_x_t = x_dict["bus"][:, t, :]
                    mask_slice = {
                        "bus": mask_dict["bus"][:, t, :],
                        "gen": mask_dict["gen"][:, t, :],
                        "branch": mask_dict["branch"][:, t, :],
                        "PQ": bus_x_t[:, PQ_H] == 1,
                        "PV": bus_x_t[:, PV_H] == 1,
                        "REF": bus_x_t[:, REF_H] == 1,
                    }
                    bus_out = self.physics_decoder(
                        P_in,
                        Q_in,
                        bus_temp,
                        bus_x_t,
                        agg_bus,
                        mask_slice,
                    )
                    residual_P, residual_Q = self.node_residuals_layer(
                        P_in, Q_in, bus_out, bus_x_t,
                    )
                    bus_residuals = torch.stack(
                        [residual_P, residual_Q], dim=-1,
                    )
                    residuals_per_t.append(bus_residuals)

                    bus_out_per_t.append(bus_out)
                    gen_out_per_t.append(gen_temp)

            # Stack per-t outputs along the time axis.
            if bus_out_per_t:
                final_bus_outputs = torch.stack(bus_out_per_t, dim=1)
            if gen_out_per_t:
                final_gen_outputs = torch.stack(gen_out_per_t, dim=1)

            # 4) Residual feedback: average the per-time-step bus
            #    residuals across T, project back to latent space, add
            #    to h_bus. The temporal axis is preserved because
            #    `physics_mlp(bus_residuals_t)` is applied to each
            #    time slice individually, which is mathematically
            #    equivalent to a per-time-step residual update — the
            #    temporal attention layer in the next iteration then
            #    lets the model mix this updated state across time.
            if residuals_per_t:
                bus_residuals_T = torch.stack(residuals_per_t, dim=1)
                # Mean-norm across (N, T, 2) → scalar per layer for the
                # LayeredWeightedPhysicsLoss read-out.
                self.layer_residuals[i] = torch.linalg.norm(
                    bus_residuals_T, dim=-1,
                ).mean()
                # Per-time-step projection then add to h_bus.
                h_bus = h_bus + self.physics_mlp(bus_residuals_T)

        return {"bus": final_bus_outputs, "gen": final_gen_outputs}
