"""Per-time-step temporal wrapper around the static GNS_heterogeneous model.

This is the simplest spatio-temporal model in the framework: a static
``GNS_heterogeneous`` is run independently at each time step in the
``[N, T, F]`` input. There is no information sharing across the time
dimension at the model level, so the representational capacity equals
that of the static model — masking strategies that depend on temporal
patterns (block_temporal, causal) benefit from the *data* shape but not
from the model learning across time.

This wrapper is the "no temporal modeling" baseline mentioned in the
methodology. It exists so the full training pipeline can be exercised
end-to-end against a temporal dataset using the existing static
spatial architecture, and serves as the lower bound against which any
temporally-attentive model will be compared in the experiments chapter.
A subsequent design adds true temporal attention; the registration name
and forward signature are chosen to remain stable across that change.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from gridfm_graphkit.datasets.globals import PQ_H, PV_H, REF_H
from gridfm_graphkit.io.registries import (
    MODELS_REGISTRY,
    PHYSICS_DECODER_REGISTRY,
)
from gridfm_graphkit.models.gnn_heterogeneous_gns import GNS_heterogeneous
from gridfm_graphkit.models.utils import PhysicsDecoderPF


# `gnn_heterogeneous_gns.GNS_heterogeneous.__init__` calls
# `get_physics_decoder(args)`, which looks up `args.task.task_name` in
# `PHYSICS_DECODER_REGISTRY`. The temporal task uses the PowerFlow physics
# path per-time-step, so we register `PhysicsDecoderPF` under the
# `"TemporalReconstruction"` task name. Idempotent: the explicit guard
# avoids the registry's `KeyError: already registered` if this module is
# reloaded.
if "TemporalReconstruction" not in PHYSICS_DECODER_REGISTRY:
    PHYSICS_DECODER_REGISTRY.register("TemporalReconstruction")(PhysicsDecoderPF)


@MODELS_REGISTRY.register("TemporalGNS_heterogeneous")
class TemporalGNS_heterogeneous(nn.Module):
    """Per-time-step wrapper around the static GNS_heterogeneous model.

    Args:
        args: experiment configuration. Same shape as for ``GNS_heterogeneous``;
            the inner static model is constructed by passing ``args`` through
            unchanged. ``args.task.task_name`` is expected to be
            ``"TemporalReconstruction"`` (registered above for the physics
            decoder lookup).

    Forward signature mirrors ``GNS_heterogeneous`` so this class is a
    drop-in replacement when the dataset produces ``[N, T, F]`` samples
    rather than ``[N, F]`` ones.
    """

    def __init__(self, args) -> None:
        super().__init__()
        self.spatial_model = GNS_heterogeneous(args)

    @property
    def layer_residuals(self):
        """Forward access to the inner static model's per-layer residuals.

        ``LayeredWeightedPhysicsLoss`` reads ``model.layer_residuals``
        directly. For the per-time-step baseline this contains only the
        residuals from the *last* time step in the window — a known
        limitation noted for the experiments chapter. A future temporal
        model that aggregates across time would replace this property
        with an aggregated dict.
        """
        return self.spatial_model.layer_residuals

    def forward(
        self,
        x_dict: Dict[str, torch.Tensor],
        edge_index_dict: Dict,
        edge_attr_dict: Dict,
        mask_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        # All entity tensors share the same T at axis 1.
        T = x_dict["bus"].shape[1]

        # Output containers, one list per node-type-or-other-key the
        # spatial model returns.
        per_time_outputs: Dict[str, List[torch.Tensor]] = {}

        for t in range(T):
            x_slice = {
                nt: x_dict[nt][:, t, :] for nt in x_dict.keys()
            }
            edge_attr_slice = {
                et: edge_attr_dict[et][:, t, :]
                for et in edge_attr_dict.keys()
            }

            # The static model's mask_dict needs not just the temporal
            # masks (bus/gen/branch) but also the bus-type flags
            # (PQ/PV/REF) the inner physics decoder reads. Those flags
            # are derived from the static portion of the bus features
            # (which doesn't change with time within QSTS), so we
            # compute them from the current slice.
            bus_x_t = x_slice["bus"]
            mask_slice = {
                "bus": mask_dict["bus"][:, t, :],
                "gen": mask_dict["gen"][:, t, :],
                "branch": mask_dict["branch"][:, t, :],
                "PQ": bus_x_t[:, PQ_H] == 1,
                "PV": bus_x_t[:, PV_H] == 1,
                "REF": bus_x_t[:, REF_H] == 1,
            }

            out_slice = self.spatial_model(
                x_slice,
                edge_index_dict,
                edge_attr_slice,
                mask_slice,
            )

            # Lazy-init the output container with the keys we get back.
            if not per_time_outputs:
                per_time_outputs = {k: [] for k in out_slice.keys()}
            for k, v in out_slice.items():
                per_time_outputs[k].append(v)

        # Stack per-time-step outputs along the time dim (axis 1) so the
        # caller gets [N, T, F_out] tensors with the same convention as
        # the input.
        return {
            k: torch.stack(slices, dim=1)
            for k, slices in per_time_outputs.items()
        }
