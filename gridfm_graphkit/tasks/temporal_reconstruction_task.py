"""Lightning task for spatio-temporal masked-reconstruction pretraining.

Concrete subclass of :class:`ReconstructionTask` registered under
``"TemporalReconstruction"`` in ``TASK_REGISTRY``.

The base ``ReconstructionTask`` provides ``forward``, ``training_step``,
and ``validation_step`` (all of which work without modification on
temporal samples — boolean indexing and the model's forward signature
are rank-agnostic). It leaves ``test_step`` and ``predict_step``
abstract, so this subclass provides minimal concrete implementations.

The per-step bodies here are kept simple — compute the masked-
reconstruction loss via ``shared_step`` and log/return it — rather
than reproducing ``PowerFlowTask``'s PF-specific residual analysis
(branch flow, node injection, PBE metrics). That analysis assumes
rank-2 tensors with specific column indexing and does not translate
cleanly to the ``[N, T, F]`` temporal output shape produced by the
temporal model. A follow-up temporally-attentive model that produces
analyzable physics outputs across time may add richer reporting.
"""

from __future__ import annotations

from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.tasks.reconstruction_tasks import ReconstructionTask


@TASK_REGISTRY.register("TemporalReconstruction")
class TemporalReconstructionTask(ReconstructionTask):
    """Spatio-temporal masked-reconstruction Lightning task."""

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        _, loss_dict = self.shared_step(batch)
        loss_dict["loss"] = loss_dict["loss"].detach()
        for metric, value in loss_dict.items():
            self.log(
                f"Test {metric}",
                value,
                batch_size=batch.num_graphs,
                sync_dist=True,
                on_epoch=True,
                logger=True,
                on_step=False,
            )
        return loss_dict["loss"]

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        output, _ = self.shared_step(batch)
        return output
