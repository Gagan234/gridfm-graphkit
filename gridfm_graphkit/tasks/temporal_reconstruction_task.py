"""Lightning task for spatio-temporal masked-reconstruction pretraining.

Thin subclass of :class:`ReconstructionTask` registered under
``"TemporalReconstruction"`` in ``TASK_REGISTRY``. The shared training,
validation, testing, and prediction logic from the base class works
without modification on temporal samples because:

- The model's forward signature ``(x_dict, edge_index_dict,
  edge_attr_dict, mask_dict)`` is unchanged; ``TemporalGNS_heterogeneous``
  produces ``[N, T, F_out]`` outputs from ``[N, T, F]`` inputs.
- Boolean indexing in the mask-aware loss functions
  (``MaskedMSE``, ``MaskedBusMSE``) flattens any-rank tensors equivalently,
  so a ``[N, T, F]`` tensor with a ``[N, T, F]`` bool mask reduces to the
  same loss expression as the static ``[N, F]`` case.
- Lightning's batch handling concatenates ``HeteroData`` along axis 0 of
  every entity tensor, which is the node/edge dim — leaving the
  temporal axis 1 intact.

The base ``ReconstructionTask`` constructor calls ``load_model`` and
``get_loss_function`` from ``param_handler``, both of which dispatch via
the existing registries — so no additional plumbing is needed in this
class.
"""

from __future__ import annotations

from gridfm_graphkit.io.registries import TASK_REGISTRY
from gridfm_graphkit.tasks.reconstruction_tasks import ReconstructionTask


@TASK_REGISTRY.register("TemporalReconstruction")
class TemporalReconstructionTask(ReconstructionTask):
    """Spatio-temporal masked-reconstruction Lightning task."""

    pass
