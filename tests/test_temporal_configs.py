"""Tests for the six temporal-reconstruction example YAML configs.

Each config is loaded, wrapped as a NestedNamespace, and exercised
through the components it references (masking transform, model, loss).
This validates that the YAML schema is correct and that all six
strategies are wired through the registries end-to-end.

Note: this does NOT run a full training step against a real dataset.
Datamodule integration that produces [N, T, F] temporal samples from
the per-scenario base dataset is a separate change. These tests verify
the *schema and component wiring*; the per-component behavior is
covered by the per-strategy unit tests in tests/test_*_masking.py.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from gridfm_graphkit.datasets.task_transforms import (
    TemporalReconstructionTransforms,
)
from gridfm_graphkit.io.param_handler import (
    NestedNamespace,
    get_loss_function,
    load_model,
)
from gridfm_graphkit.io.registries import (
    MODELS_REGISTRY,
    TASK_REGISTRY,
    TRANSFORM_REGISTRY,
)


_CONFIG_DIR = Path(__file__).resolve().parents[1] / "examples" / "config"

_TEMPORAL_CONFIGS = [
    "Temporal_Reconstruction_random_point_case118.yaml",
    "Temporal_Reconstruction_block_temporal_case118.yaml",
    "Temporal_Reconstruction_causal_case118.yaml",
    "Temporal_Reconstruction_block_spatial_case118.yaml",
    "Temporal_Reconstruction_tube_case118.yaml",
    "Temporal_Reconstruction_topology_case118.yaml",
]


def _load_args(filename: str) -> NestedNamespace:
    """Load a YAML config file from examples/config/ as a NestedNamespace."""
    with open(_CONFIG_DIR / filename) as f:
        cfg = yaml.safe_load(f)
    return NestedNamespace(**cfg)


@pytest.mark.parametrize("config_name", _TEMPORAL_CONFIGS)
def test_config_yaml_loads(config_name: str) -> None:
    """Every temporal config parses as YAML without error."""
    args = _load_args(config_name)
    # Sanity: top-level structure matches the expected schema.
    assert hasattr(args, "task")
    assert args.task.task_name == "TemporalReconstruction"
    assert hasattr(args, "masking")
    assert hasattr(args.masking, "strategy")
    assert hasattr(args, "data")
    assert hasattr(args, "model")
    assert args.model.type == "TemporalGNS_heterogeneous"
    assert hasattr(args, "training")


@pytest.mark.parametrize("config_name", _TEMPORAL_CONFIGS)
def test_config_compose_chain_instantiates(config_name: str) -> None:
    """The TemporalReconstruction Compose chain instantiates from each config.

    Exercises the masking strategy creation path end-to-end: registry
    lookup, kwargs filtering by strategy signature, the masking
    transform's own constructor, and the wrapping ApplyMasking transform.
    """
    args = _load_args(config_name)
    chain = TRANSFORM_REGISTRY.create("TemporalReconstruction", args)
    assert isinstance(chain, TemporalReconstructionTransforms)


@pytest.mark.parametrize("config_name", _TEMPORAL_CONFIGS)
def test_config_model_loads(config_name: str) -> None:
    """`load_model(args)` returns a TemporalGNS_heterogeneous from each config."""
    args = _load_args(config_name)
    model = load_model(args)
    assert type(model).__name__ == "TemporalGNS_heterogeneous"


@pytest.mark.parametrize("config_name", _TEMPORAL_CONFIGS)
def test_config_loss_function_loads(config_name: str) -> None:
    """`get_loss_function(args)` returns a callable for each config."""
    args = _load_args(config_name)
    loss_fn = get_loss_function(args)
    assert callable(loss_fn)


@pytest.mark.parametrize("config_name", _TEMPORAL_CONFIGS)
def test_config_task_is_in_registry(config_name: str) -> None:
    """The task name in each config is registered in TASK_REGISTRY."""
    args = _load_args(config_name)
    assert args.task.task_name in TASK_REGISTRY


def test_all_six_strategies_have_a_config() -> None:
    """Sanity: one config per registered masking strategy."""
    expected_strategies = {
        "random_point",
        "block_temporal",
        "causal",
        "block_spatial",
        "tube",
        "topology",
    }
    found_strategies = set()
    for cfg_name in _TEMPORAL_CONFIGS:
        args = _load_args(cfg_name)
        found_strategies.add(args.masking.strategy)
    assert found_strategies == expected_strategies
