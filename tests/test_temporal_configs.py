"""Tests for the temporal-reconstruction example YAML configs.

Two architectures share the masking framework and YAML schema:

- ``TemporalGNS_heterogeneous`` — per-time-step baseline (no cross-time
  interaction).
- ``FactorizedSpatioTemporalGNS_heterogeneous`` — factorized space-time
  attention (the thesis contribution).

Each architecture has six configs, one per masking strategy. These
tests verify the configs parse, register, and wire through the
component registries — actual training behavior is covered by
``test_factorized_st_gns.py`` (model unit tests) and the smoke
training runs.
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

_BASELINE_CONFIGS = [
    "Temporal_Reconstruction_random_point_case118.yaml",
    "Temporal_Reconstruction_block_temporal_case118.yaml",
    "Temporal_Reconstruction_causal_case118.yaml",
    "Temporal_Reconstruction_block_spatial_case118.yaml",
    "Temporal_Reconstruction_tube_case118.yaml",
    "Temporal_Reconstruction_topology_case118.yaml",
]

_FACTORIZED_CONFIGS = [
    "Factorized_ST_random_point_case118.yaml",
    "Factorized_ST_block_temporal_case118.yaml",
    "Factorized_ST_causal_case118.yaml",
    "Factorized_ST_block_spatial_case118.yaml",
    "Factorized_ST_tube_case118.yaml",
    "Factorized_ST_topology_case118.yaml",
]

_ALL_TEMPORAL_CONFIGS = _BASELINE_CONFIGS + _FACTORIZED_CONFIGS


def _load_args(filename: str) -> NestedNamespace:
    """Load a YAML config file from examples/config/ as a NestedNamespace."""
    with open(_CONFIG_DIR / filename) as f:
        cfg = yaml.safe_load(f)
    return NestedNamespace(**cfg)


@pytest.mark.parametrize("config_name", _ALL_TEMPORAL_CONFIGS)
def test_config_yaml_loads(config_name: str) -> None:
    """Every temporal config parses as YAML without error."""
    args = _load_args(config_name)
    assert hasattr(args, "task")
    assert args.task.task_name == "TemporalReconstruction"
    assert hasattr(args, "masking")
    assert hasattr(args.masking, "strategy")
    assert hasattr(args, "data")
    assert hasattr(args, "model")
    assert args.model.type in {
        "TemporalGNS_heterogeneous",
        "FactorizedSpatioTemporalGNS_heterogeneous",
    }
    assert hasattr(args, "training")


@pytest.mark.parametrize("config_name", _ALL_TEMPORAL_CONFIGS)
def test_config_compose_chain_instantiates(config_name: str) -> None:
    """The TemporalReconstruction Compose chain instantiates from each config.

    Exercises the masking strategy creation path end-to-end: registry
    lookup, kwargs filtering by strategy signature, the masking
    transform's own constructor, and the wrapping ApplyMasking transform.
    """
    args = _load_args(config_name)
    chain = TRANSFORM_REGISTRY.create("TemporalReconstruction", args)
    assert isinstance(chain, TemporalReconstructionTransforms)


@pytest.mark.parametrize("config_name", _ALL_TEMPORAL_CONFIGS)
def test_config_model_loads(config_name: str) -> None:
    """``load_model(args)`` returns the model class named in the config."""
    args = _load_args(config_name)
    model = load_model(args)
    assert type(model).__name__ == args.model.type


@pytest.mark.parametrize("config_name", _ALL_TEMPORAL_CONFIGS)
def test_config_loss_function_loads(config_name: str) -> None:
    """``get_loss_function(args)`` returns a callable for each config."""
    args = _load_args(config_name)
    loss_fn = get_loss_function(args)
    assert callable(loss_fn)


@pytest.mark.parametrize("config_name", _ALL_TEMPORAL_CONFIGS)
def test_config_task_is_in_registry(config_name: str) -> None:
    """The task name in each config is registered in TASK_REGISTRY."""
    args = _load_args(config_name)
    assert args.task.task_name in TASK_REGISTRY


def test_all_six_strategies_have_a_baseline_config() -> None:
    """Sanity: one baseline config per registered masking strategy."""
    expected_strategies = {
        "random_point",
        "block_temporal",
        "causal",
        "block_spatial",
        "tube",
        "topology",
    }
    found_strategies = set()
    for cfg_name in _BASELINE_CONFIGS:
        args = _load_args(cfg_name)
        found_strategies.add(args.masking.strategy)
    assert found_strategies == expected_strategies


def test_all_six_strategies_have_a_factorized_config() -> None:
    """Sanity: one factorized config per registered masking strategy.

    Required for the thesis ablation table — every masking strategy
    needs a paired baseline-vs-factorized comparison.
    """
    expected_strategies = {
        "random_point",
        "block_temporal",
        "causal",
        "block_spatial",
        "tube",
        "topology",
    }
    found_strategies = set()
    for cfg_name in _FACTORIZED_CONFIGS:
        args = _load_args(cfg_name)
        found_strategies.add(args.masking.strategy)
    assert found_strategies == expected_strategies


def test_baseline_and_factorized_configs_pair_up() -> None:
    """For every (strategy, network) baseline config there's a
    matching factorized config, so the ablation table has consistent
    paired rows."""
    baseline_strategies = {
        _load_args(c).masking.strategy: c for c in _BASELINE_CONFIGS
    }
    factorized_strategies = {
        _load_args(c).masking.strategy: c for c in _FACTORIZED_CONFIGS
    }
    assert set(baseline_strategies.keys()) == set(factorized_strategies.keys())
