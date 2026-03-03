<p align="center">
  <img src="https://raw.githubusercontent.com/gridfm/gridfm-graphkit/refs/heads/main/docs/figs/KIT.png" alt="GridFM logo" style="width: 40%; height: auto;"/>
  <br/>
</p>

<p align="center" style="font-size: 25px;">
    <b>gridfm-graphkit</b>
</p>


[![DOI](https://zenodo.org/badge/1007159095.svg)](https://doi.org/10.5281/zenodo.17016737)
[![Docs](https://img.shields.io/badge/docs-available-brightgreen)](https://gridfm.github.io/gridfm-graphkit/)
![Coverage](https://img.shields.io/badge/coverage-83%25-yellowgreen)
![Python](https://img.shields.io/badge/python-3.10%20%E2%80%93%203.12-blue)
![License](https://img.shields.io/badge/license-Apache%202.0-blue)

This library is brought to you by the GridFM team to train, finetune and interact with a foundation model for the electric power grid.

---

# Installation

Create and activate a virtual environment (make sure you use the right python version = 3.10, 3.11 or 3.12. I highly recommend 3.12)
```bash
python -m venv venv
source venv/bin/activate
```

Install gridfm-graphkit in editable mode
```bash
pip install -e .
```

Get PyTorch + CUDA version for torch-scatter
```bash
TORCH_CUDA_VERSION=$(python -c "import torch; print(torch.__version__ + ('+cpu' if torch.version.cuda is None else '+' + torch.version.cuda))")
```

Install the correct torch-scatter wheel
```bash
pip install torch-scatter -f https://data.pyg.org/whl/torch-${TORCH_CUDA_VERSION}.html
```


For documentation generation and unit testing, install with the optional `dev` and `test` extras:

```bash
pip install -e .[dev,test]
```


# CLI commands

Interface to train, fine-tune, evaluate, and run inference on GridFM models using YAML configs and MLflow tracking.

```bash
gridfm_graphkit <command> [OPTIONS]
```

Available commands:

* `train` - Train a new model from scratch
* `finetune` – Fine-tune an existing pre-trained model
* `evaluate` – Evaluate model performance on a dataset
* `predict` – Run inference and save predictions

---

## Training Models

```bash
gridfm_graphkit train --config path/to/config.yaml
```

### Arguments

| Argument         | Type   | Description                                                      | Default |
| ---------------- | ------ | ---------------------------------------------------------------- | ------- |
| `--config`       | `str`  | **Required**. Path to the training configuration YAML file.      | `None`       |
| `--exp_name`     | `str`  | MLflow experiment name.                                           | `timestamp`  |
| `--run_name`     | `str`  | MLflow run name.                                                  | `run`        |
| `--log_dir`      | `str`  | MLflow tracking/logging directory.                                | `mlruns`     |
| `--data_path`    | `str`  | Root dataset directory.                                           | `data`       |

### Examples

**Standard Training:**

```bash
gridfm_graphkit train --config examples/config/case30_ieee_base.yaml --data_path examples/data
```

---

## Fine-Tuning Models

```bash
gridfm_graphkit finetune --config path/to/config.yaml --model_path path/to/model.pt
```

### Arguments

| Argument       | Type  | Description                                     | Default   |
| -------------- | ----- | ----------------------------------------------- | --------- |
| `--config`     | `str` | **Required**. Fine-tuning configuration file.   | `None`    |
| `--model_path` | `str` | **Required**. Path to a pre-trained model state dict. | `None`    |
| `--exp_name`   | `str` | MLflow experiment name.                         | timestamp |
| `--run_name`   | `str` | MLflow run name.                                | `run`     |
| `--log_dir`    | `str` | MLflow logging directory.                       | `mlruns`  |
| `--data_path`  | `str` | Root dataset directory.                         | `data`    |


---

## Evaluating Models

```bash
gridfm_graphkit evaluate --config path/to/eval.yaml --model_path path/to/model.pt
```

### Arguments

| Argument              | Type  | Description                                                                                                   | Default   |
| --------------------- | ----- | ------------------------------------------------------------------------------------------------------------- | --------- |
| `--config`            | `str` | **Required**. Path to evaluation config.                                                                      | `None`    |
| `--model_path`        | `str` | Path to the trained model state dict.                                                                         | `None`    |
| `--normalizer_stats`  | `str` | Path to `normalizer_stats.pt` from a training run. Restores `fit_on_train` normalizers from saved statistics instead of re-fitting on the current data split. | `None`    |
| `--exp_name`          | `str` | MLflow experiment name.                                                                                       | timestamp |
| `--run_name`          | `str` | MLflow run name.                                                                                              | `run`     |
| `--log_dir`           | `str` | MLflow logging directory.                                                                                     | `mlruns`  |
| `--data_path`         | `str` | Dataset directory.                                                                                            | `data`    |
| `--compute_dc_ac_metrics` | `flag` | Compute ground-truth AC/DC power balance metrics on the test split.                                      | `False`   |
| `--save_output`       | `flag` | Save predictions as `<grid_name>_predictions.parquet` under MLflow artifacts (`.../artifacts/test`).       | `False`   |

### Example with saved normalizer stats

When evaluating a model on a dataset, you can pass the normalizer statistics from the original training run to ensure the same normalization parameters are used:

```bash
gridfm_graphkit evaluate \
  --config examples/config/HGNS_PF_datakit_case118.yaml \
  --model_path mlruns/<experiment_id>/<run_id>/artifacts/model/best_model_state_dict.pt \
  --normalizer_stats mlruns/<experiment_id>/<run_id>/artifacts/stats/normalizer_stats.pt \
  --data_path data
```

> **Note:** The `--normalizer_stats` flag only affects normalizers with `fit_strategy = "fit_on_train"` (e.g. `HeteroDataMVANormalizer`). Per-sample normalizers (`HeteroDataPerSampleMVANormalizer`) always recompute their statistics from the current dataset regardless of this flag.

---

## Running Predictions

```bash
gridfm_graphkit predict --config path/to/config.yaml --model_path path/to/model.pt
```

### Arguments

| Argument              | Type  | Description                                                                                                   | Default   |
| --------------------- | ----- | ------------------------------------------------------------------------------------------------------------- | --------- |
| `--config`            | `str` | **Required**. Path to prediction config file.                                                                 | `None`    |
| `--model_path`        | `str` | Path to the trained model state dict.                                                                         | `None`    |
| `--normalizer_stats`  | `str` | Path to `normalizer_stats.pt` from a training run. Restores `fit_on_train` normalizers from saved statistics. | `None`    |
| `--exp_name`          | `str` | MLflow experiment name.                                                                                       | timestamp |
| `--run_name`          | `str` | MLflow run name.                                                                                              | `run`     |
| `--log_dir`           | `str` | MLflow logging directory.                                                                                     | `mlruns`  |
| `--data_path`         | `str` | Dataset directory.                                                                                            | `data`    |
| `--output_path`       | `str` | Directory where predictions are saved as `<grid_name>_predictions.parquet`.                                  | `data`    |

Use built-in help for full command details:

```bash
gridfm_graphkit --help
gridfm_graphkit <command> --help
```

---
