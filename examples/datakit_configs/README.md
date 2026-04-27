# gridfm-datakit configurations

These YAML files are inputs for [gridfm-datakit](https://github.com/gridfm/gridfm-datakit) to generate synthetic power-flow datasets — the **input** stage of the GridFM training pipeline. They are distinct from the configs in `../config/`, which are training configs consumed by `gridfm_graphkit train|finetune|evaluate|predict`.

## Files

| File | Network | Scenarios | Purpose |
|---|---|---:|---|
| `test_case118_50.yaml` | IEEE 118-bus | 50 | Smoke test for end-to-end data-gen pipeline (Julia + PowerModels.jl + the agg-load-profile generator). Fast — completes in a couple of minutes once Julia is set up. |

## How to use

**One-time prerequisite per fresh Python env** — install the required Julia packages (gridfm-datakit's `juliapkg.json` doesn't declare them, so we install them ourselves):

```bash
python examples/scripts/bootstrap_julia_env.py
```

That script downloads Julia (via `juliapkg`) and adds PowerModels, Ipopt, JuMP, JSON, Memento, InfrastructureModels, NLsolve into the project env, then precompiles them. Takes 5–10 minutes the first time.

After bootstrap is done, generate data with:

```python
from gridfm_datakit.generate import generate_power_flow_data
generate_power_flow_data("examples/datakit_configs/test_case118_50.yaml")
```

Subsequent calls reuse the cached Julia environment and start in seconds.

Output structure: `<settings.data_dir>/<network.name>/raw/` containing parquet files (`bus_data.parquet`, `branch_data.parquet`, etc.) — see the gridfm-datakit docs for the schema.

## Notes on the chosen parameters

- `network.source: pglib` → uses the PGLib-OPF case library (the standard benchmark distribution).
- `load.generator: agg_load_profile` → uses real ERCOT temporal load profiles, sampled in temporal order. **Topology is fixed across scenarios** (`topology_perturbation: none`), preserving temporal consistency required for the time-series modeling work in the thesis.
- `load.sigma: 0.15` → per-bus log-normal noise on top of the aggregate profile (15%).
- `settings.mode: pf` → Power Flow simulation (not OPF), which is faster and includes off-nominal scenarios.
- `settings.pf_fast: false` and `settings.dcpf_fast: false` → use the slower-but-correct PowerModels.jl path. Note that even with `pf_fast: false`, Julia + PowerModels.jl is still needed for finding the maximum scaling factor at the start of generation.
