"""Bootstrap the Julia environment for gridfm-datakit on a fresh Python env.

`gridfm-datakit` uses `juliacall` (which uses `juliapkg`) to provision a private
Julia install and project environment, but its packaged `juliapkg.json` does not
declare the actual Julia packages it later imports (PowerModels, Ipopt, JuMP,
JSON, Memento, InfrastructureModels, NLsolve). On a fresh Python env the first
call to `gridfm_datakit.generate.generate_power_flow_data` therefore fails with
"Package PowerModels not found in current path."

This script adds those packages by calling the `julia` binary directly via
`subprocess`, bypassing `juliacall`. It also sets `JULIA_CONDAPKG_BACKEND=Null`
so that `CondaPkg.jl` and its `MicroMamba_jll` artifact (which has been observed
to segfault during `Pkg.add` on Empire AI alpha) are not loaded — we manage our
Python environment with miniforge already and have no use for `CondaPkg.jl`.

Run **once per fresh Python environment**:

    python examples/scripts/bootstrap_julia_env.py

Subsequent calls into `gridfm-datakit`'s data-generation APIs will then find
the packages in the project env at `<conda env>/julia_env/Project.toml`.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REQUIRED_JULIA_PACKAGES = [
    "PowerModels",
    "Ipopt",
    "JuMP",
    "JSON",
    "Memento",
    "InfrastructureModels",
    "NLsolve",
]


def find_julia() -> Path:
    """Locate the julia binary installed by juliapkg into this conda env."""
    candidate = (
        Path(sys.prefix)
        / "julia_env"
        / "pyjuliapkg"
        / "install"
        / "bin"
        / "julia"
    )
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Could not find julia at {candidate}. Run "
        "`python -c 'from juliacall import Main'` once first to let "
        "juliapkg download Julia, then re-run this script."
    )


def run_julia(julia: Path, project: str, code: str) -> None:
    """Run a snippet of Julia via the binary, with the project env set."""
    env = os.environ.copy()
    env["JULIA_PROJECT"] = project
    env["JULIA_CONDAPKG_BACKEND"] = "Null"
    print(f"$ julia -e {code!r}")
    result = subprocess.run(
        [str(julia), "--startup-file=no", "-e", code],
        env=env,
        check=False,
    )
    if result.returncode != 0:
        print(
            f"julia exited with status {result.returncode}",
            file=sys.stderr,
        )
        sys.exit(result.returncode)


def main() -> None:
    julia = find_julia()
    project = str(Path(sys.prefix) / "julia_env")

    print(f"Julia binary: {julia}")
    print(f"Project env:  {project}")
    print(f"Packages:     {REQUIRED_JULIA_PACKAGES}")
    print()

    pkg_list = "[" + ", ".join(f'"{p}"' for p in REQUIRED_JULIA_PACKAGES) + "]"
    run_julia(julia, project, f"using Pkg; Pkg.add({pkg_list})")
    run_julia(julia, project, "using Pkg; Pkg.precompile()")

    print("\nVerifying imports...")
    for pkg in REQUIRED_JULIA_PACKAGES:
        run_julia(julia, project, f'using {pkg}; println("using {pkg} OK")')

    print("\nJulia environment is ready for gridfm-datakit.")


if __name__ == "__main__":
    main()
