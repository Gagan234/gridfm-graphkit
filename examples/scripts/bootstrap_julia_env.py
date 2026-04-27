"""Bootstrap the Julia environment for gridfm-datakit on a fresh Python env.

`gridfm-datakit` uses `juliacall` (which uses `juliapkg`) to provision a private
Julia install and project environment, but its packaged `juliapkg.json` does not
declare the actual Julia packages it later imports (PowerModels, Ipopt, JuMP,
JSON, Memento, InfrastructureModels, NLsolve). This script adds those packages
into the juliapkg-managed project so that subsequent calls to data-generation
APIs like `gridfm_datakit.generate.generate_power_flow_data` work without
hitting `Package PowerModels not found in current path`.

Run **once per fresh Python environment** that has gridfm-datakit installed:

    python examples/scripts/bootstrap_julia_env.py

After it finishes, the juliapkg env at
`<conda env>/julia_env/Project.toml` will have the packages registered and
precompiled, and any later `juliacall.Main.seval("using PowerModels")` will
succeed instantly.
"""

from juliacall import Main as jl

REQUIRED_JULIA_PACKAGES = [
    "PowerModels",
    "Ipopt",
    "JuMP",
    "JSON",
    "Memento",
    "InfrastructureModels",
    "NLsolve",
]


def main() -> None:
    print(f"Adding {len(REQUIRED_JULIA_PACKAGES)} Julia packages to the juliapkg env:")
    for pkg in REQUIRED_JULIA_PACKAGES:
        print(f"  - {pkg}")
    print()

    jl.seval("using Pkg")

    for pkg in REQUIRED_JULIA_PACKAGES:
        print(f"Pkg.add({pkg!r})")
        jl.Pkg.add(pkg)

    print("\nPkg.precompile() — this may take several minutes the first time.")
    jl.Pkg.precompile()

    print("\nVerifying installation by importing each package...")
    for pkg in REQUIRED_JULIA_PACKAGES:
        jl.seval(f"using {pkg}")
        print(f"  using {pkg} OK")

    print("\nJulia environment is ready for gridfm-datakit.")


if __name__ == "__main__":
    main()
