# Phase-Field Modeling of Fatigue Crack Growth Coupled with Elastoplasticity: An Open-Source Julia Implementation

Open-source phase-field framework for simulating fatigue crack growth in elastoplastic solids, implemented in Julia using the [Gridap](https://github.com/gridap/Gridap.jl) finite element library.

> **Paper**: *An open-source phase-field framework for elastoplastic fatigue crack growth with comparative hardening study*


## Features

- J2 plasticity with power-law isotropic hardening in effective stress space
- Amor volumetric--deviatoric energy decomposition
- AT-2 phase-field regularisation with threshold-based fatigue degradation
- Taylor--Quinney weighted plastic stored energy in the crack driving force
- Cycle-jump acceleration for high-cycle fatigue
- Five benchmark problems with complete parameter sets

## Repository Structure

```
ElastoPlasticFatiguePF.jl/
├── Project.toml               # Julia environment
├── LICENSE
├── src/                       # Reusable library modules
│   ├── ElastoPlasticFatiguePF.jl   # Module entry point
│   ├── materials.jl           # MaterialData, yield_stress, plastic_strain_energy
│   ├── plasticity.jl          # PlasticStateData, return_mapping!
│   ├── energy.jl              # Amor split, history variable
│   ├── phasefield.jl          # Phase-field solver, fatigue degradation
│   ├── solver.jl              # Picard-iteration displacement solver
│   ├── projection.jl          # L2 projection, QP field types
│   └── visualization.jl       # PNG contour plots, mesh display
├── examples/                  # Self-contained benchmark scripts
│   ├── SENT.jl                # Single-edge notched tension (baseline)
│   ├── SENT_fatigue_sweep.jl  # SENT with multiple αT values
│   ├── SENT_hardening_sweep.jl# SENT with multiple hardening exponents
│   ├── hole_plate.jl          # Perforated specimen (force & disp control)
│   ├── three_point_bending.jl # Three-point bending
│   ├── compact_tension.jl     # Compact tension (CT)
│   └── double_notch.jl        # Asymmetric double-notch tension (ADNT)
└── visualization/             # Post-processing Jupyter notebooks
    ├── SENT/
    ├── hole/
    ├── 3PB/
    ├── ADNT/
    └── CT/
```

## Quick Start

### 1. Install Julia

Download Julia 1.9+ from <https://julialang.org/downloads/>.

### 2. Install dependencies

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

Or install individually:

```julia
Pkg.add(["Gridap", "GridapGmsh", "Gmsh", "Plots", "WriteVTK", "DelimitedFiles"])
```

### 3. Run a benchmark

```bash
julia examples/SENT.jl
julia examples/compact_tension.jl
julia examples/hole_plate.jl
```

Each script creates an output directory with VTK files (for ParaView) and PNG contour snapshots.

## Benchmark Summary

| Example script | Specimen | Section | Key parameters |
|---|---|---|---|
| `SENT.jl` | Single-edge notched tension | 4.1.2 | Table 3 |
| `SENT_fatigue_sweep.jl` | SENT, αT sweep | 4.1.2 | αT ∈ {5, 8, 11.25, 15, 20, 30} |
| `SENT_hardening_sweep.jl` | SENT, n sweep | 4.1.2 | n ∈ {0, 0.05, 0.1, 0.2, 0.3} |
| `hole_plate.jl` | Perforated plate | 4.1.1 | Force & displacement control |
| `three_point_bending.jl` | Three-point bending | 4.2 | Table 13 |
| `compact_tension.jl` | Compact tension (CT) | -- | GH4169 @ 550 °C |
| `double_notch.jl` | ADNT | 4.3 | -- |

## Post-processing

VTK output can be visualised with [ParaView](https://www.paraview.org). The *Warp by Vector* filter applied to the displacement field `u` provides a convenient way to overlay deformed configurations on damage or plastic strain contours.

## Citation

This code is associated with a manuscript that has been submitted to *Theoretical and Applied Fracture Mechanics*:

>  Peng Zhang, Keke Tang, and Shan-tung Tu. "An open-source phase-field framework for elastoplastic fatigue crack growth with comparative hardening study." *Theoretical and Applied Fracture Mechanics* (submitted, 2025).

If you find this code helpful in your research, please stay tuned for the official publication. Once the paper is accepted and published, we would greatly appreciate it if you could cite our work accordingly.

## Contact

If you have any questions, suggestions, or find any issues with the code, please feel free to reach out. Any feedback or corrections are greatly appreciated.

- 📧 Email: peng05@tongji.edu.cn
- 💬 WeChat: 15856672654

## License

MIT License. See [LICENSE](LICENSE).
