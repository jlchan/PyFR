# PyFR Test Cases

Vendored copy of [PyFR/PyFR-Test-Cases](https://github.com/PyFR/PyFR-Test-Cases)
plus EC-artificial-viscosity cases. Meshes and solutions are mostly **not**
committed — see `.gitignore` and per-case READMEs for regeneration steps.

## Overview

Provided test cases include:

- 2D Couette flow (`2d-couette-flow/`) — includes `couette-flow-entropy.ini` (ECAV)
- 2D Gaussian pulse (`2d-gaussian-pulse/`) — `gaussian-pulse-entropy.ini` (ECAV)
- 2D Kelvin–Helmholtz (`2d-KHZ/`) — `kh.ini` (ECAV); see case README
- 2D Euler vortex (`2d-euler-vortex/`)
- 3D Triangular aerofoil (`3d-triangular-aerofoil/`)
- 3D Taylor-Green (`3d-taylor-green/`)
- 2D incompressible cylinder flow (`2d-inc-cylinder/`)
- 2D Double Mach Reflection (`2d-double-mach-reflection/`)
- 2D Viscous Shock Tube (`2d-viscous-shock-tube/`)

Large upstream meshes are stored as `*.msh.xz`; decompress before import:

```bash
xz -dk path/to/mesh.msh.xz
pyfr import path/to/mesh.msh mesh.pyfrm
```

Structured meshes with generators (not in git):

| Case | Regenerate |
|------|------------|
| `2d-KHZ/` | `python gen_kh_quad_msh.py` → `kh-coarse.msh`; `128 128 kh.msh` for fine |
| `2d-gaussian-pulse/` | `python gen_gaussian_pulse_quad_msh.py` |

Instructions for upstream cases are in the
[PyFR documentation](https://pyfr.readthedocs.io/en/latest/examples.html).

## License

The test cases are made available under a Creative Commons Attribution 4.0
license (see <http://creativecommons.org/licenses/by/4.0/>).
