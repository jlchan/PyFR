#!/usr/bin/env python3
"""One-off check that [solver-debug] grad-hook scales ∇U by 2 on the RHS path.

    python gradhook_scale_demo.py

Expect stderr lines like:
    [pyfr.gradhook] before gradu[0][0]=...
    [pyfr.gradhook] after  gradu[0][0]=...   (≈ 2× the before value)

Uses PyFR-Test-Cases/2d-couette-flow (repo submodule or ../PyFR-Test-Cases).
OpenMP + gcc-15 as in couette-flow.ini.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

from mpi4py import MPI

from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import init_mpi
from pyfr.readers.native import NativeReader
from pyfr.solvers import get_solver

GCC15_CC = '/opt/homebrew/bin/gcc-15'


def _couette_dir(root: Path) -> Path | None:
    for p in (root / 'PyFR-Test-Cases', root.parent / 'PyFR-Test-Cases'):
        tc = p / '2d-couette-flow'
        if (tc / 'couette-flow.ini').is_file():
            return tc
    return None


def main() -> int:
    if not MPI.Is_initialized():
        init_mpi()

    root = Path(__file__).resolve().parent
    tc = _couette_dir(root)
    if tc is None:
        print('Need PyFR-Test-Cases/2d-couette-flow', file=sys.stderr)
        return 1

    msh = tc / 'couette-flow-single-quad.msh'
    if not msh.is_file():
        print(f'Missing mesh {msh}', file=sys.stderr)
        return 1

    if not Path(GCC15_CC).is_file():
        print(f'Missing compiler {GCC15_CC}', file=sys.stderr)
        return 1

    with tempfile.TemporaryDirectory() as tmp:
        pyfrm = Path(tmp) / 'mesh.pyfrm'
        subprocess.run(
            [sys.executable, '-m', 'pyfr', 'import', str(msh), str(pyfrm)],
            check=True,
        )

        cfg = Inifile.load(str(tc / 'couette-flow.ini'))
        cfg.set('solver-debug', 'grad-hook', 'scale-two')
        cfg.set('backend-openmp', 'cc', GCC15_CC)

        backend = get_backend('openmp', cfg)
        mesh = NativeReader(pyfrm).mesh
        integrator = get_solver(backend, mesh, None, cfg)

        print('Running one RHS (grad-hook ×2 active)…', flush=True)
        integrator.system.rhs(0.0, 0, 0)

    print('Done — see stderr above for [pyfr.gradhook] before/after.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
