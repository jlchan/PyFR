#!/usr/bin/env python3
"""Compare one viscous RHS: conservative vs entropy-gradient plumbing.

    python gradhook_scale_demo.py

Expect max |dU/dt_cons - dU/dt_ent| ~ machine epsilon when
con_to_ent and gradhook are exact inverses.
Uses PyFR-Test-Cases/2d-couette-flow (repo submodule or ../PyFR-Test-Cases).
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
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


def _rhs_out(cfg_path: Path, pyfrm: Path, gradvars: str) -> np.ndarray:
    cfg = Inifile.load(str(cfg_path))
    cfg.set('solver', 'gradient-variables', gradvars)
    if Path(GCC15_CC).is_file():
        cfg.set('backend-openmp', 'cc', GCC15_CC)

    backend = get_backend('openmp', cfg)
    mesh = NativeReader(pyfrm).mesh
    integrator = get_solver(backend, mesh, None, cfg)

    uin, fout = 0, 1
    integrator.system.rhs(0.0, uin, fout)

    return np.concatenate(integrator.system.ele_scal_upts(fout), axis=-1)


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

    ini = tc / 'couette-flow.ini'

    with tempfile.TemporaryDirectory() as tmp:
        pyfrm = Path(tmp) / 'mesh.pyfrm'
        subprocess.run(
            [sys.executable, '-m', 'pyfr', 'import', str(msh), str(pyfrm)],
            check=True,
        )

        print('Running one RHS (conservative gradients)…', flush=True)
        out_cons = _rhs_out(ini, pyfrm, 'conservative')

        print('Running one RHS (entropy-gradient plumbing)…', flush=True)
        out_ent = _rhs_out(ini, pyfrm, 'entropy')

    err = np.max(np.abs(out_cons - out_ent))
    print(f'max |dU/dt_cons - dU/dt_ent| = {err:.6e}', flush=True)
    if not np.allclose(out_cons, out_ent, rtol=0.0, atol=5e-9):
        print('FAIL: entropy path does not match conservative', file=sys.stderr)
        return 1

    print('PASS: entropy-gradient plumbing matches conservative.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
