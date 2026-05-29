#!/usr/bin/env python3
"""Check entropy-gradient kernels and compare one viscous RHS.

    python entropy_grad_demo.py

The pointwise kernels are checked against NumPy reference formulas. The
conservative and entropy-gradient RHS paths are then compared diagnostically;
with nonlinear entropy variables they are not expected to match to roundoff.
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


def _relerr(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = max(np.max(np.abs(actual)), np.max(np.abs(expected)), 1e-300)
    return np.max(np.abs(actual - expected)) / scale


def _cons_state(rho: float, vel: tuple[float, ...], p: float,
                gamma: float) -> np.ndarray:
    vel = np.array(vel)
    rho_v = rho*vel
    rho_e = p/(gamma - 1) + 0.5*rho*np.dot(vel, vel)

    return np.r_[rho, rho_v, rho_e]


def _cons2entropy_ref(u: np.ndarray, gamma: float) -> np.ndarray:
    rho = u[0]
    vel = u[1:-1] / rho
    v2_mag = np.dot(vel, vel)
    p = (gamma - 1)*(u[-1] - 0.5*rho*v2_mag)
    s = np.log(p) - gamma*np.log(rho)
    rho_p = rho / p

    return np.r_[(gamma - s)/(gamma - 1) - 0.5*rho_p*v2_mag,
                 rho_p*vel, -rho_p]


def _jac_entropy2cons_apply_ref(u: np.ndarray, dw: np.ndarray,
                                gamma: float) -> np.ndarray:
    ndims = len(u) - 2
    rho = u[0]
    rho_v = u[1:-1]
    rho_e = u[-1]
    vel = rho_v / rho
    v2_mag = np.dot(vel, vel)
    p = (gamma - 1)*(rho_e - 0.5*rho*v2_mag)
    a2 = gamma*p / rho
    H = a2/(gamma - 1) + 0.5*v2_mag

    jac = np.empty((ndims + 2, ndims + 2))
    jac[0, :] = u
    jac[:, 0] = u
    jac[1:-1, 1:-1] = np.outer(rho_v, vel)
    jac[1:-1, 1:-1] += np.eye(ndims)*p
    jac[1:-1, -1] = rho_v*H
    jac[-1, 1:-1] = rho_v*H
    jac[-1, -1] = rho*H*H - a2*p/(gamma - 1)

    return dw @ jac.T


def _check_entropy_kernels(cfg: Inifile) -> None:
    gamma = cfg.getfloat('constants', 'gamma')
    backend = get_backend('openmp', cfg)
    kprefix = 'pyfr.solvers.navstokes.kernels'
    backend.pointwise.register(f'{kprefix}.con_to_ent')
    backend.pointwise.register(f'{kprefix}.ent_to_con_grad')

    for ndims, state, dw in [
        (2, _cons_state(1.7, (0.4, -0.2), 2.3, gamma),
         np.array([[0.11, -0.07, 0.05, 0.03],
                   [-0.02, 0.13, -0.17, 0.19]])),
        (3, _cons_state(1.3, (0.2, -0.35, 0.45), 1.9, gamma),
         np.array([[0.07, -0.03, 0.05, -0.11, 0.13],
                   [0.17, 0.02, -0.19, 0.23, -0.29],
                   [-0.31, 0.37, 0.41, -0.43, 0.47]])),
    ]:
        nvars = ndims + 2
        tplargs = {'ndims': ndims, 'nvars': nvars, 'c': {'gamma': gamma}}

        uin = backend.matrix((1, nvars, 1),
                             initval=state.reshape(1, nvars, 1))
        vout = backend.matrix((1, nvars, 1))
        backend.run_kernels([
            backend.kernel('con_to_ent', tplargs=tplargs, dims=[1, 1],
                           uin=uin, vout=vout)
        ])

        vref = _cons2entropy_ref(state, gamma)
        verr = _relerr(vout.get()[0, :, 0], vref)
        np.testing.assert_allclose(vout.get()[0, :, 0], vref, rtol=1e-12,
                                   atol=1e-12)

        gradu = backend.matrix((ndims, 1, nvars, 1),
                               initval=dw.reshape(ndims, 1, nvars, 1))
        backend.run_kernels([
            backend.kernel('ent_to_con_grad', tplargs=tplargs, dims=[1, 1],
                           uin=uin, gradu=gradu)
        ])

        gref = _jac_entropy2cons_apply_ref(state, dw, gamma)
        gerr = _relerr(gradu.get()[:, 0, :, 0], gref)
        np.testing.assert_allclose(gradu.get()[:, 0, :, 0], gref,
                                   rtol=1e-12, atol=1e-12)

        print(f'{ndims}D pointwise entropy kernel relative errors: '
              f'V(U)={verr:.6e}, J*dV={gerr:.6e}', flush=True)


def _couette_dir(root: Path) -> Path | None:
    for p in (root / 'PyFR-Test-Cases', root.parent / 'PyFR-Test-Cases'):
        tc = p / '2d-couette-flow'
        if (tc / 'couette-flow.ini').is_file():
            return tc
    return None


def _rhs_and_grads(cfg_path: Path, pyfrm: Path,
                   gradvars: str) -> tuple[np.ndarray, np.ndarray]:
    cfg = Inifile.load(str(cfg_path))
    cfg.set('solver', 'gradient-variables', gradvars)
    if Path(GCC15_CC).is_file():
        cfg.set('backend-openmp', 'cc', GCC15_CC)

    backend = get_backend('openmp', cfg)
    mesh = NativeReader(pyfrm).mesh
    integrator = get_solver(backend, mesh, None, cfg)

    uin, fout = 0, 1
    integrator.system.rhs(0.0, uin, fout)

    rhs = np.concatenate(integrator.system.ele_scal_upts(fout), axis=-1)
    grad = np.concatenate([g.get() for g in integrator.system.eles_vect_upts],
                          axis=-1)

    return rhs, grad


def _print_rel_diff(label: str, a: np.ndarray, b: np.ndarray) -> bool:
    rel_err = _relerr(a, b)
    print(f'{label} relative difference = {rel_err:.6e}', flush=True)

    return np.isfinite(rel_err)


def main() -> int:
    if not MPI.Is_initialized():
        init_mpi()

    root = Path(__file__).resolve().parent
    tc = _couette_dir(root)
    if tc is None:
        print('Need PyFR-Test-Cases/2d-couette-flow', file=sys.stderr)
        return 1

    msh = tc / 'couette-flow-quad.msh'
    if not msh.is_file():
        print(f'Missing mesh {msh}', file=sys.stderr)
        return 1

    ini = tc / 'couette-flow.ini'
    cfg = Inifile.load(str(ini))
    if Path(GCC15_CC).is_file():
        cfg.set('backend-openmp', 'cc', GCC15_CC)

    print('Checking pointwise entropy kernels…', flush=True)
    _check_entropy_kernels(cfg)

    with tempfile.TemporaryDirectory() as tmp:
        pyfrm = Path(tmp) / 'mesh.pyfrm'
        subprocess.run(
            [sys.executable, '-m', 'pyfr', 'import', str(msh), str(pyfrm)],
            check=True,
        )

        print('Running one RHS (conservative gradients)…', flush=True)
        out_cons, grad_cons = _rhs_and_grads(ini, pyfrm, 'conservative')

        print('Running one RHS (entropy-gradient plumbing)…', flush=True)
        out_ent, grad_ent = _rhs_and_grads(ini, pyfrm, 'entropy')

    if not _print_rel_diff('Conservative-gradient vs entropy-gradient field',
                           grad_cons, grad_ent):
        print('FAIL: non-finite gradient difference', file=sys.stderr)
        return 1
    if not _print_rel_diff('Conservative-gradient vs entropy-gradient RHS',
                           out_cons, out_ent):
        print('FAIL: non-finite RHS difference', file=sys.stderr)
        return 1

    print('PASS: entropy-gradient kernels match local references.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
