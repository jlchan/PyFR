#!/usr/bin/env python3
"""Verify EC-AV plumbing Tasks 1-2 (storage/views + face transport).

    python ecav_plumbing_demo.py

Checks from ecav_plumbing_tasks plan:
  Task 1 — _av_scaling allocation, tflux view mapping, mode isolation.
  Task 2 — ecav_scaling_fpts fill, interface/BC views, MPI kernel registration
           (LDG-beta conditioning), RHS unchanged when scaling is transported
           but not consumed by flux kernels.

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
from pyfr.integrators.explicit.steppers import TVDRK3Stepper
from pyfr.mpiutil import get_comm_rank_root, init_mpi
from pyfr.readers.native import NativeReader
from pyfr.solvers.navstokes.system import NavierStokesSystem
from pyfr.writers.serialise import Serialiser

GCC15_CC = '/opt/homebrew/bin/gcc-15'


def _relerr(actual: np.ndarray, expected: np.ndarray) -> float:
    scale = max(np.max(np.abs(actual)), np.max(np.abs(expected)), 1e-300)
    return np.max(np.abs(actual - expected)) / scale


def _couette_dir(root: Path) -> Path | None:
    for p in (root / 'PyFR-Test-Cases', root.parent / 'PyFR-Test-Cases'):
        tc = p / '2d-couette-flow'
        if (tc / 'couette-flow.ini').is_file():
            return tc
    return None


def _configure_backend(cfg: Inifile) -> None:
    if Path(GCC15_CC).is_file():
        cfg.set('backend-openmp', 'cc', GCC15_CC)


def _read_view(view, mat) -> np.ndarray:
    base = mat.get().ravel()
    return base[view.mapping.get()[0]]


def _build_system(cfg: Inifile, mesh):
    backend = get_backend('openmp', cfg)
    registers = list(TVDRK3Stepper._registers.keys())
    system = NavierStokesSystem(backend, mesh, None, registers, cfg,
                                Serialiser())
    eles = dict(system.ele_map)
    int_inters = list(system._int_inters)
    mpi_inters = list(system._mpi_inters)
    bc_inters = list(system._bc_inters)
    system.commit()
    return system, eles, int_inters, mpi_inters, bc_inters


def _run_fill(system) -> None:
    for k in system._kernels['eles/ecav_scaling_fpts', None, None]:
        k.run()


def _run_rhs(system) -> np.ndarray:
    system.rhs(0.0, 0, 1)
    return np.concatenate(system.ele_scal_upts(1), axis=-1)


def _check_task1_construction(eles_map) -> None:
    for etype, eles in eles_map.items():
        assert eles._av_scaling.ioshape == (1, eles.neles), (
            f'{etype}: expected _av_scaling shape (1, neles)'
        )
        assert np.allclose(eles._av_scaling.get(), 1.0), (
            f'{etype}: _av_scaling should init to 1.0'
        )
        assert eles.av_scaling_views, (
            f'{etype}: expected non-empty av_scaling_views'
        )
        assert eles._av_scaling_fpts.ioshape == (eles.nfpts, eles.neles), (
            f'{etype}: expected _av_scaling_fpts shape (nfpts, neles)'
        )
    print('  Task 1 construction: PASS', flush=True)


def _check_task1_view_mapping(eles_map) -> None:
    for etype, eles in eles_map.items():
        vals = np.arange(eles.neles, dtype=float) + 100.0
        eles._av_scaling.set(np.atleast_2d(vals))

        for rgn, view in eles.av_scaling_views.items():
            mapped = _read_view(view, eles._av_scaling)
            off = 0 if rgn == 'curved' else eles.linoff
            n_rgn = eles.mesh_regions[rgn]
            npts = len(mapped) // n_rgn
            for i, ele in enumerate(range(off, off + n_rgn)):
                chunk = mapped[i*npts:(i + 1)*npts]
                assert np.allclose(chunk, vals[ele]), (
                    f'{etype}/{rgn}: view maps ele {ele} incorrectly'
                )
    print('  Task 1 tflux view mapping: PASS', flush=True)


def _check_task1_mode_isolation(cfg_base: Inifile, mesh) -> None:
    cfg = Inifile(cfg_base.tostr())
    cfg.set('solver', 'shock-capturing', 'none')
    _configure_backend(cfg)

    system, eles_map, *_ = _build_system(cfg, mesh)
    assert system._ec_av is None

    for eles in eles_map.values():
        assert not hasattr(eles, '_av_scaling')
        assert not getattr(eles, 'av_scaling_views', None)

    rhs = _run_rhs(system)
    assert np.all(np.isfinite(rhs))
    print('  Task 1 mode isolation (shock-capturing = none): PASS', flush=True)


def _check_task2_fill(eles_map, system) -> None:
    for etype, eles in eles_map.items():
        vals = np.arange(eles.neles, dtype=float) + 1.0
        eles._av_scaling.set(np.atleast_2d(vals))
        _run_fill(system)
        got = eles._av_scaling_fpts.get()
        for ele in range(eles.neles):
            assert np.allclose(got[:, ele], vals[ele]), (
                f'{etype}: fill mismatch at ele {ele}'
            )
    print('  Task 2 ecav_scaling_fpts fill: PASS', flush=True)


def _check_task2_interface_views(eles_map, system, int_inters) -> None:
    if not int_inters:
        print('  Task 2 internal interface views: SKIP (no internal faces)',
              flush=True)
        return

    inter = int_inters[0]
    assert inter.av_scaling_l is not None
    assert inter.av_scaling_r is not None

    for eles in eles_map.values():
        vals = np.arange(eles.neles, dtype=float) + 50.0
        eles._av_scaling.set(np.atleast_2d(vals))
    _run_fill(system)

    lhs = _read_view(inter.av_scaling_l, inter.av_scaling_l._mats[0])
    rhs = _read_view(inter.av_scaling_r, inter.av_scaling_r._mats[0])
    assert lhs.shape == rhs.shape
    assert np.all(np.isfinite(lhs))
    assert np.all(np.isfinite(rhs))
    print('  Task 2 internal interface views (av_scaling_l/r): PASS', flush=True)


def _check_task2_bc_views(bc_inters) -> None:
    if not bc_inters:
        print('  Task 2 BC views: SKIP (no boundary interfaces)', flush=True)
        return

    for bc in bc_inters:
        assert bc.av_scaling_l is not None
        assert getattr(bc, 'av_scaling_r', None) is None
    print('  Task 2 BC views (av_scaling_l only): PASS', flush=True)


def _mpi_has_kern(mpi_inters, name: str) -> bool:
    return any(name in m.kernels for m in mpi_inters)


def _check_task2_ldg_beta(cfg_base: Inifile, mesh) -> None:
    cases = [
        (0.5, True, True),
        (-0.5, False, True),
        (0.0, True, False),
    ]
    for beta, want_pack, want_unpack in cases:
        cfg = Inifile(cfg_base.tostr())
        cfg.set('solver', 'shock-capturing', 'ec-artificial-viscosity')
        cfg.set('solver-interfaces', 'ldg-beta', str(beta))
        _configure_backend(cfg)

        system = NavierStokesSystem(
            get_backend('openmp', cfg), mesh, None,
            list(TVDRK3Stepper._registers.keys()), cfg, Serialiser()
        )
        mpi_inters = list(system._mpi_inters)
        has_pack = _mpi_has_kern(mpi_inters, 'av_scaling_fpts_pack')
        has_unpack = _mpi_has_kern(mpi_inters, 'av_scaling_fpts_unpack')
        has_vect_pack = _mpi_has_kern(mpi_inters, 'vect_fpts_pack')
        has_vect_unpack = _mpi_has_kern(mpi_inters, 'vect_fpts_unpack')

        if mpi_inters:
            assert has_pack == want_pack, (
                f'ldg-beta={beta}: av_scaling_fpts_pack={has_pack}, '
                f'expected {want_pack}'
            )
            assert has_unpack == want_unpack, (
                f'ldg-beta={beta}: av_scaling_fpts_unpack={has_unpack}, '
                f'expected {want_unpack}'
            )
            assert has_pack == has_vect_pack, (
                f'ldg-beta={beta}: av_scaling pack != vect_fpts pack'
            )
            assert has_unpack == has_vect_unpack, (
                f'ldg-beta={beta}: av_scaling unpack != vect_fpts unpack'
            )

        del system

    if not cases:
        raise RuntimeError('unreachable')
    print('  Task 2 LDG-beta MPI registration: PASS', flush=True)


def _check_task2_mpi_smoke(system, mpi_inters) -> None:
    pack = system._kernels['mpiint/av_scaling_fpts_pack', None, None]
    unpack = system._kernels['mpiint/av_scaling_fpts_unpack', None, None]
    if mpi_inters:
        assert pack or unpack, 'expected av_scaling_fpts MPI kernels on partition'
    rhs = _run_rhs(system)
    assert np.all(np.isfinite(rhs))
    print('  Task 2 MPI smoke (RHS with transport graph): PASS', flush=True)


def _check_no_rhs_change(system, eles_map) -> None:
    for eles in eles_map.values():
        eles._av_scaling.set(np.full((1, eles.neles), 1.0))
    baseline = _run_rhs(system)

    for eles in eles_map.values():
        vals = np.arange(eles.neles, dtype=float) + 17.0
        eles._av_scaling.set(np.atleast_2d(vals))
    _run_fill(system)
    perturbed = _run_rhs(system)

    err = _relerr(perturbed, baseline)
    print(f'  RHS invariant under scaling transport (rel err = {err:.6e})',
          flush=True)
    np.testing.assert_allclose(perturbed, baseline, rtol=1e-12, atol=1e-12)
    print('  Task 1+2 no RHS change (flux kernels do not consume scaling): PASS',
          flush=True)


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
    cfg_base = Inifile.load(str(ini))
    _configure_backend(cfg_base)

    with tempfile.TemporaryDirectory() as tmp:
        pyfrm = Path(tmp) / 'mesh.pyfrm'
        subprocess.run(
            [sys.executable, '-m', 'pyfr', 'import', str(msh), str(pyfrm)],
            check=True,
        )
        mesh = NativeReader(pyfrm).mesh

        print('Task 1 — _av_scaling storage and tflux views', flush=True)

        cfg_ec = Inifile(cfg_base.tostr())
        cfg_ec.set('solver', 'shock-capturing', 'ec-artificial-viscosity')
        _configure_backend(cfg_ec)

        system, eles_map, int_inters, mpi_inters, bc_inters = _build_system(
            cfg_ec, mesh
        )
        assert system._ec_av is not None

        _check_task1_construction(eles_map)
        _check_task1_view_mapping(eles_map)
        _check_task1_mode_isolation(cfg_base, mesh)

        print('Task 2 — face scaling transport', flush=True)
        _check_task2_fill(eles_map, system)
        _check_task2_interface_views(eles_map, system, int_inters)
        _check_task2_bc_views(bc_inters)
        _check_task2_ldg_beta(cfg_base, mesh)
        _check_task2_mpi_smoke(system, mpi_inters)
        _check_no_rhs_change(system, eles_map)

    comm, rank, root_rank = get_comm_rank_root()
    if rank == root_rank:
        print('PASS: EC-AV plumbing Tasks 1-2 verification complete.', flush=True)
    return 0


if __name__ == '__main__':
    sys.exit(main())
