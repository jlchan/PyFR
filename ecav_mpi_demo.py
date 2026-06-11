#!/usr/bin/env python3
"""Prepare partitioned meshes and verify EC-AV MPI face-scaling transport.

Create 2-rank partitionings embedded in dedicated .pyfrm files:

    python ecav_mpi_demo.py --prepare

Run MPI checks (requires exactly 2 ranks):

    mpirun -n 2 python ecav_mpi_demo.py

Verifies av_scaling_fpts pack/send/recv/unpack on partition faces and that
remote face scalars match the neighbour rank's element values. Also smoke-tests
one RHS per case with shock-capturing = ec-artificial-viscosity.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
from mpi4py import MPI

from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.integrators.explicit.steppers import TVDRK3Stepper
from pyfr.mpiutil import get_comm_rank_root, init_mpi
from pyfr.partitioners import get_partitioner, write_partitioning
from pyfr.partitioners.base import BasePartitioner
from pyfr.progress import NullProgressSequence
from pyfr.readers.native import NativeReader
from pyfr.solvers.navstokes.system import NavierStokesSystem
from pyfr.util import subclasses
from pyfr.writers.serialise import Serialiser

from ecav_plumbing_demo import (
    GCC15_CC,
    _configure_backend,
    _read_view,
    _run_fill,
    _run_rhs,
)

NPARTS = 2
PART_NAME = '2'

CASES = (
    {
        'label': 'couette-flow',
        'dir': '2d-couette-flow',
        'msh': 'couette-flow-quad.msh',
        'pyfrm': 'couette-flow-quad.pyfrm',
        'out_pyfrm': 'couette-flow-quad-2p.pyfrm',
        'ini': 'couette-flow.ini',
    },
    {
        'label': 'gaussian-pulse',
        'dir': '2d-gaussian-pulse',
        'msh': 'gaussian-pulse-quad.msh',
        'pyfrm': 'gaussian-pulse-quad.pyfrm',
        'out_pyfrm': 'gaussian-pulse-quad-2p.pyfrm',
        'ini': 'gaussian-pulse-entropy.ini',
    },
)


def _test_cases_root(root: Path) -> Path | None:
    for p in (root / 'PyFR-Test-Cases', root.parent / 'PyFR-Test-Cases'):
        if p.is_dir():
            return p
    return None


def _resolve_case(tc_root: Path, spec: dict) -> dict:
    cdir = tc_root / spec['dir']
    return {
        **spec,
        'cdir': cdir,
        'msh_path': cdir / spec['msh'],
        'pyfrm_path': cdir / spec['pyfrm'],
        'out_path': cdir / spec['out_pyfrm'],
        'ini_path': cdir / spec['ini'],
    }


def _ensure_pyfrm(case: dict) -> None:
    if case['pyfrm_path'].is_file():
        return
    if not case['msh_path'].is_file():
        raise FileNotFoundError(f"Missing mesh {case['msh_path']}")

    subprocess.run(
        [sys.executable, '-m', 'pyfr', 'import',
         str(case['msh_path']), str(case['pyfrm_path'])],
        check=True,
    )


def _get_partitioner(nparts: int = NPARTS):
    pwts = [1] * nparts
    errors = []
    for name in sorted(cls.name for cls in subclasses(BasePartitioner)):
        try:
            return get_partitioner(name, pwts)
        except OSError as exc:
            errors.append(f'{name}: {exc}')
    raise RuntimeError(
        'No partitioners available; install METIS, Scotch, or KaHIP '
        f'({"; ".join(errors)})'
    )


def _write_partitioned_pyfrm(src: Path, dst: Path, *, force: bool = False) -> None:
    shutil.copy2(src, dst)
    with h5py.File(dst, 'r+') as mesh:
        if PART_NAME in mesh.get('partitionings', {}):
            if not force:
                return
            del mesh[f'partitionings/{PART_NAME}']

        part = _get_partitioner(NPARTS)
        pinfo = part.partition(mesh, NullProgressSequence())
        write_partitioning(mesh, PART_NAME, pinfo)


def prepare_partitioned_meshes(root: Path, *, force: bool = False) -> None:
    tc_root = _test_cases_root(root)
    if tc_root is None:
        raise FileNotFoundError('Need PyFR-Test-Cases directory')

    for spec in CASES:
        case = _resolve_case(tc_root, spec)
        print(f'Preparing {case["out_path"].name}…', flush=True)
        _ensure_pyfrm(case)
        _write_partitioned_pyfrm(case['pyfrm_path'], case['out_path'], force=force)
        print(f'  wrote {case["out_path"]}', flush=True)


def _read_face_view(view) -> np.ndarray:
    if hasattr(view, 'view'):
        mat = view.view._mats[0]
        return _read_view(view.view, mat)
    return _read_view(view, view._mats[0])


def _read_mpi_remote(xchg_mat) -> np.ndarray:
    return xchg_mat.get().ravel()


def _build_partitioned_system(cfg: Inifile, pyfrm: Path):
    backend = get_backend('openmp', cfg)
    reader = NativeReader(str(pyfrm), pname=PART_NAME)
    mesh = reader.mesh
    registers = list(TVDRK3Stepper._registers.keys())
    system = NavierStokesSystem(backend, mesh, None, registers, cfg,
                                Serialiser())
    eles = dict(system.ele_map)
    mpi_inters = list(system._mpi_inters)
    system.commit()
    reader.close()
    return system, eles, mpi_inters


def _run_av_scaling_mpi_exchange(system) -> None:
    _run_fill(system)

    pack = list(system._kernels['mpiint/av_scaling_fpts_pack', None, None])
    unpack = list(system._kernels['mpiint/av_scaling_fpts_unpack', None, None])
    send = list(system._mpireqs['av_scaling_fpts_send'])
    recv = list(system._mpireqs['av_scaling_fpts_recv'])

    for k in pack:
        k.run()

    for r in recv:
        r.Start()
    for r in send:
        r.Start()
    MPI.Request.Waitall(recv + send)

    for k in unpack:
        k.run()


def _check_mpi_exchange(mpi_inters, *, local_val: float,
                        remote_val: float) -> None:
    if not mpi_inters:
        raise AssertionError('expected MPI interfaces on a 2-rank mesh')

    comm = MPI.COMM_WORLD
    packers = comm.allreduce(
        sum('av_scaling_fpts_pack' in m.kernels for m in mpi_inters),
        op=MPI.SUM,
    )
    unpackers = comm.allreduce(
        sum('av_scaling_fpts_unpack' in m.kernels for m in mpi_inters),
        op=MPI.SUM,
    )
    assert packers > 0, 'no av_scaling_fpts_pack kernels registered'
    assert unpackers > 0, 'no av_scaling_fpts_unpack kernels registered'

    checked_remote = 0
    for m in mpi_inters:
        lhs = _read_face_view(m.av_scaling_l)
        np.testing.assert_allclose(
            lhs, local_val,
            err_msg='av_scaling_l should match the local rank constant',
        )

        if 'av_scaling_fpts_unpack' in m.kernels:
            rhs = _read_mpi_remote(m.av_scaling_r)
            np.testing.assert_allclose(
                rhs, remote_val,
                err_msg='av_scaling_r should match the remote rank constant',
            )
            checked_remote += 1

    checked_remote = comm.allreduce(checked_remote, op=MPI.SUM)
    assert checked_remote > 0, 'expected at least one recv/unpack MPI interface'


def _check_case(case: dict) -> None:
    comm, rank, root = get_comm_rank_root()

    cfg = Inifile.load(str(case['ini_path']))
    cfg.set('solver', 'shock-capturing', 'ec-artificial-viscosity')
    _configure_backend(cfg)

    system, eles, mpi_inters = _build_partitioned_system(cfg, case['out_path'])
    assert system._ec_av is not None

    local_val = float(rank + 1)
    remote_val = float(comm.size - rank)

    for eleset in eles.values():
        eleset._av_scaling.set(
            np.full((1, eleset.neles), local_val, dtype=float)
        )

    _run_av_scaling_mpi_exchange(system)
    _check_mpi_exchange(mpi_inters, local_val=local_val, remote_val=remote_val)

    rhs = _run_rhs(system)
    assert np.all(np.isfinite(rhs)), f'{case["label"]}: non-finite RHS'

    if rank == root:
        print(f'  {case["label"]}: MPI exchange + RHS smoke PASS', flush=True)


def run_mpi_tests(repo_root: Path) -> None:
    comm, rank, root_rank = get_comm_rank_root()

    if comm.size != NPARTS:
        raise RuntimeError(
            f'run with exactly {NPARTS} MPI ranks (got {comm.size})'
        )

    tc_root = _test_cases_root(repo_root)
    if tc_root is None:
        raise FileNotFoundError('Need PyFR-Test-Cases directory')

    if rank == root_rank:
        print('EC-AV MPI face-scaling transport checks', flush=True)

    for spec in CASES:
        case = _resolve_case(tc_root, spec)
        if not case['out_path'].is_file():
            raise FileNotFoundError(
                f'Missing {case["out_path"]}; run: python ecav_mpi_demo.py --prepare'
            )
        _check_case(case)

    if rank == root_rank:
        print('PASS: EC-AV MPI plumbing verification complete.', flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument(
        '--prepare', action='store_true',
        help='create *-quad-2p.pyfrm files with embedded 2-rank partitionings',
    )
    ap.add_argument(
        '--force', action='store_true',
        help='with --prepare, replace an existing partitioning',
    )
    args = ap.parse_args()

    repo_root = Path(__file__).resolve().parent

    if args.prepare:
        if MPI.Is_initialized():
            comm = MPI.COMM_WORLD
            if comm.size > 1:
                if comm.rank == 0:
                    print('--prepare must run with a single MPI rank',
                          file=sys.stderr)
                return 1
        else:
            init_mpi()

        prepare_partitioned_meshes(repo_root, force=args.force)
        print('Partitioned meshes ready. Run:', flush=True)
        print(f'  mpirun -n {NPARTS} {sys.executable} {Path(__file__).name}',
              flush=True)
        return 0

    if not MPI.Is_initialized():
        init_mpi()

    try:
        run_mpi_tests(repo_root)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        print('Try: python ecav_mpi_demo.py --prepare', file=sys.stderr)
        return 1
    except AssertionError as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    sys.exit(main())
