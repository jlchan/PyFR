"""
Print the corrected physical gradients at solution points after a single RHS.

Builds the integrator, runs one full RHS evaluation (graphs g1 + g2 + g3),
then reads back _grad_upts for each element type.  _grad_upts has shape
(ndims*nupts, nvars, neles); the ndims blocks of nupts rows correspond to
the d/dx, d/dy (, d/dz) components of the gradient.
"""

import argparse

import mpi4py.rc
mpi4py.rc.initialize = False

import numpy as np

from pyfr.backends import get_backend
from pyfr.inifile import Inifile
from pyfr.mpiutil import init_mpi
from pyfr.readers.native import NativeReader  # mesh only; no soln loaded
from pyfr.solvers import get_solver


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('mesh', help='path to .pyfrm file')
    ap.add_argument('cfg',  help='path to .ini file')
    args = ap.parse_args()

    init_mpi()

    mesh = NativeReader(args.mesh).mesh
    cfg  = Inifile.load(args.cfg)

    backend = get_backend('openmp', cfg)
    intg    = get_solver(backend, mesh, None, cfg)

    uin  = intg._idxcurr
    fout = 1 - uin

    # Full RHS: g1 fills _grad_vars_upts and interpolates to flux points,
    # g2 computes the corrected gradients and converts to physical space,
    # g3 finishes MPI flux assembly.
    intg.system.rhs(intg.tcurr, uin, fout)

    ndims    = intg.system.ndims
    convars  = intg.system.elementscls.convars(ndims, cfg)
    dir_names = ['d/dx', 'd/dy', 'd/dz'][:ndims]

    for etype, soln_banks, grad_mat in zip(intg.system.ele_types,
                                           intg.system.ele_banks,
                                           intg.system.eles_vect_upts):
        # soln shape: (nupts, nvars, neles)
        u = soln_banks[uin].get()
        nupts = u.shape[0]
        nvars = u.shape[1]
        neles = u.shape[2]

        print(f'\n=== Element type: {etype}  '
              f'(nupts={nupts}, nvars={nvars}, neles={neles}) ===')

        print(f'\n  --- solution ---')
        for v, vname in enumerate(convars):
            vals = u[:, v, :]   # (nupts, neles)
            print(f'  {vname:>8s}: min={vals.min():.6e}  '
                  f'max={vals.max():.6e}  mean={vals.mean():.6e}')

        # grad_mat shape: (ndims*nupts, nvars, neles)
        gradu = grad_mat.get().reshape(ndims, nupts, nvars, neles)

        for d, dname in enumerate(dir_names):
            print(f'\n  --- {dname} ---')
            for v, vname in enumerate(convars):
                vals = gradu[d, :, v, :]   # (nupts, neles)
                print(f'  {vname:>8s}: min={vals.min():.6e}  '
                      f'max={vals.max():.6e}  mean={vals.mean():.6e}')


if __name__ == '__main__':
    main()
