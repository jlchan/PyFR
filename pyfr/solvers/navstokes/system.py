from pyfr.mpiutil import get_comm_rank_root
from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionSystem
from pyfr.solvers.baseadvecdiff.artvisc import ArtificialViscosity
from pyfr.solvers.navstokes.ecartvisc import ECArtificialViscosity
from pyfr.solvers.navstokes.elements import NavierStokesElements
from pyfr.solvers.navstokes.inters import (NavierStokesBaseBCInters,
                                           NavierStokesIntInters,
                                           NavierStokesMPIInters)


class NavierStokesSystem(BaseAdvectionDiffusionSystem):
    name = 'navier-stokes'
    ef_solver = 'euler'

    elementscls = NavierStokesElements
    intinterscls = NavierStokesIntInters
    mpiinterscls = NavierStokesMPIInters
    bbcinterscls = NavierStokesBaseBCInters

    _shock_capturing_modes = (
        BaseAdvectionDiffusionSystem._shock_capturing_modes |
        {'ec-artificial-viscosity'}
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        comm, rank, root = get_comm_rank_root()
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        if rank == root:
            print(f'Running with shock-capturing = {shock_capturing}')

        if shock_capturing == 'ec-artificial-viscosity':
            self._ec_av = True
            self._av = ArtificialViscosity(
                self.backend, self.cfg, self.mesh, self.ele_map,
                producer='ecav',
            )
            self.backend.commit()
            self._setup_artvisc_interfaces()
            self._av.prepare_mpi()
            self._extra_kern_parts = {'vtx': self._av.kern_parts}
            self._extra_mpi_parts = self._av.mpi_parts
        else:
            self._ec_av = None
