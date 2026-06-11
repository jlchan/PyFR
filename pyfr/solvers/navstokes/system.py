from pyfr.mpiutil import get_comm_rank_root
from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionSystem
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
            for eles in self.ele_map.values():
                ECArtificialViscosity.setup_tflux_views(eles)
            self._ec_av = ECArtificialViscosity(self)
            self.backend.commit()
        else:
            self._ec_av = None
