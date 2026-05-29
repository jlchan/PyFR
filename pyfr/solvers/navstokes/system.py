from pyfr.mpiutil import get_comm_rank_root
from pyfr.solvers.baseadvecdiff import BaseAdvectionDiffusionSystem
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        comm, rank, root = get_comm_rank_root()
        if rank == root:
            gradvars = self.cfg.get(
                'solver', 'gradient-variables', 'conservative'
            )
            print(f'Running with gradient-variables = {gradvars}')
