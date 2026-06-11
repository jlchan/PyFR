from pyfr.integrators import get_integrator
from pyfr.solvers.base import BaseSystem
from pyfr.solvers.euler import EulerSystem
from pyfr.solvers.navstokes import NavierStokesSystem
from pyfr.util import subclass_where


def get_solver(backend, mesh, initsoln, cfg):
    systemcls = subclass_where(BaseSystem, name=cfg.get('solver', 'system'))

    shock_capturing = cfg.get('solver', 'shock-capturing', 'none')
    if shock_capturing not in systemcls._shock_capturing_modes:
        raise ValueError(
            f'Invalid shock-capturing option {shock_capturing!r}; expected '
            f'one of {sorted(systemcls._shock_capturing_modes)}'
        )

    # Combine with an integrator to yield the solver
    return get_integrator(backend, systemcls, mesh, initsoln, cfg)
