import numpy as np

from pyfr.solvers.baseadvecdiff import (BaseAdvectionDiffusionBCInters,
                                        BaseAdvectionDiffusionIntInters,
                                        BaseAdvectionDiffusionMPIInters)
from pyfr.solvers.euler.inters import MassFlowBCMixin, PressureBCMixin
from pyfr.solvers.navstokes.ecartvisc import ECArtificialViscosity


class TplargsMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._entropy_gradients = ECArtificialViscosity.enabled(self.cfg)

        rsolver = self.cfg.get('solver-interfaces', 'riemann-solver')
        visc_corr = self.cfg.get('solver', 'viscosity-correction', 'none')
        shock_capturing = self.cfg.get('solver', 'shock-capturing', 'none')
        av_active = (
            shock_capturing == 'artificial-viscosity'
            or ECArtificialViscosity.enabled(self.cfg)
        )
        if shock_capturing == 'entropy-filter':
            self.p_min = self.cfg.getfloat('solver-entropy-filter', 'p-min',
                                           1e-6)
        else:
            self.p_min = self.cfg.getfloat('solver-interfaces', 'p-min',
                                           5*self._be.fpdtype_eps)

        self._tplargs = dict(ndims=self.ndims, nvars=self.nvars,
                             rsolver=rsolver, visc_corr=visc_corr,
                             shock_capturing=shock_capturing, av_active=av_active,
                             c=self.c, p_min=self.p_min)


class NavierStokesIntInters(TplargsMixin,
                            BaseAdvectionDiffusionIntInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.intcflux')

        if self._entropy_gradients:
            self._be.pointwise.register(
                'pyfr.solvers.navstokes.kernels.intentconu'
            )
            self._ent_comm_lhs = self._scal_view(
                self.lhs, 'get_ent_comm_fpts_for_inters'
            )
            self._ent_comm_rhs = self._scal_view(
                self.rhs, 'get_ent_comm_fpts_for_inters'
            )
            self.kernels['con_u'] = lambda: self._be.kernel(
                'intentconu', tplargs=self._tplargs, dims=[self.ninterfpts],
                ulin=self.scal_lhs, urin=self.scal_rhs,
                ulout=self._ent_comm_lhs, urout=self._ent_comm_rhs
            )
        else:
            self._be.pointwise.register(
                'pyfr.solvers.navstokes.kernels.intconu'
            )
            self.kernels['con_u'] = lambda: self._be.kernel(
                'intconu', tplargs=self._tplargs, dims=[self.ninterfpts],
                ulin=self.scal_lhs, urin=self.scal_rhs,
                ulout=self._comm_lhs, urout=self._comm_rhs
            )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'intcflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self.scal_lhs, ur=self.scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artvisc=self.artvisc, nl=self._pnorm_lhs
        )


class NavierStokesMPIInters(TplargsMixin,
                            BaseAdvectionDiffusionMPIInters):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.mpicflux')

        if self._entropy_gradients:
            self._be.pointwise.register(
                'pyfr.solvers.navstokes.kernels.mpientconu'
            )
            self._ent_comm_lhs = self._scal_xchg_view(
                self.lhs, 'get_ent_comm_fpts_for_inters'
            )
            self._ent_comm_rhs = self._be.xchg_matrix_for_view(
                self._ent_comm_lhs
            )
            self.kernels['con_u'] = lambda: self._be.kernel(
                'mpientconu', tplargs=self._tplargs, dims=[self.ninterfpts],
                ulin=self.scal_lhs, urin=self.scal_rhs,
                ulout=self._ent_comm_lhs
            )
        else:
            self._be.pointwise.register(
                'pyfr.solvers.navstokes.kernels.mpiconu'
            )
            self.kernels['con_u'] = lambda: self._be.kernel(
                'mpiconu', tplargs=self._tplargs, dims=[self.ninterfpts],
                ulin=self.scal_lhs, urin=self.scal_rhs, ulout=self._comm_lhs
            )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'mpicflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            ul=self.scal_lhs, ur=self.scal_rhs,
            gradul=self._vect_lhs, gradur=self._vect_rhs,
            artvisc=self.artvisc, nl=self._pnorm_lhs
        )


class NavierStokesBaseBCInters(TplargsMixin, BaseAdvectionDiffusionBCInters):
    cflux_state = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Additional BC specific template arguments
        self._tplargs['bctype'] = self.type
        self._tplargs['bccfluxstate'] = self.cflux_state

        self._be.pointwise.register('pyfr.solvers.navstokes.kernels.bccflux')

        if self._entropy_gradients:
            self._be.pointwise.register(
                'pyfr.solvers.navstokes.kernels.bcentconu'
            )
            self._ent_comm_lhs = self._scal_view(
                self.lhs, 'get_ent_comm_fpts_for_inters'
            )
            self.kernels['con_u'] = lambda: self._be.kernel(
                'bcentconu', tplargs=self._tplargs, dims=[self.ninterfpts],
                extrns=self._external_args, ulin=self.scal_lhs,
                ulout=self._ent_comm_lhs, nlin=self._pnorm_lhs,
                **self._external_vals
            )
        else:
            self._be.pointwise.register(
                'pyfr.solvers.navstokes.kernels.bcconu'
            )
            self.kernels['con_u'] = lambda: self._be.kernel(
                'bcconu', tplargs=self._tplargs, dims=[self.ninterfpts],
                extrns=self._external_args, ulin=self.scal_lhs,
                ulout=self._comm_lhs, nlin=self._pnorm_lhs,
                **self._external_vals
            )
        self.kernels['comm_flux'] = lambda: self._be.kernel(
            'bccflux', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, ul=self.scal_lhs,
            gradul=self._vect_lhs, nl=self._pnorm_lhs,
            artvisc=self.artvisc, **self._external_vals
        )

    def comm_entropy_kernel(self, entmin_lhs):
        # Physics-specific callback for entropy filtering
        self._be.pointwise.register(
            'pyfr.solvers.navstokes.kernels.bccent'
        )

        return lambda: self._be.kernel(
            'bccent', tplargs=self._tplargs, dims=[self.ninterfpts],
            extrns=self._external_args, entmin_lhs=entmin_lhs,
            nl=self._pnorm_lhs, ul=self.scal_lhs, **self._external_vals
        )


class NavierStokesNoSlpIsotWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-isot-wall'
    cflux_state = 'ghost-imperm'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c['cpTw'], = self._eval_opts(['cpTw'])
        self.c |= self._exp_opts('uvw'[:self.ndims], lhs,
                                 default={'u': 0, 'v': 0, 'w': 0})


class NavierStokesNoSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'no-slp-adia-wall'
    cflux_state = 'ghost-imperm'


class NavierStokesSlpAdiaWallBCInters(NavierStokesBaseBCInters):
    type = 'slp-adia-wall'
    cflux_state = None


class NavierStokesCharRiemInvBCInters(NavierStokesBaseBCInters):
    type = 'char-riem-inv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupInflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-in-fa'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'p', 'u', 'v', 'w'][:self.ndims + 2], lhs
        )


class NavierStokesSupOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sup-out-fn'
    cflux_state = 'ghost'


class NavierStokesSubInflowFrvBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-frv'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(
            ['rho', 'u', 'v', 'w'][:self.ndims + 1], lhs,
            default={'u': 0, 'v': 0, 'w': 0}
        )


class NavierStokesSubInflowFtpttangBCInters(NavierStokesBaseBCInters):
    type = 'sub-in-ftpttang'
    cflux_state = 'ghost'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        gamma = self.cfg.getfloat('constants', 'gamma')

        # Pass boundary constants to the backend
        self.c['cpTt'], = self._eval_opts(['cpTt'])
        self.c['pt'], = self._eval_opts(['pt'])
        self.c['Rdcp'] = (gamma - 1.0)/gamma

        # Calculate u, v velocity components from the inflow angle
        theta = self._eval_opts(['theta'])[0]*np.pi/180.0
        velcomps = np.array([np.cos(theta), np.sin(theta), 1.0])

        # Adjust u, v and calculate w velocity components for 3-D
        if self.ndims == 3:
            phi = self._eval_opts(['phi'])[0]*np.pi/180.0
            velcomps[:2] *= np.sin(phi)
            velcomps[2] *= np.cos(phi)

        self.c['vc'] = velcomps[:self.ndims]


class NavierStokesSubOutflowBCInters(NavierStokesBaseBCInters):
    type = 'sub-out-fp'
    cflux_state = 'ghost'

    def __init__(self, be, lhs, elemap, cfgsect, cfg, bccomm):
        super().__init__(be, lhs, elemap, cfgsect, cfg, bccomm)

        self.c |= self._exp_opts(['p'], lhs)


class NavierStokesCharRiemInvMassFlowBCInters(MassFlowBCMixin,
                                              NavierStokesBaseBCInters):
    type = 'char-riem-inv-mass-flow'
    cflux_state = 'ghost'


class NavierStokesCharRiemInvPressureBCInters(PressureBCMixin,
                                              NavierStokesBaseBCInters):
    type = 'char-riem-inv-pressure'
    cflux_state = 'ghost'
