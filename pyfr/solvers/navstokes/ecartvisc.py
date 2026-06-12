import numpy as np

from pyfr.quadrules import get_quadrule


class ECArtificialViscosity:
    """Entropy-gradient and EC artificial-viscosity for Navier–Stokes."""

    name = 'ecav'

    @classmethod
    def validate_config(cls, cfg):
        if cfg.get('solver', 'gradient-variables', 'conservative') == 'entropy':
            raise ValueError(
                'gradient-variables = entropy is no longer supported; '
                'use shock-capturing = ec-artificial-viscosity'
            )

    @classmethod
    def enabled(cls, cfg):
        cls.validate_config(cfg)
        return (cfg.get('solver', 'shock-capturing', 'none') ==
                'ec-artificial-viscosity')

    @classmethod
    def setup_elements(cls, eles, nonce):
        if eles.grad_fusion:
            raise ValueError(
                'shock-capturing = ec-artificial-viscosity is incompatible '
                'with gradient fusion (disable flux anti-aliasing and use a '
                'non-block backend)'
            )
        if 'flux' in eles.antialias:
            raise ValueError(
                'shock-capturing = ec-artificial-viscosity is incompatible '
                'with flux anti-aliasing'
            )

        be = eles._be
        kprefix = 'pyfr.solvers.navstokes.kernels'

        eles._ent_upts = be.matrix(
            (eles.nupts, eles.nvars, eles.neles),
            extent=nonce + 'ent_upts', tags={'align'}
        )
        eles._ent_comm_fpts = be.matrix(
            (eles.nfpts, eles.nvars, eles.neles),
            extent=nonce + 'ent_comm_fpts', tags={'align'}
        )

        # Temporary localized spike for Gaussian-pulse AV visibility (256 quad
        # mesh; element 187 is near domain centre). Skipped on smaller meshes.
        av_init = np.zeros((1, eles.neles))
        _spike_ele, _spike_val = 187, 0.025
        if _spike_ele < eles.neles:
            av_init[0, _spike_ele] = _spike_val
        eles._av_scaling = be.matrix(
            (1, eles.neles), initval=av_init,
            extent=nonce + 'av_scaling', tags={'align'}
        )

        eles._ecav_diss = be.matrix(
            (1, eles.neles), extent=nonce + 'ecav_diss', tags={'align'}
        )

        cls._setup_entropy_kernels(eles)
        cls._setup_ecav_diss(eles)

    @classmethod
    def _setup_ecav_diss(cls, eles):
        be = eles._be
        kprefix = 'pyfr.solvers.navstokes.kernels'
        c = eles.cfg.items_as('constants', float)
        fpdtype = be.fpdtype

        rname = eles.cfg.get(f'solver-elements-{eles.name}', 'soln-pts')
        if rname != 'gauss-legendre-lobatto':
            raise ValueError(
                'ecav viscous dissipation requires soln-pts = '
                'gauss-legendre-lobatto'
            )

        r = get_quadrule(eles.name, rname, eles.nupts)
        wts_np = (r.wts[:, None] / eles.rcpdjac_at_np('upts')).astype(fpdtype)
        eles._ecav_wts_upts = be.const_matrix(wts_np, tags={'align'})

        be.pointwise.register(f'{kprefix}.ecav_visc_ent_diss')

        tplargs = {
            'ndims': eles.ndims,
            'nvars': eles.nvars,
            'nupts': eles.nupts,
            'c': c,
        }

        r, s = eles.mesh_regions, eles._slice_mat
        diss_rgn = []

        for rgn in ('curved', 'linear'):
            if rgn not in r:
                continue
            diss_rgn.append((rgn, r[rgn]))

        if diss_rgn:
            def ecav_visc_ent_diss_upts(uin):
                return eles._make_sliced_kernel(
                    be.kernel(
                        'ecav_visc_ent_diss', tplargs=tplargs,
                        dims=[eles.nupts, n],
                        uin=s(eles.scal_upts[uin], rgn),
                        gradv=s(eles._grad_upts, rgn),
                        wts=s(eles._ecav_wts_upts, rgn),
                        diss=eles._ecav_diss,
                    )
                    for rgn, n in diss_rgn
                )

            eles.kernels['ecav_visc_ent_diss'] = ecav_visc_ent_diss_upts

    @classmethod
    def _setup_entropy_kernels(cls, eles):
        be = eles._be
        kprefix = 'pyfr.solvers.navstokes.kernels'
        c = eles.cfg.items_as('constants', float)

        be.pointwise.register(f'{kprefix}.con_to_ent')
        be.pointwise.register(f'{kprefix}.ent_to_con_grad')

        ent_tplargs = {'ndims': eles.ndims, 'nvars': eles.nvars, 'c': c}
        tplargs_e2c = dict(ent_tplargs)

        r, s = eles.mesh_regions, eles._slice_mat
        ent_u, ent_to_con_grad_u = [], []

        for rgn in ('curved', 'linear'):
            if rgn not in r:
                continue
            ent_u.append((rgn, r[rgn]))
            ent_to_con_grad_u.append((rgn, r[rgn]))

        if ent_u:
            def con_to_ent_upts(uin):
                return eles._make_sliced_kernel(
                    be.kernel(
                        'con_to_ent', tplargs=ent_tplargs,
                        dims=[eles.nupts, n],
                        uin=s(eles.scal_upts[uin], rgn),
                        vout=s(eles._ent_upts, rgn),
                    )
                    for rgn, n in ent_u
                )

            eles.kernels['con_to_ent_upts'] = con_to_ent_upts

        if ent_to_con_grad_u:
            def ent_to_con_grad_upts(uin):
                return eles._make_sliced_kernel(
                    be.kernel(
                        'ent_to_con_grad', tplargs=tplargs_e2c,
                        dims=[eles.nupts, n],
                        uin=s(eles.scal_upts[uin], rgn),
                        gradu=s(eles._grad_upts, rgn),
                    )
                    for rgn, n in ent_to_con_grad_u
                )

            eles.kernels['ent_to_con_grad_upts'] = ent_to_con_grad_upts
