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

        eles._ecav_surf = be.matrix(
            (1, eles.neles), extent=nonce + 'ecav_surf', tags={'align'}
        )
        eles._ecav_vol = be.matrix(
            (1, eles.neles), extent=nonce + 'ecav_vol', tags={'align'}
        )
        eles._ecav_resid = be.matrix(
            (1, eles.neles), extent=nonce + 'ecav_resid', tags={'align'}
        )

        cls._setup_entropy_kernels(eles)
        cls._setup_ecav_diss(eles)
        cls._setup_ecav_resid(eles)

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
    def _setup_ecav_resid(cls, eles):
        be = eles._be
        kprefix = 'pyfr.solvers.navstokes.kernels'
        c = eles.cfg.items_as('constants', float)
        fpdtype = be.fpdtype

        pnorm = eles._pnorm_fpts.transpose(0, 2, 1)
        qwts_pnorm = (
            eles.basis.fpts_wts[:, None, None] * pnorm
        ).astype(fpdtype)
        eles._ecav_qnorm_fpts = be.const_matrix(qwts_pnorm, tags={'align'})

        be.pointwise.register(f'{kprefix}.ecav_surface_integral')
        be.pointwise.register(f'{kprefix}.ecav_volume_integral')
        be.pointwise.register(f'{kprefix}.ecav_entropy_resid')
        be.pointwise.register(f'{kprefix}.ecav_av_scaling')

        tplargs = {
            'ndims': eles.ndims,
            'nvars': eles.nvars,
            'nupts': eles.nupts,
            'c': c,
        }

        r, s = eles.mesh_regions, eles._slice_mat
        resid_rgn = []

        for rgn in ('curved', 'linear'):
            if rgn not in r:
                continue
            resid_rgn.append((rgn, r[rgn]))

        if resid_rgn:
            def ecav_surface_integral(uin):
                return eles._make_sliced_kernel(
                    be.kernel(
                        'ecav_surface_integral', tplargs=tplargs,
                        dims=[eles.nfpts, n],
                        uin=s(eles._scal_fpts, rgn),
                        qnorm=s(eles._ecav_qnorm_fpts, rgn),
                        surf=eles._ecav_surf,
                    )
                    for rgn, n in resid_rgn
                )

            def ecav_volume_integral(uin):
                return eles._make_sliced_kernel(
                    be.kernel(
                        'ecav_volume_integral', tplargs=tplargs,
                        dims=[eles.nupts, n],
                        uin=s(eles.scal_upts[uin], rgn),
                        gradv=s(eles._grad_upts, rgn),
                        wts=s(eles._ecav_wts_upts, rgn),
                        vol=eles._ecav_vol,
                    )
                    for rgn, n in resid_rgn
                )

            def _ensure_ecav_ele_views():
                if hasattr(eles, '_ecav_resid_ele_view'):
                    return
                mat_surf = eles._ecav_surf
                mat_vol = eles._ecav_vol
                mat_resid = eles._ecav_resid
                eles._ecav_surf_ele_view = be.view(
                    np.full(eles.neles, mat_surf.mid),
                    np.zeros(eles.neles, dtype=int),
                    np.arange(eles.neles),
                )
                eles._ecav_vol_ele_view = be.view(
                    np.full(eles.neles, mat_vol.mid),
                    np.zeros(eles.neles, dtype=int),
                    np.arange(eles.neles),
                )
                eles._ecav_resid_ele_view = be.view(
                    np.full(eles.neles, mat_resid.mid),
                    np.zeros(eles.neles, dtype=int),
                    np.arange(eles.neles),
                )

            def ecav_entropy_resid():
                _ensure_ecav_ele_views()
                return be.kernel(
                    'ecav_entropy_resid', tplargs={},
                    dims=[eles.neles],
                    surf=eles._ecav_surf_ele_view,
                    vol=eles._ecav_vol_ele_view,
                    resid=eles._ecav_resid_ele_view,
                )

            tplargs_av = {'eps': float(np.finfo(fpdtype).eps)}

            def ecav_av_scaling():
                _ensure_ecav_ele_views()
                if not hasattr(eles, '_ecav_diss_ele_view'):
                    mat_diss = eles._ecav_diss
                    eles._ecav_diss_ele_view = be.view(
                        np.full(eles.neles, mat_diss.mid),
                        np.zeros(eles.neles, dtype=int),
                        np.arange(eles.neles),
                    )
                if not hasattr(eles, '_av_scaling_ele_view'):
                    mat_av = eles._av_scaling
                    eles._av_scaling_ele_view = be.view(
                        np.full(eles.neles, mat_av.mid),
                        np.zeros(eles.neles, dtype=int),
                        np.arange(eles.neles),
                    )

                return be.kernel(
                    'ecav_av_scaling', tplargs=tplargs_av,
                    dims=[eles.neles],
                    resid=eles._ecav_resid_ele_view,
                    diss=eles._ecav_diss_ele_view,
                    av_scaling=eles._av_scaling_ele_view,
                )

            eles.kernels['ecav_surface_integral'] = ecav_surface_integral
            eles.kernels['ecav_volume_integral'] = ecav_volume_integral
            eles.kernels['ecav_entropy_resid'] = ecav_entropy_resid
            eles.kernels['ecav_av_scaling'] = ecav_av_scaling

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
