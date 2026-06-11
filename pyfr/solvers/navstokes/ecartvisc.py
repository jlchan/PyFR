import numpy as np


class ECArtificialViscosity:
    """Entropy-gradient and EC artificial-viscosity plumbing for Navier–Stokes."""

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

        eles.av_scaling_views = {}
        eles._av_scaling = be.matrix(
            (1, eles.neles), initval=np.full((1, eles.neles), 1.0),
            extent=nonce + 'av_scaling', tags={'align'}
        )
        eles._av_scaling_fpts = be.matrix(
            (eles.nfpts, eles.neles),
            extent=nonce + 'av_scaling_fpts', tags={'align'}
        )

        be.pointwise.register(f'{kprefix}.ecav_scaling_fpts')

        def ecav_scaling_fpts_kern():
            return be.kernel(
                'ecav_scaling_fpts', tplargs={'nfpts': eles.nfpts},
                dims=[eles.neles],
                av_scaling=eles._av_scaling_ele_view,
                av_scaling_fpts=eles._av_scaling_fpts,
            )

        eles.kernels['ecav_scaling_fpts'] = ecav_scaling_fpts_kern

        cls._setup_entropy_kernels(eles)

    @classmethod
    def _tdisf_npts(cls, eles):
        if eles.grad_fusion:
            return eles.nupts
        elif 'flux' in eles.antialias:
            return eles.nqpts
        else:
            return eles.nupts

    @classmethod
    def setup_tflux_views(cls, eles):
        be = eles._be
        npts = cls._tdisf_npts(eles)
        mat = eles._av_scaling
        neles = eles.neles

        eles._av_scaling_ele_view = be.view(
            np.full(neles, mat.mid),
            np.zeros(neles, dtype=int),
            np.arange(neles),
        )

        for rgn in ('curved', 'linear'):
            if rgn not in eles.mesh_regions:
                continue

            off = 0 if rgn == 'curved' else eles.linoff
            n_rgn = eles.mesh_regions[rgn]
            n = n_rgn * npts

            matmap = np.full(n, mat.mid)
            rmap = np.zeros(n, dtype=int)
            cmap = np.repeat(np.arange(off, off + n_rgn), npts)

            eles.av_scaling_views[rgn] = be.view(matmap, rmap, cmap)

    def __init__(self, system):
        self._system = system

        av_scaling_fpts = {et: e._av_scaling_fpts
                           for et, e in system.ele_map.items()}
        iint_v, mpi_v, bc_v = system.make_field_views(av_scaling_fpts,
                                                      vshape=())

        for i, (lhs, rhs) in zip(system._int_inters, iint_v):
            i.av_scaling_l = lhs
            i.av_scaling_r = rhs
        for m, (lhs, rhs) in zip(system._mpi_inters, mpi_v):
            m.av_scaling_l = lhs
            m.av_scaling_r = rhs
        for b, lhs in zip(system._bc_inters, bc_v):
            b.av_scaling_l = lhs

        system.register_mpi_exchange(
            'av_scaling_fpts', mpi_v,
            send=lambda m: m.c['ldg-beta'] != -0.5,
            recv=lambda m: m.c['ldg-beta'] != 0.5,
        )

    def volume_deps(self, k):
        return []

    def face_deps(self, k):
        return []

    def add_to_graph_grad_flux(self, g, k, m, deps):
        g.add_all(k['eles/ecav_scaling_fpts'])
        g.add_all(k['mpiint/av_scaling_fpts_pack'],
                  deps=k['eles/ecav_scaling_fpts'])
        for send, pack in zip(m['av_scaling_fpts_send'],
                              k['mpiint/av_scaling_fpts_pack']):
            g.add_mpi_req(send, deps=[pack])

    def add_to_graph_mpi_flux(self, g, k, deps):
        g.add_all(k['mpiint/av_scaling_fpts_unpack'])

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
