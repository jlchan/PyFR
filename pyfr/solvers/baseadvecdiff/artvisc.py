import numpy as np

from pyfr.mpiutil import autofree, get_comm_rank_root
from pyfr.polys import get_polybasis


class _VtxPeer:
    def __init__(self, be, vtx_comm, rank, xv, rm, mv, n):
        self.name = f'p{rank}'

        self.kernels = {
            'pack': lambda: be.kernel('pack', xv),
            'unpack': lambda: be.kernel('unpack', rm),
            'merge': lambda: be.kernel(
                'vtxreduce', tplargs={}, dims=[n], vtx=mv, recv=rm
            ),
        }
        self.mpireqs = {
            'vtx_send': lambda: xv.sendreq(vtx_comm, rank, 0),
            'vtx_recv': lambda: rm.recvreq(vtx_comm, rank, 0),
        }


class ArtificialViscosity:
    name = 'av'

    def __init__(self, backend, cfg, mesh, ele_map, *, producer='sensor'):
        self._be = backend
        self._mesh = mesh
        self._producer = producer

        if producer not in {'sensor', 'ecav'}:
            raise ValueError(f'invalid AV producer {producer!r}')

        # Allocate vertex buffer
        self._vtx_buf = backend.matrix((1, len(mesh.node_idxs)), extent='vtx')

        # Commit to give vtx_buf basedata (needed for view creation)
        backend.commit()

        # Register kernel templates
        kprefix = 'pyfr.solvers.baseadvecdiff.kernels'
        if producer == 'sensor':
            backend.pointwise.register(f'{kprefix}.shocksensor')
        else:
            backend.pointwise.register(
                'pyfr.solvers.navstokes.kernels.ecav_resid_av_to_vtx'
            )
        backend.pointwise.register(f'{kprefix}.avfill')

        # AV config constants (sensor mode only)
        c_av = (cfg.items_as('solver-artificial-viscosity', float)
                if producer == 'sensor' else {})

        # Per-element-type setup
        for etype, eles in ele_map.items():
            self._setup_etype(etype, eles, mesh, c_av)

    @property
    def producer_key(self):
        return ('shocksensor' if self._producer == 'sensor'
                else 'ecav_resid_av_to_vtx')

    def _setup_etype(self, etype, eles, mesh, c_av):
        be = self._be
        basis = eles.basis
        nverts = len(basis.linspts)
        enodes = mesh.spts_nodes[etype]

        # Find which shape point indices correspond to vertices
        spts = np.array(basis.spts)
        linspts = np.array(basis.linspts)
        vtx_idxs = np.argmin(
            np.linalg.norm(spts - linspts[:, None], axis=2), axis=1
        )
        vnodes = enodes[:, vtx_idxs]

        # Allocate artvisc at fpts (populated by avfill kernel)
        artvisc_fpts = be.matrix(
            (eles.nfpts, eles.neles), tags={'align'},
            extent='artvisc_fpts'
        )
        eles.artvisc_fpts = artvisc_fpts

        # Linear interpolation basis for vertex → point interpolation
        linbasis = get_polybasis(basis.name, 1, basis.linspts)

        # Build vertex views (full view + per-region views)
        self._setup_vtx_views(eles, mesh, nverts, vnodes)

        # Register kernel factories
        self._setup_kernels(eles, c_av, nverts, vnodes, linbasis, artvisc_fpts)

    def _setup_vtx_views(self, eles, mesh, nverts, vnodes):
        be = self._be

        # Build mapping arrays for vertex views
        n = eles.neles * nverts
        matmap = np.full(n, self._vtx_buf.mid)
        rmap = np.zeros(n, dtype=int)
        cmap = np.searchsorted(mesh.node_idxs, vnodes.ravel())

        # Full view for 1D kernels (producer, avfill)
        eles.vtx_view = be.view(matmap, rmap, cmap)

        # Per-region views for 2D tflux kernel
        for rgn in ('curved', 'linear'):
            if rgn in eles.mesh_regions:
                off = 0 if rgn == 'curved' else eles.linoff
                slc = slice(off*nverts, (off + eles.mesh_regions[rgn])*nverts)
                eles.vtx_views[rgn] = be.view(
                    matmap[slc], rmap[slc], cmap[slc]
                )

    def _setup_kernels(self, eles, c_av, nverts, vnodes, linbasis,
                       artvisc_fpts):
        be = self._be

        # Avfill template arguments
        tplargs_avfill = dict(
            nverts=nverts, nfpts=eles.nfpts,
            av_op=linbasis.nodal_basis_at(eles.basis.fpts).tolist()
        )

        if self._producer == 'sensor':
            tplargs_sensor = self._sensor_tplargs(eles, c_av)
            tplargs_sensor['nverts'] = nverts

            def producer_kern(uin):
                return be.kernel(
                    'shocksensor', tplargs=tplargs_sensor, dims=[eles.neles],
                    u=eles.scal_upts[uin], vtx=eles.vtx_view
                )
        else:
            fpdtype = be.fpdtype
            tplargs_ecav = {
                'nverts': nverts,
                'eps': float(np.finfo(fpdtype).eps),
            }

            def _ensure_ecav_producer_views():
                if hasattr(eles, '_ecav_surf_ele_view'):
                    return
                mat_surf = eles._ecav_surf
                mat_vol = eles._ecav_vol
                mat_diss = eles._ecav_diss
                mat_av = eles._av_scaling
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
                eles._ecav_diss_ele_view = be.view(
                    np.full(eles.neles, mat_diss.mid),
                    np.zeros(eles.neles, dtype=int),
                    np.arange(eles.neles),
                )
                eles._av_scaling_ele_view = be.view(
                    np.full(eles.neles, mat_av.mid),
                    np.zeros(eles.neles, dtype=int),
                    np.arange(eles.neles),
                )

            def producer_kern():
                _ensure_ecav_producer_views()
                return be.kernel(
                    'ecav_resid_av_to_vtx', tplargs=tplargs_ecav,
                    dims=[eles.neles],
                    surf=eles._ecav_surf_ele_view,
                    vol=eles._ecav_vol_ele_view,
                    diss=eles._ecav_diss_ele_view,
                    av_scaling=eles._av_scaling_ele_view,
                    vtx=eles.vtx_view,
                )

        def avfill_kern():
            return be.kernel(
                'avfill', tplargs=tplargs_avfill, dims=[eles.neles],
                vtx=eles.vtx_view, artvisc_fpts=artvisc_fpts
            )

        eles.kernels[self.producer_key] = producer_kern
        eles.kernels['avfill'] = avfill_kern

        # Set the getter for the exportable AV field (P1 vertex values)
        eles.artvisc_vtx_fn = lambda: self._av_at_vtx(vnodes)

    def _av_at_vtx(self, vnodes):
        vtx = self._vtx_buf.get().ravel()
        nidxs = self._mesh.node_idxs

        # Gather vertex values: (neles, nverts)
        vidxs = np.searchsorted(nidxs, vnodes.ravel())
        return vtx[vidxs].reshape(vnodes.shape)

    def _sensor_tplargs(self, eles, c_av):
        shockvar = eles.convars.index(eles.shockvar)

        ubname = eles.basis.ubasis.name
        ubdegs = eles.basis.ubasis.degrees
        uborder = eles.basis.ubasis.order

        lubdegs = get_polybasis(ubname, max(0, uborder - 1)).degrees
        ind_modes = [d not in lubdegs for d in ubdegs]

        mode_degs = [max(d) for d in ubdegs]
        ndeg = uborder + 1

        fit_degs = list(range(1, ndeg))
        if len(fit_degs) >= 2:
            x = np.log(np.array(fit_degs) + 1)
            A = np.column_stack([np.ones(len(x)), x])
            s_weights = np.linalg.pinv(A)[1].tolist()

            N = uborder
            bd2 = np.array([1.0/(d + 1)**(2*N) for d in fit_degs])
            bd2 /= bd2.sum()
            baseline_decay = bd2.tolist()
        else:
            s_weights = None
            baseline_decay = None

        return dict(
            nvars=eles.nvars, nupts=eles.nupts, svar=shockvar,
            c=c_av, order=eles.basis.order, ind_modes=ind_modes,
            invvdm=eles.basis.ubasis.invvdm.T,
            mode_degs=mode_degs, ndeg=ndeg, s_weights=s_weights,
            baseline_decay=baseline_decay
        )

    def prepare_mpi(self):
        be = self._be
        mesh = self._mesh

        # Duplicate the communicator to isolate vertex exchange tags
        comm, _, _ = get_comm_rank_root()
        self._vtx_comm = autofree(comm.Dup())

        # Register vtxreduce kernel template
        kprefix = 'pyfr.solvers.baseadvecdiff.kernels'
        be.pointwise.register(f'{kprefix}.vtxreduce')

        # Create per-peer exchange objects
        self._peers = []
        for rank, shared_global in mesh.shared_nodes.by_rank.items():
            shared_local = np.searchsorted(mesh.node_idxs, shared_global)
            n = len(shared_local)

            matmap = np.full(n, self._vtx_buf.mid)
            rmap = np.zeros(n, dtype=int)
            cmap = shared_local

            xv = be.xchg_view(matmap, rmap, cmap)
            rm = be.xchg_matrix_for_view(xv)
            mv = be.view(matmap, rmap, cmap)

            self._peers.append(
                _VtxPeer(be, self._vtx_comm, rank, xv, rm, mv, n)
            )

        # Global kernels (not per-peer)
        self.kernels = {
            'vtx_zero': lambda: be.kernel('zero', self._vtx_buf),
        }
        self.mpireqs = {}

        # Provider lists for the system
        self.kern_parts = [self] + self._peers
        self.mpi_parts = list(self._peers)

    # Shock capturing interface
    @property
    def extra_kern_parts(self):
        return {'vtx': self.kern_parts}

    @property
    def extra_mpi_parts(self):
        return self.mpi_parts

    def soln_deps(self, k):
        return []

    def flux_deps(self, k):
        return k['eles/avfill']

    def bc_flux_deps(self, k):
        return []

    def preproc_graphs(self, be, k, m, deps):
        return ()

    def postproc(self, be, k):
        pass

    def add_to_graph_pre_recv(self, g, k, m):
        if self._producer != 'sensor':
            raise RuntimeError('use EC graph hooks for ecav producer')

        pk = f'eles/{self.producer_key}'
        # Vertex exchange: post receives, zero buffer, run producer, pack, send
        g.add_mpi_reqs(m['vtx_recv'])
        g.add_all(k['vtx/vtx_zero'])
        g.add_all(k[pk], deps=k['vtx/vtx_zero'])
        g.add_all(k['vtx/pack'], deps=k[pk])
        for send, pack in zip(m['vtx_send'], k['vtx/pack']):
            g.add_mpi_req(send, deps=[pack])

    def add_to_graph_ecav_pre_recv(self, g, k, m):
        g.add_mpi_reqs(m['vtx_recv'])
        g.add_all(k['vtx/vtx_zero'])

    def add_to_graph_ecav_produce(self, g, k, m, deps):
        pk = f'eles/{self.producer_key}'
        g.add_all(k[pk], deps=k['vtx/vtx_zero'] + deps)
        g.add_all(k['vtx/pack'], deps=k[pk])
        for send, pack in zip(m['vtx_send'], k['vtx/pack']):
            g.add_mpi_req(send, deps=[pack])

    def add_to_graph_post_recv(self, g, k, deps):
        # Unpack received vertex data (may be empty when unpack is a no-op)
        g.add_all(k['vtx/unpack'])

        # Merge per-peer vertex data (fine-grained: each merge waits
        # only on its own peer's unpack)
        for m in k['vtx/merge']:
            g.add(m, deps=deps(m, 'vtx/unpack'))

        # Fill artvisc_fpts once all merges are complete
        g.add_all(k['eles/avfill'], deps=k['vtx/merge'])

    def add_to_graph_ecav_post_recv(self, g, k, deps):
        self.add_to_graph_post_recv(g, k, deps)
