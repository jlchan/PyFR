<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.entropyvars'/>

// Fused surface entropy residual and face entropy conversion at flux points.
<%pyfr:kernel name='ecav_surface_ent_fpts' ndim='2'
              uin='in fpdtype_t[${str(nvars)}]'
              qnorm='in fpdtype_t[${str(ndims)}]'
              ent='out fpdtype_t[${str(nvars)}]'
              surf='out broadcast-col reduce(sum) fpdtype_t[1]'>
    ${pyfr.expand('con_to_ent_state', 'uin', 'ent')};

    fpdtype_t node_surf = 0.0;

% for j in range(ndims):
    node_surf += uin[${j + 1}] * qnorm[${j}];
% endfor

    surf[0] = node_surf;
</%pyfr:kernel>
