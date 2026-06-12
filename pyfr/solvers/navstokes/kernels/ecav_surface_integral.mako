<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// Surface term of entropy residual: integral of momentum dotted with weighted
// outward physical normals at flux points.
<%pyfr:kernel name='ecav_surface_integral' ndim='2'
              uin='in fpdtype_t[${str(nvars)}]'
              qnorm='in fpdtype_t[${str(ndims)}]'
              surf='out broadcast-col reduce(sum) fpdtype_t[1]'>
    fpdtype_t node_surf = 0.0;

% for j in range(ndims):
    node_surf += uin[${j + 1}] * qnorm[${j}];
% endfor

    surf[0] = node_surf;
</%pyfr:kernel>
