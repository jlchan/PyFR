<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>
<%include file='pyfr.solvers.baseadvec.kernels.smats'/>
<%include file='pyfr.solvers.baseadvecdiff.kernels.transform_grad'/>

<% smats = 'smats_l' if 'linear' in ktype else 'smats' %>
<% rcpdjac = 'rcpdjac_l' if 'linear' in ktype else 'rcpdjac' %>

// Volume term of entropy residual: integral of sum_i (-dV/dx_i) . F_i at upts.
// gradv holds reference-space M4·V; transform to physical space in registers.
<%pyfr:kernel name='ecav_volume_integral' ndim='2'
              uin='in fpdtype_t[${str(nvars)}]'
              gradv='in fpdtype_t[${str(ndims)}][${str(nvars)}]'
              wts='in fpdtype_t'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='in fpdtype_t'
              verts='in broadcast-col fpdtype_t[${str(nverts)}][${str(ndims)}]'
              upts='in broadcast-row fpdtype_t[${str(ndims)}]'
              vol='out broadcast-col reduce(sum) fpdtype_t[1]'>
% if 'linear' in ktype:
    // Compute the S matrices
    fpdtype_t ${smats}[${ndims}][${ndims}], djac;
    ${pyfr.expand('calc_smats_detj', 'verts', 'upts', smats, 'djac')};
    fpdtype_t ${rcpdjac} = 1 / djac;
% endif

    fpdtype_t physgrad[${ndims}][${nvars}];
% for i in range(ndims):
% for j in range(nvars):
    physgrad[${i}][${j}] = gradv[${i}][${j}];
% endfor
% endfor

    ${pyfr.expand('transform_grad', 'physgrad', smats, rcpdjac)};

    fpdtype_t p;
    fpdtype_t v[${ndims}];
    fpdtype_t f[${ndims}][${nvars}];

    ${pyfr.expand('inviscid_flux', 'uin', 'f', 'p', 'v')};

    fpdtype_t node_vol = 0.0;

    for (int i = 0; i < ${ndims}; i++)
    {
    % for j in range(nvars):
        node_vol -= physgrad[i][${j}] * f[i][${j}];
    % endfor
    }

    vol[0] = wts*node_vol;
</%pyfr:kernel>
