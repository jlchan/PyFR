<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.euler.kernels.flux'/>

// Volume term of entropy residual: integral of sum_i (-dV/dx_i) . F_i at upts.
<%pyfr:kernel name='ecav_volume_integral' ndim='2'
              uin='in fpdtype_t[${str(nvars)}]'
              gradv='in fpdtype_t[${str(ndims)}][${str(nvars)}]'
              wts='in fpdtype_t'
              vol='out broadcast-col reduce(sum) fpdtype_t[1]'>
    fpdtype_t p;
    fpdtype_t v[${ndims}];
    fpdtype_t f[${ndims}][${nvars}];

    ${pyfr.expand('inviscid_flux', 'uin', 'f', 'p', 'v')};

    fpdtype_t node_vol = 0.0;

    for (int i = 0; i < ${ndims}; i++)
    {
    % for j in range(nvars):
        node_vol -= gradv[i][${j}] * f[i][${j}];
    % endfor
    }

    vol[0] = wts*node_vol;
</%pyfr:kernel>
