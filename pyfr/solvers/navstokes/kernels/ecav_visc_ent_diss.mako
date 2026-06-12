<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.entgrad_common'/>
<%namespace file='pyfr.solvers.navstokes.kernels.entgrad_common' name='egrad'/>

// GLL-weighted volume integral of viscous entropy dissipation from physical ∇V.
% if ndims == 2:
<%pyfr:kernel name='ecav_visc_ent_diss' ndim='2'
              uin='in fpdtype_t[${str(nvars)}]'
              gradv='in fpdtype_t[${str(ndims)}][${str(nvars)}]'
              wts='in fpdtype_t'
              diss='out broadcast-col reduce(sum) fpdtype_t[1]'>
    ${egrad.ent_to_con_thermo()}

    fpdtype_t node_diss = 0.0;

    for (int i = 0; i < ${ndims}; i++)
    {
        fpdtype_t dw[${nvars}];
        fpdtype_t du0, du1, du2, du3;
    % for j in range(nvars):
        dw[${j}] = gradv[i][${j}];
    % endfor

        ${pyfr.expand('ent_to_con_jac2', 'dw',
                      'du0', 'du1', 'du2', 'du3')};

        node_diss += dw[0]*du0 + dw[1]*du1 + dw[2]*du2 + dw[3]*du3;
    }

    diss[0] = wts*node_diss;
</%pyfr:kernel>
% elif ndims == 3:
<%pyfr:kernel name='ecav_visc_ent_diss' ndim='2'
              uin='in fpdtype_t[${str(nvars)}]'
              gradv='in fpdtype_t[${str(ndims)}][${str(nvars)}]'
              wts='in fpdtype_t'
              diss='out broadcast-col reduce(sum) fpdtype_t[1]'>
    ${egrad.ent_to_con_thermo()}

    fpdtype_t node_diss = 0.0;

    for (int i = 0; i < ${ndims}; i++)
    {
        fpdtype_t dw[${nvars}];
        fpdtype_t du0, du1, du2, du3, du4;
    % for j in range(nvars):
        dw[${j}] = gradv[i][${j}];
    % endfor

        ${pyfr.expand('ent_to_con_jac3', 'dw',
                      'du0', 'du1', 'du2', 'du3', 'du4')};

        node_diss += dw[0]*du0 + dw[1]*du1 + dw[2]*du2
                   + dw[3]*du3 + dw[4]*du4;
    }

    diss[0] = wts*node_diss;
</%pyfr:kernel>
% endif
