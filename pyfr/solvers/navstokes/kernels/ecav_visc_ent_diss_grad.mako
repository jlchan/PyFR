<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.entgrad_common'/>
<%namespace file='pyfr.solvers.navstokes.kernels.entgrad_common' name='egrad'/>

// Calculate both the viscous entropy dissipation from physical ∇V, then 
// perform an in-place conversion of the same field to conservative ∇U.
<%pyfr:kernel name='ecav_visc_ent_diss_grad' ndim='2'
              uin='in fpdtype_t[${str(nvars)}]'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              wts='in fpdtype_t'
              diss='out broadcast-col reduce(sum) fpdtype_t[1]'>
    ${egrad.ent_to_con_thermo()}

    fpdtype_t node_diss = 0.0;

    for (int i = 0; i < ${ndims}; i++)
    {
        fpdtype_t dw[${nvars}];
    % for j in range(nvars):
        dw[${j}] = gradu[i][${j}];
    % endfor

    % if ndims == 2:
        fpdtype_t du0, du1, du2, du3;
        ${pyfr.expand('ent_to_con_jac2', 'dw',
                      'du0', 'du1', 'du2', 'du3')};

        node_diss += dw[0]*du0 + dw[1]*du1 + dw[2]*du2 + dw[3]*du3;

        gradu[i][0] = du0;
        gradu[i][1] = du1;
        gradu[i][2] = du2;
        gradu[i][3] = du3;
    % elif ndims == 3:
        fpdtype_t du0, du1, du2, du3, du4;
        ${pyfr.expand('ent_to_con_jac3', 'dw',
                      'du0', 'du1', 'du2', 'du3', 'du4')};

        node_diss += dw[0]*du0 + dw[1]*du1 + dw[2]*du2
                   + dw[3]*du3 + dw[4]*du4;

        gradu[i][0] = du0;
        gradu[i][1] = du1;
        gradu[i][2] = du2;
        gradu[i][3] = du3;
        gradu[i][4] = du4;
    % endif
    }

    diss[0] = wts*node_diss;
</%pyfr:kernel>
