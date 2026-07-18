<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.entgrad_common'/>
<%namespace file='pyfr.solvers.navstokes.kernels.entgrad_common' name='egrad'/>
<%include file='pyfr.solvers.baseadvec.kernels.smats'/>
<%include file='pyfr.solvers.baseadvecdiff.kernels.transform_grad'/>

<% smats = 'smats_l' if 'linear' in ktype else 'smats' %>
<% rcpdjac = 'rcpdjac_l' if 'linear' in ktype else 'rcpdjac' %>

// Viscous entropy dissipation from physical ∇V, then in-place ∇V→∇U.
// gradu holds reference-space corrected gradients on entry.
<%pyfr:kernel name='ecav_visc_ent_diss_grad' ndim='2'
              uin='in fpdtype_t[${str(nvars)}]'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'
              wts='in fpdtype_t'
              smats='in fpdtype_t[${str(ndims)}][${str(ndims)}]'
              rcpdjac='in fpdtype_t'
              verts='in broadcast-col fpdtype_t[${str(nverts)}][${str(ndims)}]'
              upts='in broadcast-row fpdtype_t[${str(ndims)}]'
              diss='out broadcast-col reduce(sum) fpdtype_t[1]'>
% if 'linear' in ktype:
    // Compute the S matrices
    fpdtype_t ${smats}[${ndims}][${ndims}], djac;
    ${pyfr.expand('calc_smats_detj', 'verts', 'upts', smats, 'djac')};
    fpdtype_t ${rcpdjac} = 1 / djac;
% endif

    ${egrad.ent_to_con_thermo()}

    fpdtype_t physgrad[${ndims}][${nvars}];
% for i in range(ndims):
% for j in range(nvars):
    physgrad[${i}][${j}] = gradu[${i}][${j}];
% endfor
% endfor

    ${pyfr.expand('transform_grad', 'physgrad', smats, rcpdjac)};

    fpdtype_t node_diss = 0.0;

    for (int i = 0; i < ${ndims}; i++)
    {
        fpdtype_t dw[${nvars}];
    % for j in range(nvars):
        dw[${j}] = physgrad[i][${j}];
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
