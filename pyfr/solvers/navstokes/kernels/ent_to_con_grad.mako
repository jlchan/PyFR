<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.entgrad_common'/>
<%namespace file='pyfr.solvers.navstokes.kernels.entgrad_common' name='egrad'/>

<%pyfr:kernel name='ent_to_con_grad' ndim='2'
              uin='in fpdtype_t[${str(nvars)}]'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'>
    ${egrad.ent_to_con_thermo()}

    for (int i = 0; i < ${ndims}; i++)
    {
        fpdtype_t dw[${nvars}];
    % for j in range(nvars):
        dw[${j}] = gradu[i][${j}];
    % endfor

    % if ndims == 2:
        ${pyfr.expand('ent_to_con_jac2', 'dw',
                      'gradu[i][0]', 'gradu[i][1]',
                      'gradu[i][2]', 'gradu[i][3]')};
    % elif ndims == 3:
        ${pyfr.expand('ent_to_con_jac3', 'dw',
                      'gradu[i][0]', 'gradu[i][1]',
                      'gradu[i][2]', 'gradu[i][3]',
                      'gradu[i][4]')};
    % endif
    }
</%pyfr:kernel>
