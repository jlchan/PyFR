<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.entropyvars'/>

<%pyfr:kernel name='mpientconu' ndim='1'
              ulin='in view fpdtype_t[${str(nvars)}]'
              urin='in mpi fpdtype_t[${str(nvars)}]'
              ulout='out view fpdtype_t[${str(nvars)}]'>
    fpdtype_t vl[${nvars}], vr[${nvars}];

    ${pyfr.expand('con_to_ent_state', 'ulin', 'vl')};
    ${pyfr.expand('con_to_ent_state', 'urin', 'vr')};

% for i in range(nvars):
% if c['ldg-beta'] == -0.5:
    ulout[${i}] = vl[${i}];
% elif c['ldg-beta'] == 0.5:
    ulout[${i}] = vr[${i}];
% else:
    ulout[${i}] = vr[${i}]*${0.5 + c['ldg-beta']}
                + vl[${i}]*${0.5 - c['ldg-beta']};
% endif
% endfor
</%pyfr:kernel>
