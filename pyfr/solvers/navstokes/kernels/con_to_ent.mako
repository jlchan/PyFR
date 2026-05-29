<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.entropyvars'/>

<%pyfr:kernel name='con_to_ent' ndim='2'
              uin='in fpdtype_t[${str(nvars)}]'
              vout='out fpdtype_t[${str(nvars)}]'>
    ${pyfr.expand('con_to_ent_state', 'uin', 'vout')};
</%pyfr:kernel>
