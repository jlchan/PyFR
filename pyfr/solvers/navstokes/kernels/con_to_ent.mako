<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='con_to_ent' ndim='2'
              uin='in fpdtype_t[${str(nvars)}]'
              vout='out fpdtype_t[${str(nvars)}]'>
% for i in range(nvars):
    vout[${i}] = 3.0*uin[${i}];
% endfor
</%pyfr:kernel>
