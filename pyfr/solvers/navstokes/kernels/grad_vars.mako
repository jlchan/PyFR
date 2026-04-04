<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='grad_vars' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              v='out fpdtype_t[${str(nvars)}]'>
% for i in range(nvars):
    v[${i}] = u[${i}];
% endfor
</%pyfr:kernel>
