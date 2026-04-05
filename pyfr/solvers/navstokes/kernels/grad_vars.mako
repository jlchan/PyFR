<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='grad_vars' ndim='2'
              u='in fpdtype_t[${str(nvars)}]'
              v='out fpdtype_t[${str(nvars)}]'>
    v[0] = u[0] + u[1];
    v[1] = u[1];
    v[2] = u[2];
    v[3] = u[3];
    v[4] = u[4];

</%pyfr:kernel>
