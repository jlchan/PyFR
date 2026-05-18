<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

#include <stdio.h>

<%pyfr:kernel name='gradhook' ndim='2'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'>
% if debug_grad_hook:
    if (_xi == 0 && _xj == 0) {
        fprintf(stderr, "[pyfr.gradhook] before gradu[0][0]=%.17g\\n",
                (double)gradu[0][0]);
    }
% endif
% for i, j in pyfr.ndrange(ndims, nvars):
    ## gradu[${i}][${j}] *= ${scale};
    gradu[${i}][${j}] += 1;
% endfor
% if debug_grad_hook:
    if (_xi == 0 && _xj == 0) {
        fprintf(stderr, "[pyfr.gradhook] after  gradu[0][0]=%.17g\\n",
                (double)gradu[0][0]);
    }
% endif
</%pyfr:kernel>
