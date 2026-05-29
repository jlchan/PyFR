<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='ent_to_con_grad' ndim='2'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'>
% for i, j in pyfr.ndrange(ndims, nvars):
    gradu[${i}][${j}] *= 0.333333333333333333;
% endfor
</%pyfr:kernel>
