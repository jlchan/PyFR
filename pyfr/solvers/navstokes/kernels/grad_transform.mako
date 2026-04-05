<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='grad_transform' ndim='2'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'>
% for i in range(ndims):
% for j in range(nvars):
    gradu[${i}][${j}] = gradu[${i}][${j}];
% endfor
% endfor
</%pyfr:kernel>
