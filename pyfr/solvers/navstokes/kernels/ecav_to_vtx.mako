<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// Scatter element-local _av_scaling[ele] to all vertices with reduce(max).
<%pyfr:kernel name='ecav_to_vtx' ndim='1'
              av_scaling='in view(1) fpdtype_t'
              vtx='out view(${str(nverts)}) reduce(max) fpdtype_t'>
% for i in range(nverts):
    vtx[${i}] = av_scaling;
% endfor
</%pyfr:kernel>
