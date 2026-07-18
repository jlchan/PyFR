<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// Fused per-element entropy residual, AV scaling, and vertex scatter.
<%pyfr:kernel name='ecav_resid_av_to_vtx' ndim='1'
              surf='in view(1) fpdtype_t'
              vol='in view(1) fpdtype_t'
              diss='in view(1) fpdtype_t'
              av_scaling='out view(1) fpdtype_t'
              vtx='out view(${str(nverts)}) reduce(max) fpdtype_t'>
    fpdtype_t resid = surf + vol;
    fpdtype_t a = resid < 0.0 ? -resid : 0.0;
    fpdtype_t b = diss;
    fpdtype_t scaling = a*b / (b*b + ${eps});

    av_scaling = scaling;
% for i in range(nverts):
    vtx[${i}] = scaling;
% endfor
</%pyfr:kernel>
