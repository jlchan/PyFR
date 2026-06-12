<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// Per-element AV coefficient: a*b / (b*b + eps), a = -min(0, resid), b = diss.
<%pyfr:kernel name='ecav_av_scaling' ndim='1'
              resid='in view(1) fpdtype_t'
              diss='in view(1) fpdtype_t'
              av_scaling='out view(1) fpdtype_t'>
    fpdtype_t a = resid < 0.0 ? -resid : 0.0;
    fpdtype_t b = diss;

    av_scaling = a*b / (b*b + ${eps});
</%pyfr:kernel>
