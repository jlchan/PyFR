<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// Per-element entropy residual: surface term plus volume term.
<%pyfr:kernel name='ecav_entropy_resid' ndim='1'
              surf='in view(1) fpdtype_t'
              vol='in view(1) fpdtype_t'
              resid='out view(1) fpdtype_t'>
    resid = surf + vol;
</%pyfr:kernel>
