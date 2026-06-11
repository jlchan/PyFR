<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

// Broadcast the element-constant EC-AV coefficient to all face flux points
// so make_field_views / MPI can expose per-side av_scaling_l/r at interfaces.
<%pyfr:kernel name='ecav_scaling_fpts' ndim='1'
              av_scaling='in view(1) fpdtype_t'
              av_scaling_fpts='out fpdtype_t[${str(nfpts)}]'>
% for j in range(nfpts):
    av_scaling_fpts[${j}] = av_scaling;
% endfor
</%pyfr:kernel>
