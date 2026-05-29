<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='con_to_ent_state' params='uin, vout'>
% for i in range(nvars):
    vout[${i}] = 3.0*uin[${i}];
% endfor
</%pyfr:macro>
