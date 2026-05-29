<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='con_to_ent_state' params='uin, vout'>
% if ndims == 2:
    fpdtype_t rho = uin[0];
    fpdtype_t v1 = uin[1] / rho;
    fpdtype_t v2 = uin[2] / rho;
    fpdtype_t v2_mag = v1*v1 + v2*v2;
    fpdtype_t p = ${c['gamma'] - 1.0}*(uin[3] - 0.5*rho*v2_mag);
    fpdtype_t s = log(p) - ${c['gamma']}*log(rho);
    fpdtype_t rho_p = rho / p;

    vout[0] = ${1.0/(c['gamma'] - 1.0)}*(${c['gamma']} - s)
              - 0.5*rho_p*v2_mag;
    vout[1] = rho_p*v1;
    vout[2] = rho_p*v2;
    vout[3] = -rho_p;
% elif ndims == 3:
    fpdtype_t rho = uin[0];
    fpdtype_t v1 = uin[1] / rho;
    fpdtype_t v2 = uin[2] / rho;
    fpdtype_t v3 = uin[3] / rho;
    fpdtype_t v2_mag = v1*v1 + v2*v2 + v3*v3;
    fpdtype_t p = ${c['gamma'] - 1.0}*(uin[4] - 0.5*rho*v2_mag);
    fpdtype_t s = log(p) - ${c['gamma']}*log(rho);
    fpdtype_t rho_p = rho / p;

    vout[0] = ${1.0/(c['gamma'] - 1.0)}*(${c['gamma']} - s)
              - 0.5*rho_p*v2_mag;
    vout[1] = rho_p*v1;
    vout[2] = rho_p*v2;
    vout[3] = rho_p*v3;
    vout[4] = -rho_p;
% endif
</%pyfr:macro>
