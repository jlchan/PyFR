<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%def name="ent_to_con_thermo()">
% if ndims == 2:
    fpdtype_t rho = uin[0];
    fpdtype_t rho_v1 = uin[1];
    fpdtype_t rho_v2 = uin[2];
    fpdtype_t rho_e = uin[3];
    fpdtype_t v1 = rho_v1 / rho;
    fpdtype_t v2 = rho_v2 / rho;
    fpdtype_t v2_mag = v1*v1 + v2*v2;
    fpdtype_t p = ${c['gamma'] - 1.0}*(rho_e - 0.5*rho*v2_mag);
    fpdtype_t a2 = ${c['gamma']}*p / rho;
    fpdtype_t H = ${1.0/(c['gamma'] - 1.0)}*a2 + 0.5*v2_mag;
    fpdtype_t rho_h_v1 = rho_v1*H;
    fpdtype_t rho_h_v2 = rho_v2*H;
    fpdtype_t h44 = rho*H*H
                    - ${1.0/(c['gamma'] - 1.0)}*a2*p;
% elif ndims == 3:
    fpdtype_t rho = uin[0];
    fpdtype_t rho_v1 = uin[1];
    fpdtype_t rho_v2 = uin[2];
    fpdtype_t rho_v3 = uin[3];
    fpdtype_t rho_e = uin[4];
    fpdtype_t v1 = rho_v1 / rho;
    fpdtype_t v2 = rho_v2 / rho;
    fpdtype_t v3 = rho_v3 / rho;
    fpdtype_t v2_mag = v1*v1 + v2*v2 + v3*v3;
    fpdtype_t p = ${c['gamma'] - 1.0}*(rho_e - 0.5*rho*v2_mag);
    fpdtype_t a2 = ${c['gamma']}*p / rho;
    fpdtype_t H = ${1.0/(c['gamma'] - 1.0)}*a2 + 0.5*v2_mag;
    fpdtype_t rho_h_v1 = rho_v1*H;
    fpdtype_t rho_h_v2 = rho_v2*H;
    fpdtype_t rho_h_v3 = rho_v3*H;
    fpdtype_t h55 = rho*H*H
                    - ${1.0/(c['gamma'] - 1.0)}*a2*p;
    fpdtype_t h22 = rho_v1*v1 + p;
    fpdtype_t h23 = rho_v1*v2;
    fpdtype_t h24 = rho_v1*v3;
    fpdtype_t h33 = rho_v2*v2 + p;
    fpdtype_t h34 = rho_v2*v3;
    fpdtype_t h44 = rho_v3*v3 + p;
% endif
</%def>

<%pyfr:macro name='ent_to_con_jac2' params='dw, o0, o1, o2, o3'>
    o0 = rho*dw[0] + rho_v1*dw[1]
       + rho_v2*dw[2] + rho_e*dw[3];
    o1 = rho_v1*dw[0] + (rho_v1*v1 + p)*dw[1]
       + rho_v1*v2*dw[2] + rho_h_v1*dw[3];
    o2 = rho_v2*dw[0] + rho_v1*v2*dw[1]
       + (rho_v2*v2 + p)*dw[2] + rho_h_v2*dw[3];
    o3 = rho_e*dw[0] + rho_h_v1*dw[1]
       + rho_h_v2*dw[2] + h44*dw[3];
</%pyfr:macro>

<%pyfr:macro name='ent_to_con_jac3' params='dw, o0, o1, o2, o3, o4'>
    o0 = rho*dw[0] + rho_v1*dw[1]
       + rho_v2*dw[2] + rho_v3*dw[3]
       + rho_e*dw[4];
    o1 = rho_v1*dw[0] + h22*dw[1]
       + h23*dw[2] + h24*dw[3]
       + rho_h_v1*dw[4];
    o2 = rho_v2*dw[0] + h23*dw[1]
       + h33*dw[2] + h34*dw[3]
       + rho_h_v2*dw[4];
    o3 = rho_v3*dw[0] + h24*dw[1]
       + h34*dw[2] + h44*dw[3]
       + rho_h_v3*dw[4];
    o4 = rho_e*dw[0] + rho_h_v1*dw[1]
       + rho_h_v2*dw[2] + rho_h_v3*dw[3]
       + h55*dw[4];
</%pyfr:macro>
