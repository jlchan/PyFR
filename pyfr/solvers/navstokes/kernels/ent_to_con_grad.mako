<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:kernel name='ent_to_con_grad' ndim='2'
              uin='in fpdtype_t[${str(nvars)}]'
              gradu='inout fpdtype_t[${str(ndims)}][${str(nvars)}]'>
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
% for i in range(ndims):
    fpdtype_t dw0_${i} = gradu[${i}][0];
    fpdtype_t dw1_${i} = gradu[${i}][1];
    fpdtype_t dw2_${i} = gradu[${i}][2];
    fpdtype_t dw3_${i} = gradu[${i}][3];

    gradu[${i}][0] = rho*dw0_${i} + rho_v1*dw1_${i}
                     + rho_v2*dw2_${i} + rho_e*dw3_${i};
    gradu[${i}][1] = rho_v1*dw0_${i} + (rho_v1*v1 + p)*dw1_${i}
                     + rho_v1*v2*dw2_${i} + rho_h_v1*dw3_${i};
    gradu[${i}][2] = rho_v2*dw0_${i} + rho_v1*v2*dw1_${i}
                     + (rho_v2*v2 + p)*dw2_${i}
                     + rho_h_v2*dw3_${i};
    gradu[${i}][3] = rho_e*dw0_${i} + rho_h_v1*dw1_${i}
                     + rho_h_v2*dw2_${i} + h44*dw3_${i};
% endfor
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
% for i in range(ndims):
    fpdtype_t dw0_${i} = gradu[${i}][0];
    fpdtype_t dw1_${i} = gradu[${i}][1];
    fpdtype_t dw2_${i} = gradu[${i}][2];
    fpdtype_t dw3_${i} = gradu[${i}][3];
    fpdtype_t dw4_${i} = gradu[${i}][4];

    gradu[${i}][0] = rho*dw0_${i} + rho_v1*dw1_${i}
                     + rho_v2*dw2_${i} + rho_v3*dw3_${i}
                     + rho_e*dw4_${i};
    gradu[${i}][1] = rho_v1*dw0_${i} + h22*dw1_${i}
                     + h23*dw2_${i} + h24*dw3_${i}
                     + rho_h_v1*dw4_${i};
    gradu[${i}][2] = rho_v2*dw0_${i} + h23*dw1_${i}
                     + h33*dw2_${i} + h34*dw3_${i}
                     + rho_h_v2*dw4_${i};
    gradu[${i}][3] = rho_v3*dw0_${i} + h24*dw1_${i}
                     + h34*dw2_${i} + h44*dw3_${i}
                     + rho_h_v3*dw4_${i};
    gradu[${i}][4] = rho_e*dw0_${i} + rho_h_v1*dw1_${i}
                     + rho_h_v2*dw2_${i} + rho_h_v3*dw3_${i}
                     + h55*dw4_${i};
% endfor
% endif
</%pyfr:kernel>
