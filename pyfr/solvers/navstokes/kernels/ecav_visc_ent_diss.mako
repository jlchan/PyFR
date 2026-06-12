<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>

<%pyfr:macro name='accum_visc_ent_diss_dir2' params='dw, node_diss, rho, rho_v1, rho_v2, rho_e, v1, v2, p, rho_h_v1, rho_h_v2, h44'>
    fpdtype_t du0 = rho*dw[0] + rho_v1*dw[1]
                  + rho_v2*dw[2] + rho_e*dw[3];
    fpdtype_t du1 = rho_v1*dw[0] + (rho_v1*v1 + p)*dw[1]
                  + rho_v1*v2*dw[2] + rho_h_v1*dw[3];
    fpdtype_t du2 = rho_v2*dw[0] + rho_v1*v2*dw[1]
                  + (rho_v2*v2 + p)*dw[2] + rho_h_v2*dw[3];
    fpdtype_t du3 = rho_e*dw[0] + rho_h_v1*dw[1]
                  + rho_h_v2*dw[2] + h44*dw[3];

    node_diss += dw[0]*du0 + dw[1]*du1 + dw[2]*du2 + dw[3]*du3;
</%pyfr:macro>

<%pyfr:macro name='accum_visc_ent_diss_dir3' params='dw, node_diss, rho, rho_v1, rho_v2, rho_v3, rho_e, v1, v2, v3, p, rho_h_v1, rho_h_v2, rho_h_v3, h22, h23, h24, h33, h34, h44, h55'>
    fpdtype_t du0 = rho*dw[0] + rho_v1*dw[1]
                  + rho_v2*dw[2] + rho_v3*dw[3]
                  + rho_e*dw[4];
    fpdtype_t du1 = rho_v1*dw[0] + h22*dw[1]
                  + h23*dw[2] + h24*dw[3]
                  + rho_h_v1*dw[4];
    fpdtype_t du2 = rho_v2*dw[0] + h23*dw[1]
                  + h33*dw[2] + h34*dw[3]
                  + rho_h_v2*dw[4];
    fpdtype_t du3 = rho_v3*dw[0] + h24*dw[1]
                  + h34*dw[2] + h44*dw[3]
                  + rho_h_v3*dw[4];
    fpdtype_t du4 = rho_e*dw[0] + rho_h_v1*dw[1]
                  + rho_h_v2*dw[2] + rho_h_v3*dw[3]
                  + h55*dw[4];

    node_diss += dw[0]*du0 + dw[1]*du1 + dw[2]*du2
               + dw[3]*du3 + dw[4]*du4;
</%pyfr:macro>

// GLL-weighted volume integral of viscous entropy dissipation from physical ∇V.
% if ndims == 2:
<%pyfr:kernel name='ecav_visc_ent_diss' ndim='2'
              uin='in fpdtype_t[${str(nvars)}]'
              gradv='in fpdtype_t[${str(ndims)}][${str(nvars)}]'
              wts='in fpdtype_t'
              diss='out broadcast-col reduce(sum) fpdtype_t[1]'>
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
    fpdtype_t h44 = rho*H*H - ${1.0/(c['gamma'] - 1.0)}*a2*p;

    fpdtype_t node_diss = 0.0;

    for (int i = 0; i < ${ndims}; i++)
    {
        fpdtype_t dw[${nvars}];
    % for j in range(nvars):
        dw[${j}] = gradv[i][${j}];
    % endfor

        ${pyfr.expand('accum_visc_ent_diss_dir2', 'dw', 'node_diss',
                      'rho', 'rho_v1', 'rho_v2', 'rho_e',
                      'v1', 'v2', 'p', 'rho_h_v1', 'rho_h_v2', 'h44')};
    }

    diss[0] = wts*node_diss;
</%pyfr:kernel>
% elif ndims == 3:
<%pyfr:kernel name='ecav_visc_ent_diss' ndim='2'
              uin='in fpdtype_t[${str(nvars)}]'
              gradv='in fpdtype_t[${str(ndims)}][${str(nvars)}]'
              wts='in fpdtype_t'
              diss='out broadcast-col reduce(sum) fpdtype_t[1]'>
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
    fpdtype_t h55 = rho*H*H - ${1.0/(c['gamma'] - 1.0)}*a2*p;
    fpdtype_t h22 = rho_v1*v1 + p;
    fpdtype_t h23 = rho_v1*v2;
    fpdtype_t h24 = rho_v1*v3;
    fpdtype_t h33 = rho_v2*v2 + p;
    fpdtype_t h34 = rho_v2*v3;
    fpdtype_t h44 = rho_v3*v3 + p;

    fpdtype_t node_diss = 0.0;

    for (int i = 0; i < ${ndims}; i++)
    {
        fpdtype_t dw[${nvars}];
    % for j in range(nvars):
        dw[${j}] = gradv[i][${j}];
    % endfor

        ${pyfr.expand('accum_visc_ent_diss_dir3', 'dw', 'node_diss',
                      'rho', 'rho_v1', 'rho_v2', 'rho_v3', 'rho_e',
                      'v1', 'v2', 'v3', 'p', 'rho_h_v1', 'rho_h_v2',
                      'rho_h_v3', 'h22', 'h23', 'h24', 'h33', 'h34',
                      'h44', 'h55')};
    }

    diss[0] = wts*node_diss;
</%pyfr:kernel>
% endif
