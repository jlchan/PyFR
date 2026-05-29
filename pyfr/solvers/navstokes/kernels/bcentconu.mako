<%inherit file='base'/>
<%namespace module='pyfr.backends.base.makoutil' name='pyfr'/>
<%include file='pyfr.solvers.navstokes.kernels.bcs.${bctype}'/>
<%include file='pyfr.solvers.navstokes.kernels.entropyvars'/>

<%pyfr:kernel name='bcentconu' ndim='1'
              ulin='in view fpdtype_t[${str(nvars)}]'
              ulout='out view fpdtype_t[${str(nvars)}]'
              nlin='in fpdtype_t[${str(ndims)}]'>
    fpdtype_t mag_nl = sqrt(${pyfr.dot('nlin[{i}]', i=ndims)});
    fpdtype_t norm_nl[] = ${pyfr.array('(1 / mag_nl)*nlin[{i}]', i=ndims)};
    fpdtype_t ucomm[${nvars}];

    ${pyfr.expand('bc_ldg_state', 'ulin', 'norm_nl', 'ucomm')};
    ${pyfr.expand('con_to_ent_state', 'ucomm', 'ulout')};
</%pyfr:kernel>
