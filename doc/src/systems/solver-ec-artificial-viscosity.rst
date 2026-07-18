**********************************
[solver-ec-artificial-viscosity]
**********************************

Parameterises entropy-conservative artificial viscosity (ECAV) for
Navier–Stokes with

1. ``enabled`` --- activate ECAV:

    ``true`` | ``false``

Example:

.. code-block:: ini

    [solver-ec-artificial-viscosity]
    enabled = true

ECAV can be combined with ``shock-capturing = entropy-filter`` and
``[solver-entropy-filter]``. It is incompatible with
``shock-capturing = artificial-viscosity`` (legacy sensor AV) and with
flux anti-aliasing.

ECAV currently assumes Gauss-Legendre-Lobatto quadrature: solution
points must use ``soln-pts = gauss-legendre-lobatto`` in the relevant
``[solver-elements-*]`` sections (enforced at startup). The bundled
examples also use ``flux-pts = gauss-legendre-lobatto`` at interfaces.

Wall boundary conditions
------------------------

When ECAV is enabled, gradient recovery uses entropy variables at
interfaces. The existing ``no-slp-adia-wall`` boundary condition remains
the supported stationary, adiabatic no-slip wall and is entropy stable
in this mode; no separate wall type is required.

At boundary faces, PyFR:

1. Forms the LDG common conservative state with zero wall velocity via
   ``bc_ldg_state``, then converts it to entropy variables for the
   gradient-correction operators (``bcentconu``). For a stationary wall
   this yields zero middle entropy-variable components
   (``rho_p * u``, ``rho_p * v``, …).

2. Recovers physical entropy-variable gradients, then maps them back to
   conservative gradients (``ecav_visc_ent_diss_grad``) before face flux
   assembly.

3. Assembles the boundary viscous flux through ``ghost-imperm`` using
   the same inviscid reflected ghost state as without ECAV, together
   with a wall-side gradient that enforces zero normal conductive heat
   flux (adiabatic wall).

Use ``no-slp-adia-wall`` in ``[soln-bcs-*]`` as usual when running with
ECAV. For moving or isothermal walls, use ``no-slp-isot-wall`` instead;
that BC is not documented here as entropy stable under ECAV.

Used in the following Examples:

1. ``PyFR-Test-Cases/2d-gaussian-pulse/`` (combined with entropy filter)

2. ``PyFR-Test-Cases/2d-KHZ/``

3. ``PyFR-Test-Cases/2d-couette-flow/couette-flow-entropy.ini``
