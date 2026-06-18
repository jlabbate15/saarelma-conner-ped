"""Non-dimensional Firedrake solver for the coupled three-equation
Saarelma-Connor pedestal model.

Implements the foundational non-dimensionalisation derived in
Appendix A.8 of docs/Labbate_APAM9301_Final_06152026.tex:

    L  = |x_inner|                length scale (m)
    n0 = ne_x0                    density scale (m^-3)
    T0 = T_e(0) + T_i(0)          energy scale (J)
    tau = 1 / (n0 * S0)           time scale (s),   S0 = S_i(0)

with derived scales
    [V]_0 = L / tau = L * n0 * S0          velocity (m/s)
    [D]_0 = L^2 / tau = L^2 * n0 * S0      diffusivity (m^2/s)
    [p]_0 = n0 * T0                        pressure (Pa).

Hatted variables:
    hat_x = x / L,                     hat_x in [-1, 0]
    hat_n_e = n_e / n0,                hat_n_e(0) = 1
    hat_n_FC = <n_FC> / n0,            hat_n_FC(0) = nFC_x0/n0
    hat_n_CX = <n_CX> / n0             hat_n_CX(0) = nCX_x0/n0
    hat_T = (T_e + T_i) / T0
    hat_S_i = S_i / S0,   hat_S_CX = S_CX / S0
    hat_V_FC = |V_FC| / [V]_0,   hat_V_CX = |V_CX| / [V]_0
    hat_g = <|grad r|^2>           (already dimensionless)

Dimensionless transport coefficients:
    hat_C_ETG = C_ETG / (L^2 * n0^2 * S0)
    hat_D_NEO = D_NEO / (L^2 * n0 * S0)
    hat_A_KBM = A_KBM / (L^2 * n0 * S0)
    hat_B_KBM = B_KBM * T0 / (L^3 * S0)

The dimensionless residuals (Eqs. (eq:weak-hat-A8),
(eq:weak-hat-nFC-A8), (eq:weak-hat-nCX-A8) in App. A.8):

    F1_hat: int_{-1}^{0} hat_g [ hat_C_ETG/hat_n_e * hat_n_e'
              + hat_A_KBM * hat_n_e'
              + hat_B_KBM * hat_T * (hat_n_e')^2
              + hat_B_KBM * hat_n_e * hat_dT_dxhat * hat_n_e'
              + hat_D_NEO * hat_n_e' ] v_e' dxhat
            - int_{-1}^{0} hat_n_e * hat_S_i * (hat_n_FC + hat_n_CX) v_e dxhat
            + delta_N * hat_g(-1) * hat_D|_{-1} * hat_dne_dx_inner * v_e(-1) = 0

    F2_hat: int_{-1}^{0} [ hat_V_FC * d/dxhat[ f_FC * hat_g * hat_n_FC ]
              - hat_n_e * (hat_S_i + hat_S_CX) * hat_n_FC ] v_F dxhat = 0

    F3_hat: int_{-1}^{0} [ d/dxhat[ hat_V_CX * f_CX * hat_g * hat_n_CX ]
              - hat_n_e * (hat_S_i * hat_n_CX - 0.5 * hat_S_CX * hat_n_FC)
              ] v_C dxhat = 0

Solver entry point: ``saarelma_connor_nondim.solve_coupled(...)`` --
identical signature to the parent class's :meth:`solve_coupled`.

This module subclasses :class:`src.solver.saarelma_connor` so all
equilibrium / kinetic / cross-section setup is inherited unchanged.
``src/solver.py`` is left untouched.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

try:
    from firedrake import (
        IntervalMesh, FunctionSpace, MixedFunctionSpace, Function,
        TestFunctions, Constant, DirichletBC,
        dx, ds, split, solve, SpatialCoordinate,
    )
    _FIREDRAKE_AVAILABLE = True
    _FIREDRAKE_IMPORT_ERR = None
except Exception as _firedrake_import_err:
    _FIREDRAKE_AVAILABLE = False
    _FIREDRAKE_IMPORT_ERR = _firedrake_import_err

from src.solver import saarelma_connor


# Conversion constant (eV -> J)
_EV2J = 1.60218e-19


class saarelma_connor_nondim(saarelma_connor):
    """Drop-in non-dimensional variant of :class:`saarelma_connor`.

    The constructor signature is identical to the parent class.  The only
    public method that is re-implemented is :meth:`solve_coupled`, which
    assembles and solves the dimensionless residuals derived in
    Appendix~A.8 of the paper rather than their SI counterparts.

    Reference scales are computed lazily inside :meth:`solve_coupled`
    (after ``setup_solver_grids`` and any inner-boundary search are done).
    They are stored as instance attributes:

        self._L_nd, self._n0_nd, self._T0_nd, self._S0_nd  : 4 free scales
        self._tau_nd, self._V0_nd, self._D0_nd             : derived scales

    After convergence the SI-unit profiles are recovered and stored on
    the same attributes (``self.x_sol``, ``self.ne_sol``, ``self.nFC_sol``,
    ``self.nCX_sol``) as the parent solver, so all downstream code
    (plotting, EPED feed, etc.) keeps working without modification.
    The non-dimensional profiles are also kept available as
    ``self.hat_x_sol``, ``self.hat_ne_sol``, ``self.hat_nFC_sol``,
    ``self.hat_nCX_sol`` for diagnostics.
    """

    # ------------------------------------------------------------------
    # Reference scales
    # ------------------------------------------------------------------

    def _set_nondim_scales(self, verbose=False):
        """Compute and cache the four free reference scales of
        Eq.~(eq:nondim-scales-A8) plus the derived scales.

        Convention (see App.~A.8):

            L  = |x_inner|              (m)
            n0 = ne_x0                  (m^-3)         -- separatrix value
            T0 = (T_e(0)+T_i(0)) * J/eV (J)            -- separatrix value
            S0 = S_i(0)                 (m^3/s)        -- separatrix value
            tau    = 1 / (n0 * S0)      (s)
            [V]_0  = L / tau            (m/s)
            [D]_0  = L^2 / tau          (m^2/s)

        """
        if not hasattr(self, "x_inner") or self.x_inner is None:
            raise RuntimeError(
                "x_inner is not set.  Call setup_solver_grids and/or "
                "find_inner_boundary before _set_nondim_scales."
            )
        if not hasattr(self, "ne_x0"):
            raise RuntimeError("ne_x0 is not set on this instance.")

        self._L_nd  = float(abs(self.x_inner))
        self._n0_nd = float(self.ne_x0)

        # Te + Ti at separatrix.  T_e_pres / T_i_pres are stored in eV
        # (cf. solver.py L196--L200), and the separatrix is index -1
        # because x_init = r_psi - r_psi[-1] (cf. solver.py L781).
        T_e0_eV = float(self.T_e_pres[-1])
        T_i0_eV = float(self.T_i_pres[-1])
        self._T0_nd = (T_e0_eV + T_i0_eV) * _EV2J  # in J

        # S_i at separatrix on the pressure grid (built in
        # setup_solver_grids -> S_i_pres on psi_N_pres, last index = sep.)
        S0 = float(self.S_i_pres[-1])
        if not np.isfinite(S0) or S0 <= 0.0:
            raise RuntimeError(
                f"S_i(0) = {S0!r} m^3/s is not a usable separatrix "
                "ionisation rate; cannot non-dimensionalise."
            )
        self._S0_nd = S0

        # Derived scales
        self._tau_nd = 1.0 / (self._n0_nd * self._S0_nd)
        self._V0_nd  = self._L_nd / self._tau_nd
        self._D0_nd  = (self._L_nd ** 2) / self._tau_nd

        # Pre-compute scales that show up repeatedly inside the residuals
        # (purely for readability of the rest of the class):
        #     ETG_scale = L^2 * n0^2 * S0   (= n0 * [D]_0)
        #     KBMB_scale = L^3 * S0 / T0    (= [B_KBM])
        self._ETG_scale_nd  = (self._L_nd ** 2) * (self._n0_nd ** 2) * self._S0_nd
        self._KBMB_scale_nd = (self._L_nd ** 3) * self._S0_nd / self._T0_nd

        if verbose:
            print(
                "[nondim] reference scales:\n"
                f"  L  = {self._L_nd:.4e} m\n"
                f"  n0 = {self._n0_nd:.4e} m^-3\n"
                f"  T0 = {self._T0_nd:.4e} J  ({(T_e0_eV + T_i0_eV):.3f} eV)\n"
                f"  S0 = {self._S0_nd:.4e} m^3/s\n"
                f"  tau    = {self._tau_nd:.4e} s\n"
                f"  [V]_0  = {self._V0_nd:.4e} m/s\n"
                f"  [D]_0  = {self._D0_nd:.4e} m^2/s"
            )

    # ------------------------------------------------------------------
    # SI <-> non-dim helpers
    # ------------------------------------------------------------------

    def _to_hat_x(self, x_si):
        """Map an SI x array (m) onto hat_x = x / L."""
        return np.asarray(x_si) / self._L_nd

    def _from_hat_x(self, hat_x):
        """Map hat_x back to SI x = L * hat_x."""
        return np.asarray(hat_x) * self._L_nd

    def _interp_si_to_hat(self, hat_x_dofs, si_array_on_xinit):
        """Evaluate an SI quantity defined on ``self.x_init`` (sorted)
        at the dimensionless mesh DOFs ``hat_x_dofs``.

        Returns the SI value at L * hat_x_dofs (i.e. *no* rescaling).
        Callers rescale to hat-units as needed.

        This basically just interpolates the SI array to the x_dofs grid.
        Because x_dofs is the same size as hat_x_dofs, once you are on x_dofs you don't need to interpolate again.
        """
        x_si = self._from_hat_x(hat_x_dofs)
        return np.interp(x_si, self.x_init, np.asarray(si_array_on_xinit)) # interpolates x_init to x_dofs grid

    # ------------------------------------------------------------------
    # Mesh and frozen non-dim coefficient Functions
    # ------------------------------------------------------------------

    def _ensure_firedrake_discretization_nondim(self, mesh_n, fe_degree, force=False):
        """Build (or reuse) the dimensionless mesh on hat_x in [-1, 0]
        and the equilibrium-only non-dim coefficient Functions.

        Cache keys live alongside the parent's ``self._fd_cache`` keys
        but use a ``_nd`` suffix to avoid colliding with any cached
        SI-version state.

        Returns
        -------
        mesh, V, W, hat_x_dofs, hat_g_fd, hat_Si_fd, hat_Scx_fd, hat_Vcx_fd
        """
        if not _FIREDRAKE_AVAILABLE:
            raise ImportError(
                "Firedrake is not available in this environment.  "
                "Install Firedrake (https://www.firedrakeproject.org/) "
                "to use this solver.  Original import error:\n"
                f"  {_FIREDRAKE_IMPORT_ERR}"
            )

        mesh_key = ("nd", int(mesh_n), int(fe_degree))
        if (not force
                and self._fd_cache.get("mesh_key_nd") == mesh_key
                and "mesh_nd" in self._fd_cache):
            return (
                self._fd_cache["mesh_nd"],
                self._fd_cache["V_nd"],
                self._fd_cache["W_nd"],
                self._fd_cache["hat_x_dofs"],
                self._fd_cache["hat_g_fd"],
                self._fd_cache["hat_Si_fd"],
                self._fd_cache["hat_Scx_fd"],
                self._fd_cache["hat_Vcx_fd"],
            )

        mesh = IntervalMesh(int(mesh_n), -1.0, 0.0)
        V = FunctionSpace(mesh, "CG", int(fe_degree))
        W = MixedFunctionSpace([V, V, V])

        x_coord_func = Function(V).interpolate(SpatialCoordinate(mesh)[0])
        hat_x_dofs = x_coord_func.dat.data.copy()

        def _make_func(arr, name=""):
            f = Function(V, name=name)
            f.dat.data[:] = arr
            return f

        # hat_g(hat_x) = <|grad r|^2>(L*hat_x)  -- already dimensionless
        hat_g_arr = self._interp_si_to_hat(hat_x_dofs, self.gradr2_fsa) # move gradr2_fsa to x_dofs grid
        hat_g_fd = _make_func(hat_g_arr, "hat_g")

        # hat_S_i(hat_x), hat_S_CX(hat_x): rate coefficients / S0
        hat_Si_arr  = self._interp_si_to_hat(hat_x_dofs, self.S_i_pres)  / self._S0_nd
        hat_Scx_arr = self._interp_si_to_hat(hat_x_dofs, self.S_cx_pres) / self._S0_nd
        hat_Si_fd  = _make_func(hat_Si_arr,  "hat_S_i")
        hat_Scx_fd = _make_func(hat_Scx_arr, "hat_S_cx")

        # hat_V_CX(hat_x) = |V_CX|(L*hat_x) / [V]_0
        hat_Vcx_arr = self._interp_si_to_hat(hat_x_dofs, np.abs(self.V_cx_pres)) / self._V0_nd
        hat_Vcx_fd  = _make_func(hat_Vcx_arr, "hat_V_cx")

        # Drop any mixed solution that was cached against a previous mesh.
        self._fd_cache.pop("u_nd", None)
        self._fd_cache.pop("u_prev_nd", None)
        self._fd_cache.update({
            "mesh_key_nd": mesh_key,
            "mesh_nd": mesh,
            "V_nd": V,
            "W_nd": W,
            "hat_x_dofs": hat_x_dofs,
            "hat_g_fd": hat_g_fd,
            "hat_Si_fd": hat_Si_fd,
            "hat_Scx_fd": hat_Scx_fd,
            "hat_Vcx_fd": hat_Vcx_fd,
        })
        return mesh, V, W, hat_x_dofs, hat_g_fd, hat_Si_fd, hat_Scx_fd, hat_Vcx_fd

    def _get_or_create_mixed_solution_nondim(self, W, force=False):
        """Return cached ``(hat_u, hat_u_prev)`` or allocate them."""
        if (not force
                and self._fd_cache.get("W_nd") is W
                and "u_nd" in self._fd_cache
                and "u_prev_nd" in self._fd_cache):
            return self._fd_cache["u_nd"], self._fd_cache["u_prev_nd"]

        u = Function(W, name="hat_u")
        u_prev = Function(W, name="hat_u_prev")
        self._fd_cache["W_nd"] = W
        self._fd_cache["u_nd"] = u
        self._fd_cache["u_prev_nd"] = u_prev
        return u, u_prev

    # ------------------------------------------------------------------
    # Pressure / KBM coefficients in non-dim units
    # ------------------------------------------------------------------

    def calc_pressure_quantities_nondim(self, hat_n_e, average_alpha_pedestal=True):
        """Non-dim version of :meth:`saarelma_connor.calc_pressure_quantities`.

        Computes the pedestal-averaged Connor-Hastie alpha in physical
        (SI) units (because alpha is a physical quantity defined by an
        SI integral) and then rescales the resulting transport
        contributions A_KBM, B_KBM, D_KBM into their dimensionless
        counterparts hat_A_KBM, hat_B_KBM, hat_D_KBM.

        Parameters
        ----------
        hat_n_e : array_like, shape (n_dofs,)
            Dimensionless electron density at the mesh DOFs (DOF order,
            not necessarily monotonic in hat_x).
        average_alpha_pedestal : bool, default True
            If True, use the pedestal-averaged alpha (preferred -- this
            matches solver.py behaviour).  Otherwise use the local alpha.

        Sets
        ----
        self._hat_A_KBM, self._hat_B_KBM, self._hat_D_KBM : ndarray
            Dimensionless KBM coefficients in DOF order.
        self.alpha_bar_ped : float
            Pedestal-averaged alpha (same definition as the parent).
        """
        hat_x_dofs = self._fd_cache["hat_x_dofs"]
        sort_idx = np.argsort(hat_x_dofs)
        unsort_idx = np.argsort(sort_idx)
        hat_x = hat_x_dofs[sort_idx]
        hat_n_e_sorted = np.asarray(hat_n_e)[sort_idx]

        # Reconstruct SI quantities on the sorted hat-x grid for the
        # alpha calculation.
        x_si = self._from_hat_x(hat_x)              # m
        n_e_sorted = hat_n_e_sorted * self._n0_nd   # m^-3
        T_e_eV = np.interp(x_si, self.x_init, self.T_e_pres)        # eV
        T_i_eV = np.interp(x_si, self.x_init, self.T_i_pres)        # eV
        self.T_e_xdofs = T_e_eV                                     # diagnostic
        G_KBM_grid = self.C_KBM * (self.c_s * self.rho_s ** 2) / self.a
        G_KBM = np.interp(x_si, self.x_init, G_KBM_grid)            # m^2/s
        alpha_nodp = np.interp(x_si, self.x_init, self._alpha_nodp_xinit)

        # Connor-Hastie alpha = alpha_nodp * dp/dx (SI)
        _pres = n_e_sorted * (T_e_eV + T_i_eV) * _EV2J # Pa, neglecting neutral pressures and assuming quasi-neutral plasma
        dpdx = np.gradient(_pres, x_si)                # Pa/m
        _alpha = alpha_nodp * dpdx                     # dimensionless

        if average_alpha_pedestal:
            alpha_bar = float(np.mean(_alpha))
            self.alpha_bar_ped = alpha_bar
            gate = alpha_bar > self.alpha_crit
            D_KBM_si = np.where(gate, (alpha_bar - self.alpha_crit) * G_KBM, 0.0)
            A_KBM_si = np.where(gate, -G_KBM * self.alpha_crit, 0.0)
            B_KBM_si = np.where(gate, G_KBM * alpha_nodp, 0.0)
        else:
            gate = _alpha > self.alpha_crit
            D_KBM_si = np.where(gate, (_alpha - self.alpha_crit) * G_KBM, 0.0)
            A_KBM_si = np.where(gate, -G_KBM * self.alpha_crit, 0.0)
            B_KBM_si = np.where(gate, G_KBM * alpha_nodp, 0.0)

        # Rescale to hat units (App. A.8 Eqs. (eq:hat-A-KBM), (eq:hat-B-KBM)):
        #   hat_A = A / [D]_0
        #   hat_B = B * T0 / (L^3 * S0) = B / KBMB_scale
        #   hat_D_KBM = D_KBM / [D]_0
        hat_A_KBM_sorted = A_KBM_si / self._D0_nd
        hat_B_KBM_sorted = B_KBM_si / self._KBMB_scale_nd
        hat_D_KBM_sorted = D_KBM_si / self._D0_nd

        self._hat_A_KBM = hat_A_KBM_sorted[unsort_idx]
        self._hat_B_KBM = hat_B_KBM_sorted[unsort_idx]
        self._hat_D_KBM = hat_D_KBM_sorted[unsort_idx]

        # Diagnostics in DOF order (mirrors the parent class)
        self.G_KBM = G_KBM[unsort_idx]
        self.alpha_nodp = alpha_nodp[unsort_idx]
        self.pres = _pres[unsort_idx]

    # ------------------------------------------------------------------
    # Weak forms
    # ------------------------------------------------------------------

    def _build_f1_weak_form_nondim(
            self, hat_ne, hat_g_fd,
            hat_C_ETG_fd, hat_D_NEO_fd,
            hat_A_KBM_fd, hat_B_KBM_fd,
            hat_T_fd, hat_dT_dx_fd,
            hat_Si_fd, hat_nFC, hat_nCX,
            v_e, ne_inner_bc, hat_dne_dx_inner_c):
        """Dimensionless n_e weak form -- Eq. (eq:weak-hat-A8) of App. A.8.

        Integrals are over hat_x in [-1, 0] (the Firedrake mesh).  The
        Neumann boundary contribution at hat_x = -1 (boundary id 1) is
        added when ne_inner_bc == "neumann", evaluated using the
        Dirichlet-type expansion of hat_D consistent with the parent
        solver.
        """
        ne_dx = hat_ne.dx(0)

        flux_a33 = (
            (hat_C_ETG_fd / hat_ne) * ne_dx
            + hat_A_KBM_fd * ne_dx
            + hat_B_KBM_fd * hat_T_fd * ne_dx * ne_dx
            + hat_B_KBM_fd * hat_ne * hat_dT_dx_fd * ne_dx
            + hat_D_NEO_fd * ne_dx
        )

        F1 = (
            hat_g_fd * flux_a33 * v_e.dx(0)
            - hat_ne * hat_Si_fd * (hat_nFC + hat_nCX) * v_e
        ) * dx

        if ne_inner_bc == "neumann":
            # Boundary id 1 = left endpoint = inner boundary = hat_x = -1.
            hat_D_bc = (
                hat_C_ETG_fd / hat_ne
                + hat_A_KBM_fd
                + hat_B_KBM_fd * hat_T_fd * hat_dne_dx_inner_c
                + hat_B_KBM_fd * hat_ne * hat_dT_dx_fd
                + hat_D_NEO_fd
            )
            F1 = F1 + hat_g_fd * hat_D_bc * hat_dne_dx_inner_c * v_e * ds(1)

        return F1

    def _build_f2_weak_form_nondim(
            self, hat_ne, hat_nFC, hat_g_fd, hat_Si_fd, hat_Scx_fd,
            hat_VFC_const, fFC_const, v_F):
        """Dimensionless FC-neutral weak form -- Eq. (eq:weak-hat-nFC-A8)."""
        flux_FC = hat_VFC_const * (fFC_const * hat_g_fd * hat_nFC).dx(0)
        return (flux_FC - hat_ne * (hat_Si_fd + hat_Scx_fd) * hat_nFC) * v_F * dx

    def _build_f3_weak_form_nondim(
            self, hat_ne, hat_nFC, hat_nCX, hat_g_fd,
            hat_Si_fd, hat_Scx_fd, hat_Vcx_fd, fCX_const, half, v_C):
        """Dimensionless CX-neutral weak form -- Eq. (eq:weak-hat-nCX-A8)."""
        flux_CX = (hat_Vcx_fd * fCX_const * hat_g_fd * hat_nCX).dx(0)
        rhs = hat_ne * (hat_Si_fd * hat_nCX - half * hat_Scx_fd * hat_nFC)
        return (flux_CX - rhs) * v_C * dx

    # ------------------------------------------------------------------
    # Driver
    # ------------------------------------------------------------------

    def solve_coupled_nondim(self,
                      x_res=20,
                      fe_degree=2,
                      ne_inner_bc="neumann",
                      bc_origin="p-file",
                      ne_inner=None,
                      dne_dx_inner=None,
                      initial_guess="linear",
                      tanh_width=None,
                      tanh_center=None,
                      linear_solver="lu",
                      ksp_rtol=1e-8,
                      ksp_max_it=200,
                      reuse_setup=True,
                      verbose=None):
        """Non-dimensional Firedrake solver for the coupled three-equation
        Saarelma--Connor neutral-transport pedestal model.

        Equivalent to :meth:`saarelma_connor.solve_coupled` but assembles
        and solves the dimensionless residuals derived in App.~A.8
        (foundational scaling, no free $\\Lambda$ prefactor).  Inputs and
        outputs are in SI units; the rescaling is invisible to the
        caller.

        Boundary conditions
        ===================
            hat_n_e (hat_x = 0)        = 1
            hat_n_FC(hat_x = 0)        = nFC_x0 / n0
            hat_n_CX(hat_x = 0)        = nCX_x0 / n0
            hat_n_e (hat_x = -1)       = ne_inner / n0   (Dirichlet option)
         OR hat_n_e'(hat_x = -1)       = L * dne_dx_inner / n0   (Neumann option)

        Parameters
        ----------
        See :meth:`saarelma_connor.solve_coupled`.  All physical inputs
        are in SI units.

        Sets
        ----
        SI-unit attributes (parent-class compatibility):
            self.x_sol, self.ne_sol, self.nFC_sol, self.nCX_sol
            self.u_fd  (mixed Firedrake Function -- now stores hat values)
            self.W_fd, self.V_fd

        Non-dim attributes (diagnostics):
            self.hat_x_sol, self.hat_ne_sol, self.hat_nFC_sol, self.hat_nCX_sol
            self._L_nd, self._n0_nd, self._T0_nd, self._S0_nd
            self._tau_nd, self._V0_nd, self._D0_nd
        """
        if not _FIREDRAKE_AVAILABLE:
            raise ImportError(
                "Firedrake is not available in this environment.  "
                "Install Firedrake to use this solver.  Original import "
                f"error:\n  {_FIREDRAKE_IMPORT_ERR}"
            )

        v = self.verbose if verbose is None else bool(verbose)
        force_setup = not reuse_setup

        # Equilibrium-only quantities (FSA |grad r|^2, S_i_pres, etc.) and
        # ETG coefficient -- exactly the same setup as the parent class.
        self._ensure_firedrake_coefficient_grids(x_res, force=force_setup)
        self.construct_C_ETG()

        # Inner boundary location (in SI x); same logic as solver.py.
        if self.psi_N_inner_boundary is None:
            self.find_inner_boundary()
        else:
            self.x_inner = np.interp(self.psi_N_inner_boundary, self.psi_N_pres, self.x_init)

        # Read off ne(x_inner), dne/dx(x_inner) in SI -- same conventions
        # as solver.py.solve_coupled.
        if bc_origin == "p-file":
            ne_inner_val = float(np.interp(self.x_inner, self.x_init, self.n_e_pres))
            self.ne_inner = ne_inner_val
            dne_dx_pres = np.gradient(self.n_e_pres, self.x_init)
            dne_dx_inner_val = float(np.interp(self.x_inner, self.x_init, dne_dx_pres))
        elif bc_origin == "user":
            ne_inner_val = float(ne_inner)
            dne_dx_inner_val = float(dne_dx_inner)
        elif bc_origin == "p-file user combo":
            if ne_inner_bc == "neumann":
                dne_dx_pres = np.gradient(self.n_e_pres, self.x_init)
                dne_dx_inner_val = float(np.interp(self.x_inner, self.x_init, dne_dx_pres))
                ne_inner_val = float(ne_inner)
                self.ne_inner = ne_inner_val
            elif ne_inner_bc == "dirichlet":
                ne_inner_val = float(ne_inner)
                self.ne_inner = ne_inner_val
                dne_dx_inner_val = float(dne_dx_inner)
        else:
            raise ValueError(
                f"bc_origin must be 'p-file' or 'user' or 'p-file user combo', got {bc_origin!r}."
            )

        ne_inner_bc = str(ne_inner_bc).lower()
        if ne_inner_bc not in ("dirichlet", "neumann"):
            raise ValueError(
                f"ne_inner_bc must be 'dirichlet' or 'neumann', got {ne_inner_bc!r}."
            )

        if float(self.x_inner) >= 0.0:
            raise ValueError(
                f"x_inner = {self.x_inner} must be strictly less than 0 (separatrix)."
            )

        # Compute reference scales now that x_inner / ne_x0 / etc. are known.
        self._set_nondim_scales(verbose=v)

        # Mesh + frozen non-dim coefficient Functions on hat_x in [-1, 0].
        mesh, V, W, hat_x_dofs, hat_g_fd, hat_Si_fd, hat_Scx_fd, hat_Vcx_fd = (
            self._ensure_firedrake_discretization_nondim(
                x_res, fe_degree, force=force_setup,
            )
        )

        # Constants (App. A.8 Eq. (eq:hat-V-A8))
        hat_VFC_const = Constant(abs(self.V_FC) / self._V0_nd)
        fFC_const     = Constant(self.fFC)
        fCX_const     = Constant(self.fCX)
        half          = Constant(0.5)

        u, u_prev = self._get_or_create_mixed_solution_nondim(W, force=force_setup)

        # ------------------------------------------------------------------
        # Initial guess (in SI, then rescaled to hat-units).
        # ------------------------------------------------------------------
        x_dofs_si = self._from_hat_x(hat_x_dofs)  # m
        x_left_si = -self._L_nd
        x_right_si = 0.0

        if initial_guess == "linear":
            xi = (x_dofs_si - x_left_si) / (x_right_si - x_left_si)
            if ne_inner_bc == "dirichlet": # ne_inner_val is known
                ne_init  = ne_inner_val + (self.ne_x0 - ne_inner_val) * xi
            elif ne_inner_bc == "neumann": # ne_inner_val is unknown
                if dne_dx_inner_val >= 0:
                    raise ValueError(f"dne/dx(x_inner) = {dne_dx_inner_val:.3e} m^-4 must be negative for Neumann boundary condition.")
                ne_inner_val = dne_dx_inner_val*self.x_inner + self.ne_x0
                ne_init  = ne_inner_val + (self.ne_x0 - ne_inner_val) * xi
            ne_init  = ne_inner_val + (self.ne_x0 - ne_inner_val) * xi
            nFC_init = self.nFC_x0 * xi
            nCX_init = self.nCX_x0 * xi

        elif initial_guess == "pfile":
            ne_init = np.interp(x_dofs_si, self.x_init, self.n_e_pres)

            order_desc = np.argsort(x_dofs_si)[::-1]
            x_desc = x_dofs_si[order_desc]
            ne_desc = np.interp(x_desc, self.x_init, self.n_e_pres)
            Si_desc = np.interp(x_desc, self.x_init, self.S_i_pres)
            Scx_desc = np.interp(x_desc, self.x_init, self.S_cx_pres)
            integrand_init = (
                ne_desc * (Si_desc + Scx_desc) / (self.fFC * abs(self.V_FC))
            )
            cumint_desc = cumulative_trapezoid(integrand_init, x_desc, initial=0.0)
            nFC_on_desc = self.nFC_x0 * np.exp(cumint_desc)
            nFC_init = np.empty_like(x_dofs_si)
            nFC_init[order_desc] = nFC_on_desc

            ratio_CX = self.nCX_x0 / self.nFC_x0 if self.nFC_x0 > 0 else 0.0
            nCX_init = ratio_CX * nFC_init

        elif initial_guess == "tanh":
            width  = float(tanh_width)  if tanh_width  is not None else 0.1 * abs(x_left_si)
            if width <= 0:
                raise ValueError(f"tanh_width must be positive, got {width}.")
            if ne_inner_bc == "dirichlet": # ne_inner_val is known
                center = float(tanh_center) if tanh_center is not None else -width
                s_ne   = 0.5 * (1.0 - np.tanh((x_dofs_si - center) / (0.5 * width)))
                s_neut = 1.0 - s_ne
                ne_init  = self.ne_x0 + (ne_inner_val - self.ne_x0) * s_ne
                nFC_init = self.nFC_x0 * s_neut
                nCX_init = self.nCX_x0 * s_neut
            elif ne_inner_bc == "neumann":
                if dne_dx_inner_val >= 0:
                    raise ValueError(
                        f"dne/dx(x_inner) = {dne_dx_inner_val:.3e} m^-4 must be negative for Neumann boundary condition."
                    )
                # default: x_inner sits one width inside the foot of the tanh
                # so sech^2 at x_inner is well-conditioned (~0.07)
                center = float(tanh_center) if tanh_center is not None else self.x_inner + width
                arg = (self.x_inner - center) / (0.5 * width)
                sech2 = 1.0 / np.cosh(arg) ** 2
                if sech2 < 1e-6:
                    raise ValueError(
                        "tanh transition is too far from x_inner to match the requested "
                        "slope; reduce |center - x_inner| or increase width."
                    )
                ne_inner_val = self.ne_x0 - dne_dx_inner_val * width / sech2
                self.ne_inner = ne_inner_val

                s_ne   = 0.5 * (1.0 - np.tanh((x_dofs_si - center) / (0.5 * width)))
                s_neut = 1.0 - s_ne
                ne_init  = self.ne_x0 + (ne_inner_val - self.ne_x0) * s_ne
                nFC_init = self.nFC_x0 * s_neut
                nCX_init = self.nCX_x0 * s_neut

        else:
            raise ValueError(
                f"Unknown initial_guess={initial_guess!r}; expected "
                "'linear', 'pfile', or 'tanh'."
            )

        u.subfunctions[0].dat.data[:] = ne_init  / self._n0_nd
        u.subfunctions[1].dat.data[:] = nFC_init / self._n0_nd
        u.subfunctions[2].dat.data[:] = nCX_init / self._n0_nd
        u_prev.assign(u)

        # Rescale BC values into hat-units (App. A.8 Eq. (eq:hat-flux-A8) etc.) using initial guesses to accommodate Dirichlet or Neumann boundary condition choice
        hat_ne_x0      = 1.0
        hat_ne_inner   = ne_init[0]   / self._n0_nd
        hat_nFC_x0     = self.nFC_x0    / self._n0_nd
        hat_nCX_x0     = self.nCX_x0    / self._n0_nd
        hat_dne_dx_inner = self._L_nd * np.gradient(ne_init, x_dofs_si)[0] / self._n0_nd

        if v:
            print(f"[nondim] x_inner          = {self.x_inner:.4e} m")
            print(f"[nondim] ne_inner_bc      = {ne_inner_bc!r}")
            print(f"[nondim] ne(x_inner)      = {ne_inner_val:.3e} m^-3 ({bc_origin})")
            print(f"[nondim] dne/dx(x_inner)  = {dne_dx_inner_val:.3e} m^-4 ({bc_origin})")
            print(f"[nondim] hat_ne(0)        = {hat_ne_x0}")
            print(f"[nondim] hat_ne(-1)       = {hat_ne_inner:.3e}")
            print(f"[nondim] hat_nFC(0)       = {hat_nFC_x0:.3e}")
            print(f"[nondim] hat_nCX(0)       = {hat_nCX_x0:.3e}")
            print(f"[nondim] hat_dne/dxhat(-1)= {hat_dne_dx_inner:.3e}")

        # ------------------------------------------------------------------
        # KBM (frozen at the initial guess for the SNES solve, exactly as
        # in solver.py.solve_coupled).
        # ------------------------------------------------------------------
        self.calc_pressure_quantities_nondim(
            u.subfunctions[0].dat.data,
            average_alpha_pedestal=True,
        )

        # Frozen ETG and NEO contributions in hat-units.
        hat_C_ETG_arr = (
            self._interp_si_to_hat(hat_x_dofs, self.C_ETG) / self._ETG_scale_nd
        )
        hat_D_NEO_arr = (
            self._interp_si_to_hat(hat_x_dofs, self.D_NEO) / self._D0_nd
        )

        f = Function(V, name="hat_C_ETG"); f.dat.data[:] = hat_C_ETG_arr
        self._fd_cache["hat_C_ETG_fd"] = f
        f = Function(V, name="hat_D_NEO"); f.dat.data[:] = hat_D_NEO_arr
        self._fd_cache["hat_D_NEO_fd"] = f
        f = Function(V, name="hat_A_KBM"); f.dat.data[:] = self._hat_A_KBM
        self._fd_cache["hat_A_KBM_fd"] = f
        f = Function(V, name="hat_B_KBM"); f.dat.data[:] = self._hat_B_KBM
        self._fd_cache["hat_B_KBM_fd"] = f
        f = Function(V, name="hat_D_KBM"); f.dat.data[:] = self._hat_D_KBM
        self._fd_cache["hat_D_KBM_fd"] = f

        # hat_T(hat_x) and d hat_T / d hat_x at mesh DOFs.
        T_e_J = self.T_e_pres * _EV2J # J
        T_i_J = self.T_i_pres * _EV2J # J
        hat_T_arr = (
            self._interp_si_to_hat(hat_x_dofs, T_e_J + T_i_J) / self._T0_nd
        )
        # d hat_T / d hat_x = (L / T0) * d (T_e + T_i) / dx
        dT_dx_J_arr = np.gradient(T_e_J + T_i_J, self.x_init)
        hat_dT_dx_arr = (
            self._interp_si_to_hat(hat_x_dofs, dT_dx_J_arr)
            * self._L_nd / self._T0_nd
        )

        f = Function(V, name="hat_T");        f.dat.data[:] = hat_T_arr
        self._fd_cache["hat_T_fd"] = f
        f = Function(V, name="hat_dT_dxhat"); f.dat.data[:] = hat_dT_dx_arr
        self._fd_cache["hat_dT_dx_fd"] = f

        if v:
            self._plot_profiles(
                x_dofs=x_dofs_si,
                ne=ne_init,
                nFC=nFC_init,
                nCX=nCX_init,
                title=f"Initial guess (SI, from non-dim solver): '{initial_guess}'",
            )

        # ------------------------------------------------------------------
        # Boundary conditions in non-dim units.
        # IntervalMesh boundary IDs: 1 = left (hat_x = -1), 2 = right (0).
        # ------------------------------------------------------------------
        hat_ne_x0_c    = Constant(hat_ne_x0)
        hat_ne_inner_c = Constant(hat_ne_inner)
        hat_nFC_x0_c   = Constant(hat_nFC_x0)
        hat_nCX_x0_c   = Constant(hat_nCX_x0)
        hat_dne_dx_inner_c = Constant(hat_dne_dx_inner)

        bcs = [
            DirichletBC(W.sub(0), hat_ne_x0_c,  2),
            DirichletBC(W.sub(1), hat_nFC_x0_c, 2),
            DirichletBC(W.sub(2), hat_nCX_x0_c, 2),
        ]
        if ne_inner_bc == "dirichlet":
            bcs.append(DirichletBC(W.sub(0), hat_ne_inner_c, 1))

        # ------------------------------------------------------------------
        # Assemble residuals and solve via SNES.
        # ------------------------------------------------------------------
        v_e, v_F, v_C = TestFunctions(W)
        hat_ne_curr, hat_nFC_curr, hat_nCX_curr = split(u)

        F1 = self._build_f1_weak_form_nondim(
            hat_ne_curr,
            self._fd_cache["hat_g_fd"],
            self._fd_cache["hat_C_ETG_fd"],
            self._fd_cache["hat_D_NEO_fd"],
            self._fd_cache["hat_A_KBM_fd"],
            self._fd_cache["hat_B_KBM_fd"],
            self._fd_cache["hat_T_fd"],
            self._fd_cache["hat_dT_dx_fd"],
            self._fd_cache["hat_Si_fd"],
            hat_nFC_curr, hat_nCX_curr,
            v_e, ne_inner_bc, hat_dne_dx_inner_c,
        )
        F2 = self._build_f2_weak_form_nondim(
            hat_ne_curr, hat_nFC_curr,
            self._fd_cache["hat_g_fd"],
            self._fd_cache["hat_Si_fd"],
            self._fd_cache["hat_Scx_fd"],
            hat_VFC_const, fFC_const, v_F,
        )
        F3 = self._build_f3_weak_form_nondim(
            hat_ne_curr, hat_nFC_curr, hat_nCX_curr,
            self._fd_cache["hat_g_fd"],
            self._fd_cache["hat_Si_fd"],
            self._fd_cache["hat_Scx_fd"],
            self._fd_cache["hat_Vcx_fd"],
            fCX_const, half, v_C,
        )

        F = F1 + F2 + F3
        snes_params = self._build_petsc_solver_parameters(
            linear_solver=linear_solver,
            ksp_rtol=ksp_rtol,
            ksp_max_it=ksp_max_it,
        )
        solve(F == 0, u, bcs=bcs, solver_parameters=snes_params)

        # ------------------------------------------------------------------
        # Extract converged hat profiles, then recover SI profiles.
        # ------------------------------------------------------------------
        hat_ne_fd, hat_nFC_fd, hat_nCX_fd = u.subfunctions
        sort_idx = np.argsort(hat_x_dofs)
        self.hat_x_sol   = hat_x_dofs[sort_idx]
        self.hat_ne_sol  = hat_ne_fd .dat.data[sort_idx]
        self.hat_nFC_sol = hat_nFC_fd.dat.data[sort_idx]
        self.hat_nCX_sol = hat_nCX_fd.dat.data[sort_idx]

        # SI profiles -- same attribute names as parent solver so plotting
        # / EPED feed / etc. all keep working unchanged.
        self.x_sol   = self.hat_x_sol  * self._L_nd
        self.ne_sol  = self.hat_ne_sol  * self._n0_nd
        self.nFC_sol = self.hat_nFC_sol * self._n0_nd
        self.nCX_sol = self.hat_nCX_sol * self._n0_nd

        self.u_fd = u
        self.W_fd = W
        self.V_fd = V

        if v:
            print(
                f"[nondim] hat residuals solved.\n"
                f"  hat_n_e  in [{self.hat_ne_sol.min():.3e}, {self.hat_ne_sol.max():.3e}]\n"
                f"  hat_n_FC in [{self.hat_nFC_sol.min():.3e}, {self.hat_nFC_sol.max():.3e}]\n"
                f"  hat_n_CX in [{self.hat_nCX_sol.min():.3e}, {self.hat_nCX_sol.max():.3e}]"
            )

        return self.x_sol, self.ne_sol, self.nFC_sol, self.nCX_sol
