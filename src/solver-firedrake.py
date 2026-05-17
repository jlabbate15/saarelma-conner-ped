"""Firedrake-based solver for the full coupled three-equation
Saarelma-Connor neutral-transport pedestal model.

This module subclasses :class:`saarelma_connor` from ``src/solver.py`` and
re-uses its initialisation pipeline (MHD load, kinetic-profile load,
cross-sections, diffusion coefficient, flux-surface-averages, form
factors).  Instead of the iterative ``solve_bvp`` call used in the parent
class, here we use Firedrake to solve the *full* coupled system of three
ODEs in one shot, after collapsing the poloidal integrals analytically
via the standard form-factor substitution.

Form-factor substitution
========================
Define
    alpha(x) =  oint R^2 dtheta                         (geometric)
    beta(x)  =  oint R^2 |grad r|^2 dtheta = alpha * g  (geometric)
    g(x)     =  <|grad r|^2>                            (gradr2_fsa)
    f_FC     =  <|grad r|^2 n_FC>/(<n_FC><|grad r|^2>)  (form factor)
    f_CX     =  <|grad r|^2 n_CX>/(<n_CX><|grad r|^2>)  (form factor)
With the radial component conventions V_{FC,r} = -V_FC, V_{CX,r} = -V_CX
(V_FC, V_CX positive magnitudes; minus sign captures the inward flow),
the original three equations,

    d/dx ( D_ped * oint R^2|grad r|^2 dth * dn_e/dx )
        = - n_e * oint R^2 dth * S_i * (n_FC + n_CX)

    d/dx ( oint R^2|grad r|^2 dth * V_{FC,r} * n_FC )
        = - n_e * oint R^2 dth * (S_i + S_CX) * n_FC

    d/dx ( oint R^2|grad r|^2 dth * V_{CX,r} * n_CX )
        = - n_e * oint R^2 dth * ( S_i n_CX - 1/2 S_CX n_FC )

reduce, after applying the form factors and substituting V_{*,r} = -V_*,
to the cleaner form actually solved here:

    d/dx ( D_ped * alpha * g * dn_e/dx )
        = - n_e * alpha * S_i * (n_FC + n_CX)

    d/dx ( V_FC * f_FC * alpha * g * n_FC )
        =   n_e * alpha * (S_i + S_CX) * n_FC

    d/dx ( V_CX * f_CX * alpha * g * n_CX )
        =   n_e * alpha * ( S_i n_CX - 1/2 S_CX n_FC )

Boundary conditions
===================
Following the user's request:
    n_e (x = 0)         = ne_x0          (Dirichlet at separatrix)
    n_e (x = -infty)    = n_e(x_inner)   (Dirichlet, taken from p-file)
    n_FC(x = 0)         = nFC_x0         (Dirichlet at separatrix)
    n_CX(x = 0)         = nCX_x0         (Dirichlet at separatrix)
The first-order n_FC and n_CX equations are well-posed with a single BC
imposed at the inflow boundary x = 0 (since V_{*,r} < 0 means information
propagates from the separatrix inward).
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import simpson, cumulative_trapezoid
import matplotlib.pyplot as plt

from src.solver import saarelma_connor

try:
    from firedrake import (
        IntervalMesh, FunctionSpace, MixedFunctionSpace, Function,
        TestFunctions, Constant, DirichletBC, dx, split, solve,
        SpatialCoordinate,
    )
    _FIREDRAKE_AVAILABLE = True
except Exception as _firedrake_import_err:  # pragma: no cover - import guard
    _FIREDRAKE_AVAILABLE = False
    _FIREDRAKE_IMPORT_ERR = _firedrake_import_err


class saarelma_connor_firedrake(saarelma_connor):
    """Saarelma-Connor pedestal model solved with Firedrake.

    Inherits the entire initialisation chain (MHD/kinetic load,
    cross-sections, diffusion coefficient, flux-surface averages, form
    factors) from :class:`saarelma_connor` and replaces the iterative
    ``solve_bvp`` call with a Firedrake finite-element solve of the full
    coupled (n_e, n_FC, n_CX) system after the form-factor substitution.

    Parameters
    ----------
    nCX_x0 : float, optional
        m^-3, charge-exchange neutral density at the separatrix
        (Dirichlet boundary condition).  Defaults to 0.1 * nFC_x0 if not
        supplied, matching the typical CX/FC density ratio close to the
        separatrix.
    *args, **kwargs : passed straight to :class:`saarelma_connor`.
    """

    def __init__(self, *args, nCX_x0=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.nCX_x0 = nCX_x0 if nCX_x0 is not None else 0.1 * self.nFC_x0
        self.alpha_fsa = None  # filled in by calc_alpha_fsa()

    # ------------------------------------------------------------------ #
    # New geometric helper: alpha(x) = oint R^2 dtheta on each surface    #
    # ------------------------------------------------------------------ #
    def calc_alpha_fsa(self):
        """Compute alpha(x) = oint R^2 dtheta at every psi_N_pres surface.

        The flux-surface contour is extracted exactly the same way as in
        :meth:`saarelma_connor.calc_gradr`, so the resulting alpha array
        is consistent with the existing ``gradr_fsa``/``gradr2_fsa``
        denominator (``simpson(R_c**2, theta_c)``).

        Sets
        ----
        self.alpha_fsa : ndarray, shape (n_psi,)
            Geometric factor alpha at each psi_N_pres surface (m^2).
        """
        Z_axis = self.eq['zaxis']
        R_axis = self.eq['raxis']
        n_psi = len(self.psi_N_pres)
        self.alpha_fsa = np.full(n_psi, np.nan)

        fig, ax = plt.subplots()
        for i, psi_val in enumerate(self.psi_pres):
            ax.cla()
            cs = ax.contour(self.rgrid, self.zgrid, self.psi_RZ,
                            levels=[psi_val])
            segs = cs.allsegs[0]
            if not segs:
                continue
            seg = max(segs, key=lambda s: len(s))
            R_c, Z_c = seg[:, 0], seg[:, 1]

            theta_c = np.arctan2(Z_c - Z_axis, R_c - R_axis)
            idx = np.argsort(theta_c)
            theta_c, R_c, Z_c = theta_c[idx], R_c[idx], Z_c[idx]

            theta_c = np.append(theta_c, theta_c[0] + 2*np.pi)
            R_c = np.append(R_c, R_c[0])
            Z_c = np.append(Z_c, Z_c[0])

            self.alpha_fsa[i] = simpson(R_c**2, theta_c)
        plt.close(fig)

        valid = np.isfinite(self.alpha_fsa) & (self.alpha_fsa > 0)
        if valid.any() and not valid.all():
            self.alpha_fsa[:] = interp1d(
                self.psi_N_pres[valid], self.alpha_fsa[valid],
                kind='linear', bounds_error=False, fill_value='extrapolate'
            )(self.psi_N_pres)

    # ------------------------------------------------------------------ #
    # Main entry point: build & solve the coupled FE system               #
    # ------------------------------------------------------------------ #
    def solve_firedrake(self,
                        x_res=200,
                        mesh_n=400,
                        fe_degree=2,
                        ne_inner=None,
                        initial_guess="linear",
                        use_picard=True,
                        picard_tol=1e-4,
                        max_picard=40,
                        relax=0.5,
                        verbose=None):
        """Solve the coupled (n_e, n_FC, n_CX) system with Firedrake.

        Parameters
        ----------
        x_res : int
            Resolution used to set up the coefficient grid via
            :meth:`saarelma_connor.setup_solver_grids`.
        mesh_n : int
            Number of cells in the 1D Firedrake mesh.
        fe_degree : int
            Polynomial degree of the CG finite-element basis.
        ne_inner : float or None
            Dirichlet value of n_e at the inner boundary x = x_inner
            (representing the user's prescribed n_e(x = -infty)).  If
            ``None`` (the default), the value is read from the p-file at
            the inner boundary, mirroring the behaviour of the parent
            solver.  Pass an explicit number to use a purely
            "boundary-condition-only" formulation.
        initial_guess : {"linear", "pfile"}
            * ``"linear"`` (default) -- build the initial guess from the
              four prescribed boundary conditions only:
                  n_e  : linear from ne_inner at x_inner to ne_x0 at 0
                  n_FC : linear from 0           at x_inner to nFC_x0 at 0
                  n_CX : linear from 0           at x_inner to nCX_x0 at 0
              No use of the p-file profile or the analytical exponential
              decay estimate.
            * ``"pfile"`` -- legacy guess: p-file electron density profile
              and the analytical FC exponential decay used by the parent
              solver.
        use_picard : bool
            If True (default) use a Picard fixed-point iteration that
            lags the bilinear nonlinearity at each step.  If False, do a
            single full-Newton solve (less robust, but faster when it
            does converge).
        picard_tol : float
            Relative L2 convergence tolerance on n_e for the Picard loop.
        max_picard : int
            Maximum number of Picard iterations.
        relax : float, in (0, 1]
            Under-relaxation factor on the Picard update.  ``1.0`` means
            no relaxation; smaller values trade speed for robustness.
            The default is 0.5, well-suited to the linear initial guess.
        verbose : bool or None
            Override ``self.verbose`` for the duration of this solve.

        Sets
        ----
        self.x_sol     : ndarray
        self.ne_sol    : ndarray
        self.nFC_sol   : ndarray
        self.nCX_sol   : ndarray
            Converged profiles on the (sorted) DOF grid.
        self.u_fd      : Firedrake mixed Function (raw FE solution).
        """
        if not _FIREDRAKE_AVAILABLE:
            raise ImportError(
                "Firedrake is not available in this environment.  "
                "Install Firedrake (https://www.firedrakeproject.org/) "
                f"to use this solver.  Original import error:\n"
                f"  {_FIREDRAKE_IMPORT_ERR}"
            )

        v = self.verbose if verbose is None else bool(verbose)

        # ------------------------------------------------------------------
        # 1.  Run the same set-up pipeline as the parent solver
        # ------------------------------------------------------------------
        self.form_factor(type='FC')
        self.form_factor(type='cx')
        self.setup_solver_grids(res=x_res)
        self.find_inner_boundary()
        if self.alpha_fsa is None:
            self.calc_alpha_fsa()

        # ------------------------------------------------------------------
        # 2.  Build sorted coefficient arrays for fast np.interp lookup
        # ------------------------------------------------------------------
        sort_idx = np.argsort(self.x_init)
        x_sorted        = self.x_init      [sort_idx]
        D_sorted        = self.D_ped       [sort_idx]
        g_sorted        = self.gradr2_fsa  [sort_idx]
        alpha_sorted    = self.alpha_fsa   [sort_idx]
        Si_sorted       = self.S_i_pres    [sort_idx]
        Scx_sorted      = self.S_cx_pres   [sort_idx]
        Vcx_sorted      = np.abs(self.V_cx_pres[sort_idx])
        ne_pfile_sorted = self.n_e_pres    [sort_idx]

        if ne_inner is None:
            # Fallback: take the value from the p-file at x_inner
            ne_inner_val = float(np.interp(self.x_inner, x_sorted, ne_pfile_sorted))
            ne_inner_src = "p-file"
        else:
            ne_inner_val = float(ne_inner)
            ne_inner_src = "user"
        if v:
            print(f"[firedrake] x_inner       = {self.x_inner:.4e} m")
            print(f"[firedrake] ne(x_inner)   = {ne_inner_val:.3e} m^-3 ({ne_inner_src})")
            print(f"[firedrake] ne_x0         = {self.ne_x0:.3e} m^-3")
            print(f"[firedrake] nFC_x0        = {self.nFC_x0:.3e} m^-3")
            print(f"[firedrake] nCX_x0        = {self.nCX_x0:.3e} m^-3")
            print(f"[firedrake] initial_guess = {initial_guess!r}")

        # ------------------------------------------------------------------
        # 3.  Construct the 1D mesh and mixed CG function space
        # ------------------------------------------------------------------
        x_left  = float(self.x_inner)
        x_right = 0.0
        if x_left >= x_right:
            raise ValueError(
                f"x_inner = {x_left} must be strictly less than 0 (separatrix)."
            )
        mesh = IntervalMesh(mesh_n, x_left, x_right)
        V    = FunctionSpace(mesh, "CG", fe_degree)
        W    = MixedFunctionSpace([V, V, V])

        # DOF coordinates on V (used to evaluate np.interp into FE Functions).
        # The current Firedrake API does not expose tabulate_dof_coordinates
        # on a FunctionSpace, so we obtain the per-DOF x position by
        # interpolating the spatial coordinate onto V; the returned array is
        # in dat.data order, which is exactly what we need for direct
        # buffer assignment.
        x_coord_func = Function(V).interpolate(SpatialCoordinate(mesh)[0])
        x_dofs = x_coord_func.dat.data.copy()

        def _make_func(arr_sorted, name=""):
            """Project a numpy array defined on x_sorted onto V via np.interp."""
            f = Function(V, name=name)
            f.dat.data[:] = np.interp(x_dofs, x_sorted, arr_sorted)
            return f

        D_fd     = _make_func(D_sorted,     "D_ped")
        g_fd     = _make_func(g_sorted,     "gradr2_fsa")
        alpha_fd = _make_func(alpha_sorted, "alpha_fsa")
        Si_fd    = _make_func(Si_sorted,    "S_i")
        Scx_fd   = _make_func(Scx_sorted,   "S_cx")
        Vcx_fd   = _make_func(Vcx_sorted,   "V_cx")

        VFC_const = Constant(abs(self.V_FC))
        fFC_const = Constant(self.fFC)
        fCX_const = Constant(self.fCX)
        half      = Constant(0.5)

        beta_fd = Function(V, name="beta")  # beta = alpha * g, helper
        beta_fd.dat.data[:] = alpha_fd.dat.data[:] * g_fd.dat.data[:]

        # ------------------------------------------------------------------
        # 4.  Initial guess
        # ------------------------------------------------------------------
        if initial_guess == "linear":
            # BC-only initial guess.  Build linear ramps directly on the
            # FE DOF coordinates -- no use of the p-file profile or any
            # analytical decay estimate.
            #
            #   n_e (x) = ne_inner_val + (ne_x0  - ne_inner_val) * (x - x_left) / (x_right - x_left)
            #   n_FC(x) =                 nFC_x0                  * (x - x_left) / (x_right - x_left)
            #   n_CX(x) =                 nCX_x0                  * (x - x_left) / (x_right - x_left)
            #
            # n_e uses both Dirichlet ends (the only equation that has
            # two of them).  n_FC and n_CX use their single Dirichlet
            # value at x = 0 and ramp linearly down to 0 at x_inner --
            # zero is a physically reasonable estimate deep in the
            # plasma where neutrals have been ionised away.
            xi = (x_dofs - x_left) / (x_right - x_left)         # in [0,1]
            ne_init_data  = ne_inner_val + (self.ne_x0 - ne_inner_val) * xi
            nFC_init_data = self.nFC_x0 * xi
            nCX_init_data = self.nCX_x0 * xi

            u      = Function(W, name="u")
            u_prev = Function(W, name="u_prev")
            u.subfunctions[0].dat.data[:] = ne_init_data
            u.subfunctions[1].dat.data[:] = nFC_init_data
            u.subfunctions[2].dat.data[:] = nCX_init_data
            u_prev.assign(u)

        elif initial_guess == "pfile":
            # Legacy initial guess: p-file n_e, exponential-decay n_FC.
            #   n_FC(x) ~ nFC_x0 * exp( int_0^x ne (Si+Scx)/(fFC*V_FC) dx' )
            # is computed in descending-x ordering so the integral sign is
            # unambiguous for x < 0 inside the pedestal.
            order_desc      = np.argsort(x_sorted)[::-1]
            x_desc          = x_sorted[order_desc]
            integrand_init  = (ne_pfile_sorted[order_desc]
                               * (Si_sorted[order_desc] + Scx_sorted[order_desc])
                               / (self.fFC * abs(self.V_FC)))
            cumint_desc     = cumulative_trapezoid(integrand_init, x_desc, initial=0.0)
            nFC_ratio_desc  = np.exp(cumint_desc)
            nFC_init_sorted = np.empty_like(nFC_ratio_desc)
            nFC_init_sorted[order_desc] = self.nFC_x0 * nFC_ratio_desc

            ratio_CX        = self.nCX_x0 / self.nFC_x0 if self.nFC_x0 > 0 else 0.0
            nCX_init_sorted = ratio_CX * nFC_init_sorted

            u      = Function(W, name="u")
            u_prev = Function(W, name="u_prev")
            u.sub(0).assign(_make_func(ne_pfile_sorted,  "ne_init"))
            u.sub(1).assign(_make_func(nFC_init_sorted,  "nFC_init"))
            u.sub(2).assign(_make_func(nCX_init_sorted,  "nCX_init"))
            u_prev.assign(u)

        else:
            raise ValueError(
                f"Unknown initial_guess={initial_guess!r}; expected "
                f"'linear' or 'pfile'."
            )

        # ------------------------------------------------------------------
        # 6.  Dirichlet boundary conditions
        # IntervalMesh boundary IDs:  1 = left (x = x_inner),  2 = right (x = 0)
        # ------------------------------------------------------------------
        ne_x0_c    = Constant(self.ne_x0)
        ne_inner_c = Constant(ne_inner_val)
        nFC_x0_c   = Constant(self.nFC_x0)
        nCX_x0_c   = Constant(self.nCX_x0)

        bcs = [
            DirichletBC(W.sub(0), ne_inner_c, 1),  # n_e(x_inner)
            DirichletBC(W.sub(0), ne_x0_c,    2),  # n_e(0)
            DirichletBC(W.sub(1), nFC_x0_c,   2),  # n_FC(0)
            DirichletBC(W.sub(2), nCX_x0_c,   2),  # n_CX(0)
        ]

        # ------------------------------------------------------------------
        # 7.  Variational forms
        #
        # Eq (1) -- diffusion equation.  Multiplying d/dx(D beta n_e') =
        # -n_e alpha S_i (n_FC+n_CX) by -1 puts it in the standard
        # divergence form -d/dx(D beta n_e') = n_e alpha S_i (n_FC+n_CX).
        # Galerkin + IBP (with v_e = 0 at both Dirichlet ends, so the
        # boundary flux term vanishes) gives the residual
        #     int D beta n_e' v_e' dx  -  int n_e alpha S_i (n_FC+n_CX) v_e dx = 0
        #
        # Eqs (2,3) -- first-order ODEs in n_FC, n_CX.  We use the strong
        # residual form (no IBP) so the Galerkin solution matches a
        # forward marching solve from the inflow boundary x = 0:
        #     int [ d/dx(V_FC f_FC beta n_FC) - n_e alpha (S_i+S_CX) n_FC ] v_F dx = 0
        #     int [ d/dx(V_CX f_CX beta n_CX)
        #           - n_e alpha (S_i n_CX - 1/2 S_CX n_FC)              ] v_C dx = 0
        # ------------------------------------------------------------------
        v_e, v_F, v_C = TestFunctions(W)

        # --- Picard iteration -------------------------------------------
        # The system is bilinear in (n_e, n_FC, n_CX) -- every nonlinearity
        # is a *product* of two unknowns.  The cleanest way to make it
        # linear at each step is to evaluate the "lagged" factor at u_prev
        # while differentiating against the "current" unknown in u.  This
        # gives a properly linear sub-problem each Picard step and keeps
        # Newton-style convergence behaviour for the eventual fixed point.
        ne_prev,  nFC_prev,  nCX_prev  = split(u_prev)
        ne_curr,  nFC_curr,  nCX_curr  = split(u)

        # ----- Sign accounting for the residuals -----
        # Eq 1 weak form (after IBP, v_e = 0 at both Dirichlet ends):
        #     int D beta n_e' v_e' dx  -  int n_e alpha S_i (n_FC+n_CX) v_e dx = 0
        # Eqs 2, 3 strong-form residuals (no IBP):
        #     int [ d/dx(V_FC f_FC beta n_FC) - n_e alpha (S_i+S_CX) n_FC ] v_F dx = 0
        #     int [ d/dx(V_CX f_CX beta n_CX) - n_e alpha (S_i n_CX - 1/2 S_CX n_FC) ] v_C dx = 0
        if use_picard:
            # Eq 1: lag (n_FC + n_CX) on RHS
            F1 = (D_fd * beta_fd * ne_curr.dx(0) * v_e.dx(0)
                  - ne_curr * alpha_fd * Si_fd * (nFC_prev + nCX_prev) * v_e) * dx
            # Eq 2: lag n_e on RHS
            F2 = ((VFC_const * fFC_const * beta_fd * nFC_curr).dx(0) * v_F
                  - ne_prev * alpha_fd * (Si_fd + Scx_fd) * nFC_curr * v_F) * dx
            # Eq 3: lag n_e on RHS, lag n_FC in CX-source
            F3 = ((Vcx_fd * fCX_const * beta_fd * nCX_curr).dx(0) * v_C
                  - ne_prev * alpha_fd * (
                        Si_fd * nCX_curr - half * Scx_fd * nFC_prev
                    ) * v_C) * dx
        else:
            # Full Newton on the truly nonlinear residual -- use ne_curr
            # everywhere.  More fragile but useful for debugging.
            F1 = (D_fd * beta_fd * ne_curr.dx(0) * v_e.dx(0)
                  - ne_curr * alpha_fd * Si_fd * (nFC_curr + nCX_curr) * v_e) * dx
            F2 = ((VFC_const * fFC_const * beta_fd * nFC_curr).dx(0) * v_F
                  - ne_curr * alpha_fd * (Si_fd + Scx_fd) * nFC_curr * v_F) * dx
            F3 = ((Vcx_fd * fCX_const * beta_fd * nCX_curr).dx(0) * v_C
                  - ne_curr * alpha_fd * (
                        Si_fd * nCX_curr - half * Scx_fd * nFC_curr
                    ) * v_C) * dx

        F = F1 + F2 + F3

        # PETSc solver parameters: direct LU factorisation is plenty for a
        # 1D problem with only ~3*mesh_n DOFs.
        solver_params = {
            "snes_type": "newtonls",
            "snes_max_it": 50,
            "snes_atol": 1.0e-12,
            "snes_rtol": 1.0e-10,
            "snes_linesearch_type": "bt",
            "ksp_type": "preonly",
            "pc_type": "lu",
            "mat_type": "aij",
        }

        # ------------------------------------------------------------------
        # 8.  Solve
        # ------------------------------------------------------------------
        if use_picard:
            rel = np.inf
            for k in range(max_picard):
                solve(F == 0, u, bcs=bcs, solver_parameters=solver_params)

                # Direct manipulation of underlying numpy buffers is the
                # most reliable way to mix two MixedFunctions in Firedrake.
                ne_data_new  = u.subfunctions[0].dat.data
                ne_data_old  = u_prev.subfunctions[0].dat.data
                nFC_data_new = u.subfunctions[1].dat.data
                nFC_data_old = u_prev.subfunctions[1].dat.data
                nCX_data_new = u.subfunctions[2].dat.data
                nCX_data_old = u_prev.subfunctions[2].dat.data

                # Convergence diagnostic on n_e BEFORE relaxation, so the
                # tolerance refers to the actual change introduced by the
                # linear solve, not the dampened one.
                diff = ne_data_new - ne_data_old
                err = float(np.sqrt(np.sum(diff * diff)))
                ncr = float(np.sqrt(np.sum(ne_data_new * ne_data_new)))
                rel = err / max(ncr, 1.0e-30)
                if v:
                    print(f"[firedrake] Picard iter {k:>3d}:  "
                          f"||dne||/||ne|| = {rel:.3e}")

                # Under-relaxed update: u <- (1-relax)*u_prev + relax*u
                if relax < 1.0:
                    ne_data_new [:] = (1.0 - relax) * ne_data_old  + relax * ne_data_new
                    nFC_data_new[:] = (1.0 - relax) * nFC_data_old + relax * nFC_data_new
                    nCX_data_new[:] = (1.0 - relax) * nCX_data_old + relax * nCX_data_new

                if rel < picard_tol:
                    if v:
                        print(f"[firedrake] Picard converged in {k+1} iterations.")
                    break
                u_prev.assign(u)
            else:
                import warnings
                warnings.warn(
                    f"Picard iteration did not reach tol={picard_tol:g} "
                    f"after {max_picard} iterations (last rel = {rel:.3e}).",
                    RuntimeWarning,
                )
        else:
            solve(F == 0, u, bcs=bcs, solver_parameters=solver_params)

        # ------------------------------------------------------------------
        # 9.  Extract & store the converged profiles
        # ------------------------------------------------------------------
        ne_fd, nFC_fd, nCX_fd = u.subfunctions

        sort_x_idx  = np.argsort(x_dofs)
        self.x_sol  = x_dofs[sort_x_idx]
        self.ne_sol  = ne_fd .dat.data[sort_x_idx]
        self.nFC_sol = nFC_fd.dat.data[sort_x_idx]
        self.nCX_sol = nCX_fd.dat.data[sort_x_idx]

        self.u_fd = u
        self.W_fd = W
        self.V_fd = V

        return self.x_sol, self.ne_sol, self.nFC_sol, self.nCX_sol
