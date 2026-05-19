"""Firedrake-based solver for the *coupled three-equation* Saarelma-Connor
neutral-transport pedestal model in its un-reduced form.

System solved
=============
Origin: Saarelma & Connor 2023 Nucl. Fusion 63 052002
Unknowns:
    n_e(x)         (electron density; treated as a flux function)
    <n_FC>(x)      (FSA Franck-Condon neutral density)
    <n_CX>(x)      (FSA charge-exchange neutral density)
with flux-surface average (FSA) defined by  <A>(x) = oint R^2 A dtheta / oint R^2 dtheta
on the radial coordinate x = r - r_sep (separatrix at x = 0, core at x<0).

With form factors
    f_FC(x) = <|grad r|^2 n_FC> / ( <n_FC> <|grad r|^2> )
    f_CX(x) = <|grad r|^2 n_CX> / ( <n_CX> <|grad r|^2> )
the coupled system to solve is:

    d/dx [ <|grad r|^2> D_ped dn_e/dx ]
        = - n_e S_i ( <n_FC> + <n_CX> )

    |V_FC| d/dx [ f_FC <n_FC> <|grad r|^2> ]
        = + n_e (S_i + S_CX) <n_FC>

    d/dx [ |V_CX| f_CX <n_CX> <|grad r|^2> ]
        =  n_e ( <n_CX> S_i - (S_CX/2) <n_FC> )

Boundary conditions
===================
    n_e (x = 0)         = ne_x0         (Dirichlet at separatrix)
    n_e (x = x_inner)   = ne_inner      (Dirichlet, inner pedestal boundary)
    <n_FC>(x = 0)       = nFC_x0        (Dirichlet at separatrix)
    <n_CX>(x = 0)       = nCX_x0        (Dirichlet at separatrix)
The first-order ODEs for <n_FC> and <n_CX> have characteristic flowing
from the separatrix inward (V_{*,r} < 0), so a single Dirichlet BC at
x = 0 is well-posed.  The diffusion equation is second-order in n_e and
takes Dirichlet at both ends.
"""

import numpy as np
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid
import matplotlib.pyplot as plt

from src.solver import saarelma_connor

try:
    from firedrake import (
        IntervalMesh, FunctionSpace, MixedFunctionSpace, Function,
        TestFunctions, TrialFunction, Constant, DirichletBC,
        dx, ds, split, solve, lhs, rhs,
        LinearVariationalProblem, LinearVariationalSolver,
        SpatialCoordinate,
    )
    _FIREDRAKE_AVAILABLE = True
except Exception as _firedrake_import_err:  # raise an error if Firedrake is not available
    _FIREDRAKE_AVAILABLE = False
    _FIREDRAKE_IMPORT_ERR = _firedrake_import_err


class saarelma_connor_firedrake(saarelma_connor):
    """Saarelma-Connor pedestal model solved with Firedrake.

    Inherits the entire initialisation chain (MHD/kinetic load,
    cross-sections, diffusion coefficient, flux-surface averages, form
    factors) from :class:`saarelma_connor` and replaces the iterative
    ``solve_bvp`` call with a Firedrake finite-element solve of the
    coupled (n_e, <n_FC>, <n_CX>) system.

    Parameters
    ----------
    nCX_x0 : float, optional
        m^-3, CX neutral density at the separatrix (Dirichlet BC at
        x = 0).  Defaults to 0.1 * nFC_x0.
    *args, **kwargs : passed straight to :class:`saarelma_connor`. Takes all the same arguments as the parent class
    """

    def __init__(self, *args, nCX_x0=None, **kwargs):
        super().__init__(*args, **kwargs) # call the parent class (saarelma-connor class) init method
        self.nCX_x0 = nCX_x0 if nCX_x0 is not None else 0.1 * self.nFC_x0 # set nCX boundary condition, defaults to 0.1 * nFC_x0
        self._fd_cache = {}

    def invalidate_firedrake_cache(self):
        """Drop cached Firedrake meshes, coefficients, and linear solvers.

        Called automatically by :meth:`update_free_params`.  Call manually
        after changing equilibrium inputs or other quantities that affect
        ``setup_solver_grids`` / ``calc_gradr``.
        """
        self._fd_cache = {}

    def update_free_params(self, *args, **kwargs):
        super().update_free_params(*args, **kwargs)
        self.invalidate_firedrake_cache()

    def _plot_profiles(self, x_dofs, ne, nFC, nCX, title=""):
        """Plot n_e, <n_FC>, <n_CX> vs x with a secondary psi_N axis on top.

        Used as a diagnostic from ``solve_firedrake`` (gated on ``verbose``)
        to visualise either the initial guess or any intermediate / final
        profile triple.

        Parameters
        ----------
        x_dofs : ndarray
            Per-DOF x coordinates (m), unsorted (as stored in dat.data).
        ne, nFC, nCX : ndarray
            Profiles in the same DOF order as ``x_dofs`` (m^-3).
        title : str
            Figure suptitle.
        """
        # Sort by ascending x for clean line plots (DOF order is not guaranteed
        # to be spatial).
        sort_idx = np.argsort(x_dofs)
        x_plot = x_dofs[sort_idx]
        profiles = [ne[sort_idx], nFC[sort_idx], nCX[sort_idx]]
        labels   = [r"$n_e$",
                    r"$\langle n_{FC} \rangle$",
                    r"$\langle n_{CX} \rangle$"]
        colours  = ["tab:blue", "tab:orange", "tab:green"]

        # x <-> psi_N mapping built off the parent grid (self.x_init is
        # monotonically increasing toward the separatrix; self.psi_N_pres
        # is the corresponding normalised poloidal flux).  Used by
        # secondary_xaxis to render psi_N on top of each panel.
        x_to_psiN = interp1d(self.x_init, self.psi_N_pres,
                             kind='linear', bounds_error=False,
                             fill_value='extrapolate')
        psiN_to_x = interp1d(self.psi_N_pres, self.x_init,
                             kind='linear', bounds_error=False,
                             fill_value='extrapolate')

        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        if title:
            fig.suptitle(title, fontsize=12)
        for ax, prof, label, col in zip(axes, profiles, labels, colours):
            ax.plot(x_plot, prof, lw=2, color=col)
            ax.set_xlabel(r"$x$ (m)")
            ax.set_ylabel(f"{label} (m$^{{-3}}$)")
            ax.grid(True, alpha=0.3)
            secax = ax.secondary_xaxis('top',
                                       functions=(x_to_psiN, psiN_to_x))
            secax.set_xlabel(r"$\psi_N$")
        plt.show()


    @staticmethod
    def _build_picard_linear_solver_parameters(linear_solver="lu",
                                               ksp_rtol=1e-8,
                                               ksp_max_it=200):
        """PETSc options for :class:`LinearVariationalSolver` (Picard sub-steps).

        Each Picard step solves ``a(u, v) = L(v)`` with fixed ``a`` and
        updated ``L`` from lagged ``u_prev``.  Reusing the same
        :class:`LinearVariationalSolver` lets PETSc reuse the LU
        factorization of ``a`` across iterations.
        """
        linear_solver = str(linear_solver).lower()
        if linear_solver not in ("lu", "gamg"):
            raise ValueError(
                f"linear_solver must be 'lu' or 'gamg', got {linear_solver!r}."
            )

        params = {"mat_type": "aij"}
        if linear_solver == "lu":
            params.update({
                "ksp_type": "preonly",
                "pc_type": "lu",
            })
        else:
            params.update({
                "ksp_type": "gmres",
                "ksp_rtol": float(ksp_rtol),
                "ksp_max_it": int(ksp_max_it),
                "ksp_gmres_restart": 30,
                "pc_type": "gamg",
                "mg_levels_ksp_type": "richardson",
                "mg_levels_pc_type": "sor",
                "mg_levels_ksp_max_it": 5,
            })
        return params

    @staticmethod
    def _build_petsc_solver_parameters(linear_solver="lu",
                                       ksp_rtol=1e-8,
                                       ksp_max_it=200):
        """PETSc options for SNES ``solve(F == 0, ...)`` (full Newton path)."""
        linear_solver = str(linear_solver).lower()
        if linear_solver not in ("lu", "gamg"):
            raise ValueError(
                f"linear_solver must be 'lu' or 'gamg', got {linear_solver!r}."
            )

        params = {
            "snes_type": "newtonls",
            "snes_max_it": 50,
            "snes_atol": 1.0e-8,
            "snes_rtol": 1.0e-8,
            "snes_linesearch_type": "bt",
            "mat_type": "aij",
        }

        if linear_solver == "lu":
            params.update({
                "ksp_type": "preonly",
                "pc_type": "lu",
            })
        else:
            params.update({
                "ksp_type": "gmres",
                "ksp_rtol": float(ksp_rtol),
                "ksp_max_it": int(ksp_max_it),
                "ksp_gmres_restart": 30,
                "pc_type": "gamg",
                "mg_levels_ksp_type": "richardson",
                "mg_levels_pc_type": "sor",
                "mg_levels_ksp_max_it": 5,
            })

        return params

    def _ensure_firedrake_coefficient_grids(self, x_res, force=False):
        """Run ``form_factor`` + ``setup_solver_grids`` once per ``x_res``."""
        key = int(x_res)
        if force or self._fd_cache.get("x_res") != key:
            self.form_factor(type='FC')
            self.form_factor(type='cx')
            self.setup_solver_grids(res=x_res)
            self._fd_cache["x_res"] = key

    def _firedrake_mesh_key(self, x_left, mesh_n, fe_degree):
        return (round(float(x_left), 12), int(mesh_n), int(fe_degree))

    def _ensure_firedrake_discretization(self, x_left, mesh_n, fe_degree, force=False):
        """Build or reuse mesh, spaces, and coefficient Functions on the mesh."""
        mesh_key = self._firedrake_mesh_key(x_left, mesh_n, fe_degree)
        if (not force
                and self._fd_cache.get("mesh_key") == mesh_key
                and "mesh" in self._fd_cache):
            return (
                self._fd_cache["mesh"],
                self._fd_cache["V"],
                self._fd_cache["W"],
                self._fd_cache["x_dofs"],
                self._fd_cache["D_fd"],
                self._fd_cache["g_fd"],
                self._fd_cache["Si_fd"],
                self._fd_cache["Scx_fd"],
                self._fd_cache["Vcx_fd"],
            )

        mesh = IntervalMesh(mesh_n, x_left, 0.0)
        V = FunctionSpace(mesh, "CG", fe_degree)
        W = MixedFunctionSpace([V, V, V])

        x_coord_func = Function(V).interpolate(SpatialCoordinate(mesh)[0])
        x_dofs = x_coord_func.dat.data.copy()

        def _make_func(arr, name=""):
            f = Function(V, name=name)
            f.dat.data[:] = np.interp(x_dofs, self.x_init, arr)
            return f

        D_fd = _make_func(self.D_ped, "D_ped")
        g_fd = _make_func(self.gradr2_fsa, "gradr2_fsa")
        Si_fd = _make_func(self.S_i_pres, "S_i")
        Scx_fd = _make_func(self.S_cx_pres, "S_cx")
        Vcx_fd = _make_func(abs(self.V_cx_pres), "V_cx")

        self._fd_cache.pop("u", None)
        self._fd_cache.pop("u_prev", None)
        self._fd_cache.update({
            "mesh_key": mesh_key,
            "mesh": mesh,
            "V": V,
            "W": W,
            "x_dofs": x_dofs,
            "D_fd": D_fd,
            "g_fd": g_fd,
            "Si_fd": Si_fd,
            "Scx_fd": Scx_fd,
            "Vcx_fd": Vcx_fd,
        })
        return mesh, V, W, x_dofs, D_fd, g_fd, Si_fd, Scx_fd, Vcx_fd

    def _get_or_create_mixed_solution(self, W, force=False):
        """Return cached ``(u, u_prev)`` on ``W``, or allocate new Functions."""
        if (not force
                and self._fd_cache.get("W") is W
                and "u" in self._fd_cache
                and "u_prev" in self._fd_cache):
            return self._fd_cache["u"], self._fd_cache["u_prev"]

        u = Function(W, name="u")
        u_prev = Function(W, name="u_prev")
        self._fd_cache["W"] = W
        self._fd_cache["u"] = u
        self._fd_cache["u_prev"] = u_prev
        return u, u_prev

    @staticmethod
    def _build_picard_bilinear_and_linear_forms(
            W, D_fd, g_fd, Si_fd, Scx_fd, Vcx_fd,
            VFC_const, fFC_const, fCX_const, half,
            u_prev, ne_inner_bc, dne_dx_inner_c):
        """Split lagged Picard residual into ``a(u,v)=L(v)`` for linear solves."""
        v_e, v_F, v_C = TestFunctions(W)
        u_trial = TrialFunction(W)
        ne_t, nFC_t, nCX_t = split(u_trial)
        ne_p, nFC_p, nCX_p = split(u_prev)

        F1 = (g_fd * D_fd * ne_t.dx(0) * v_e.dx(0)
              - ne_t * Si_fd * (nFC_p + nCX_p) * v_e) * dx
        F2 = ((VFC_const * fFC_const * g_fd * nFC_t).dx(0) * v_F
              - ne_p * (Si_fd + Scx_fd) * nFC_t * v_F) * dx
        F3 = ((Vcx_fd * fCX_const * g_fd * nCX_t).dx(0) * v_C
              - ne_p * (Si_fd * nCX_t - half * Scx_fd * nFC_p) * v_C) * dx

        if ne_inner_bc == "neumann":
            F1 = F1 + g_fd * D_fd * dne_dx_inner_c * v_e * ds(1)

        F = F1 + F2 + F3
        return lhs(F), rhs(F)

    @staticmethod
    def _create_picard_linear_solver(a_form, L_form, u, bcs, solver_params):
        """Build a :class:`LinearVariationalSolver` for one Picard solve loop."""
        problem = LinearVariationalProblem(
            a_form, L_form, u, bcs=bcs,
            constant_jacobian=False,
        )
        return LinearVariationalSolver(
            problem, solver_parameters=solver_params,
        )

    def _interpolate_coefficients_to_mesh(self):
        """Re-project parent-grid coefficients onto cached Firedrake ``V``."""
        if "V" not in self._fd_cache:
            return
        x_dofs = self._fd_cache["x_dofs"]

        def _fill(func, arr):
            func.dat.data[:] = np.interp(x_dofs, self.x_init, arr)

        _fill(self._fd_cache["D_fd"], self.D_ped)
        _fill(self._fd_cache["g_fd"], self.gradr2_fsa)
        _fill(self._fd_cache["Si_fd"], self.S_i_pres)
        _fill(self._fd_cache["Scx_fd"], self.S_cx_pres)
        _fill(self._fd_cache["Vcx_fd"], abs(self.V_cx_pres))


    def solve_firedrake(self,
                        x_res=200,
                        mesh_n=400,
                        fe_degree=2,
                        ne_inner=None,
                        ne_inner_bc="neumann",
                        dne_dx_inner=None,
                        initial_guess="linear",
                        tanh_width=None,
                        tanh_center=None,
                        use_picard=True,
                        picard_tol=1e-4,
                        max_picard=40,
                        relax=0.5,
                        linear_solver="lu",
                        ksp_rtol=1e-8,
                        ksp_max_it=200,
                        reuse_setup=True,
                        verbose=None):
        """Solve the coupled (n_e, <n_FC>, <n_CX>) system with Firedrake.

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
            Value of n_e at the inner boundary x = x_inner used for
            (a) the Dirichlet BC when ``ne_inner_bc == "dirichlet"`` and
            (b) the initial guess at x = x_inner in all cases.  If
            ``None`` (the default), the value is read from the p-file at
            the inner boundary, mirroring the behaviour of the parent
            solver.  Pass an explicit number to use a purely
            "boundary-condition-only" formulation.
        ne_inner_bc : {"neumann", "dirichlet"}, default "neumann"
            Type of boundary condition imposed on n_e at x_inner.

            * ``"neumann"`` (default, Saarelma-consistent):
                  dn_e/dx |_{x_inner} = dne_dx_inner
              Matches assumption A7 in docs/derivation_eq15.tex and
              docs/derivation_eq16.tex.  The pedestal-top density value
              is allowed to *emerge* from the balance of fueling,
              ionization, transport, and the separatrix Dirichlet --
              i.e. the model is genuinely *predictive* for the pedestal
              top.  Mathematically, you're matching the pedestal solve
              to the core-side particle flux (since the diffusive flux
              is Gamma_e ~ -D_ped <|grad r|^2> dn_e/dx).

            * ``"dirichlet"``:
                  n_e(x_inner) = ne_inner
              Anchors the pedestal-top density to a known value (e.g.
              from a p-file or an experimental fit).  Useful when you
              trust the pedestal-top density more than the pedestal-top
              gradient, or when you want the solve to reproduce a
              specific experimental top density rather than predict it.

            Trade-offs (see the inline note where the BCs are
            constructed for the full version):
              - Predictive vs. anchored.
              - Sensitivity to noisy p-file data: Dirichlet is more
                forgiving of noisy gradients; Neumann is more forgiving
                of an inaccurate inner-boundary density.
              - The two converge to similar answers if the chosen
                x_inner is sufficiently deep that the p-file value and
                p-file slope at x_inner are mutually consistent with
                each other and with the pedestal model.
        dne_dx_inner : float or None
            Value of dn_e/dx (m^-4) at x = x_inner, used as the Neumann
            BC value when ``ne_inner_bc == "neumann"``.  If ``None``
            (default), it is read from the numerical gradient of the
            p-file electron density at x = x_inner (the same procedure
            the parent solver's ``first_step`` uses to set
            ``self.dne_dx_neginf``).  Must be strictly negative in the
            pedestal (n_e decreasing outward).
        initial_guess : {"linear", "pfile", "tanh"}
            * ``"linear"`` (default) -- build the initial guess from the
              four prescribed boundary conditions only:
                  n_e  : linear from ne_inner at x_inner to ne_x0 at 0
                  n_FC : linear from 0           at x_inner to nFC_x0 at 0
                  n_CX : linear from 0           at x_inner to nCX_x0 at 0
              No use of the p-file profile or the analytical exponential
              decay estimate.
            * ``"pfile"`` -- legacy guess: p-file electron density profile
              and an analytical FC exponential decay (same construction as
              the parent solver's first-step routine).
            * ``"tanh"`` -- H-mode-pedestal-like tanh initial guess.
              n_e transitions from ne_inner_val (pedestal top) to ne_x0
              (separatrix) through a tanh of half-width ``tanh_width/2``
              centred at ``tanh_center``.  Neutrals get the mirrored
              shape (peaked at the separatrix, decaying inward).  Default
              width / centre give a typical narrow pedestal just inside
              the separatrix.
        tanh_width : float or None
            Pedestal width (m) for the ``"tanh"`` initial guess.  The
            tanh argument is ``(x - tanh_center) / (tanh_width/2)``, so
            ``tanh_width`` is roughly the 90%-10% transition width.
            If ``None`` (default), set to ``0.1 * |x_inner|``.
        tanh_center : float or None
            Pedestal-foot x-position (m) for the ``"tanh"`` initial
            guess; should be <= 0.  If ``None`` (default), set to
            ``-tanh_width`` so the pedestal sits just inside the
            separatrix.
        use_picard : bool
            If True (default), run a Picard loop where each step is one
            :class:`LinearVariationalSolver` solve on the lagged residual
            (no SNES).  If False, use a single SNES Newton solve on the
            fully coupled nonlinear residual (less robust; useful for
            cross-checks).
        picard_tol : float
            Relative L2 convergence tolerance on n_e for the Picard loop.
        max_picard : int
            Maximum number of Picard iterations.
        relax : float, in (0, 1]
            Under-relaxation factor on the Picard update.  ``1.0`` means
            no relaxation; smaller values trade speed for robustness.
            The default is 0.5, well-suited to the linear initial guess.
        linear_solver : {"lu", "gamg"}, default ``"lu"``
            Picard: direct/iterative solve of each linear sub-step
            (:meth:`_build_picard_linear_solver_parameters`).  Newton:
            linear solve of each SNES Jacobian step
            (:meth:`_build_petsc_solver_parameters`).

            * ``"lu"`` -- sparse direct LU (default; fast for 1D meshes).
            * ``"gamg"`` -- GMRES + PETSc algebraic multigrid (GAMG)
              preconditioner.  Prefer this when refining ``mesh_n`` /
              ``fe_degree`` makes direct solves expensive.
        ksp_rtol : float, default ``1e-8``
            GMRES relative tolerance when ``linear_solver="gamg"``.
        ksp_max_it : int, default ``200``
            Maximum GMRES iterations per Newton step when using ``"gamg"``.
        reuse_setup : bool, default ``True``
            If True, reuse cached ``setup_solver_grids`` / mesh / coefficient
            projections when ``x_res``, ``mesh_n``, ``fe_degree``, and
            ``x_inner`` are unchanged since the last solve.  Call
            :meth:`invalidate_firedrake_cache` after changing free parameters
            or equilibrium inputs (also done automatically by
            :meth:`update_free_params`).
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
        force_setup = not reuse_setup

        self._ensure_firedrake_coefficient_grids(x_res, force=force_setup)

        # set the inner boundary location and read off the values used for
        # either the Dirichlet or Neumann BC at x = x_inner.
        self.find_inner_boundary() # set self.psi_N_inner_boundary and self.x_inner, where nFC and nCX fall below their thresholds

        # ne(x_inner) -- used for the Dirichlet BC if requested, and for the
        # initial guess at the inner boundary in all cases.
        if ne_inner is None:
            ne_inner_val = float(np.interp(self.x_inner, self.x_init, self.n_e_pres))
            ne_inner_src = "p-file"
        else:
            ne_inner_val = float(ne_inner)
            ne_inner_src = "user"

        # dne/dx(x_inner) -- used for the Neumann BC if requested.  Computed
        # the same way as solver.first_step (np.gradient on the p-file n_e).
        if dne_dx_inner is None:
            dne_dx_pres        = np.gradient(self.n_e_pres, self.x_init)
            dne_dx_inner_val   = float(np.interp(self.x_inner, self.x_init, dne_dx_pres))
            dne_dx_inner_src   = "p-file"
        else:
            dne_dx_inner_val   = float(dne_dx_inner)
            dne_dx_inner_src   = "user"

        ne_inner_bc = str(ne_inner_bc).lower()
        if ne_inner_bc not in ("dirichlet", "neumann"):
            raise ValueError(
                f"ne_inner_bc must be 'dirichlet' or 'neumann', "
                f"got {ne_inner_bc!r}."
            )
        # Saarelma's model assumes the pedestal slope is steeper everywhere
        # inside the domain than at x_inner, so dne/dx(x_inner) must be
        # strictly negative (n_e decreasing outward).  Mirrors the same check
        # in solver.first_step.
        if ne_inner_bc == "neumann" and dne_dx_inner_val >= 0:
            raise ValueError(
                f"Neumann BC value dne/dx(x_inner) = {dne_dx_inner_val:.3e} m^-4 "
                f"is zero or positive.  It must be strictly negative for a "
                f"pedestal solve (density decreasing outward).  Either pass a "
                f"valid dne_dx_inner explicitly or move x_inner deeper (e.g. "
                f"lower psi_N_inner_boundary)."
            )

        if v:
            print(f"[firedrake] x_inner        = {self.x_inner:.4e} m")
            print(f"[firedrake] ne_inner_bc    = {ne_inner_bc!r}")
            print(f"[firedrake] ne(x_inner)    = {ne_inner_val:.3e} m^-3 ({ne_inner_src})")
            print(f"[firedrake] dne/dx(x_in)   = {dne_dx_inner_val:.3e} m^-4 ({dne_dx_inner_src})")
            print(f"[firedrake] ne_x0          = {self.ne_x0:.3e} m^-3")
            print(f"[firedrake] nFC_x0         = {self.nFC_x0:.3e} m^-3")
            print(f"[firedrake] nCX_x0         = {self.nCX_x0:.3e} m^-3")
            # print(f"[firedrake] initial_guess  = {initial_guess!r}")

        
        x_left  = float(self.x_inner)
        x_right = 0.0
        if x_left >= x_right:
            raise ValueError(
                f"x_inner = {x_left} must be strictly less than 0 (separatrix)."
            )

        mesh, V, W, x_dofs, D_fd, g_fd, Si_fd, Scx_fd, Vcx_fd = (
            self._ensure_firedrake_discretization(
                x_left, mesh_n, fe_degree, force=force_setup,
            )
        )
        self._interpolate_coefficients_to_mesh()

        def _make_func(arr, name=""):
            """Project a numpy array on self.x_init onto V via np.interp."""
            f = Function(V, name=name)
            f.dat.data[:] = np.interp(x_dofs, self.x_init, arr)
            return f

        # define constants for the Firedrake system
        VFC_const = Constant(abs(self.V_FC))
        fFC_const = Constant(self.fFC)
        fCX_const = Constant(self.fCX)
        half      = Constant(0.5)

        u, u_prev = self._get_or_create_mixed_solution(W, force=force_setup)

        # set up initial guess
        if initial_guess == "linear":
            # BC-only initial guess: linear ramps between the four
            # prescribed Dirichlet values.  n_FC and n_CX ramp linearly
            # from 0 at x_inner (physically reasonable -- the neutrals
            # have been ionised away deep in the plasma) to their
            # separatrix values at x = 0.
            xi = (x_dofs - x_left) / (x_right - x_left)         # in [0,1]
            ne_init_data  = ne_inner_val + (self.ne_x0 - ne_inner_val) * xi
            nFC_init_data = self.nFC_x0 * xi
            nCX_init_data = self.nCX_x0 * xi

            u.subfunctions[0].dat.data[:] = ne_init_data
            u.subfunctions[1].dat.data[:] = nFC_init_data
            u.subfunctions[2].dat.data[:] = nCX_init_data
            u_prev.assign(u)

        elif initial_guess == "pfile":
            # Legacy initial guess: p-file n_e, exponential-decay n_FC
            # built from  d<n_FC>/dx = ne (Si+Scx)/(fFC |V_FC|) <n_FC>
            # integrated from the separatrix inward.
            x_desc          = self.x_init[::-1]
            integrand_init  = (self.n_e_pres[::-1]
                               * (self.S_i_pres[::-1] + self.S_cx_pres[::-1])
                               / (self.fFC * abs(self.V_FC)))
            cumint_desc     = cumulative_trapezoid(integrand_init, x_desc, initial=0.0)
            nFC_init        = self.nFC_x0 * np.exp(cumint_desc)[::-1]

            ratio_CX  = self.nCX_x0 / self.nFC_x0 if self.nFC_x0 > 0 else 0.0
            nCX_init  = ratio_CX * nFC_init

            u.sub(0).assign(_make_func(self.n_e_pres, "ne_init"))
            u.sub(1).assign(_make_func(nFC_init,      "nFC_init"))
            u.sub(2).assign(_make_func(nCX_init,      "nCX_init"))
            u_prev.assign(u)

        elif initial_guess == "tanh":
            # H-mode-pedestal-like tanh initial guess.
            #   s_ne(x)   = 0.5 * (1 - tanh((x - center) / (width/2)))
            #               -> 1 deep inside the pedestal, -> 0 at the separatrix
            #   n_e(x)    = ne_x0 + (ne_inner_val - ne_x0) * s_ne(x)
            # Neutrals get the mirrored shape (peaked at separatrix):
            #   s_neut(x) = 1 - s_ne(x)
            #   n_FC(x)   = nFC_x0 * s_neut(x)
            #   n_CX(x)   = nCX_x0 * s_neut(x)
            width  = float(tanh_width)  if tanh_width  is not None else 0.1 * abs(x_left)
            center = float(tanh_center) if tanh_center is not None else -width
            if width <= 0:
                raise ValueError(f"tanh_width must be positive, got {width}.")

            s_ne   = 0.5 * (1.0 - np.tanh((x_dofs - center) / (0.5 * width)))
            s_neut = 1.0 - s_ne
            ne_init_data  = self.ne_x0  + (ne_inner_val - self.ne_x0) * s_ne
            nFC_init_data = self.nFC_x0 * s_neut
            nCX_init_data = self.nCX_x0 * s_neut

            u.subfunctions[0].dat.data[:] = ne_init_data
            u.subfunctions[1].dat.data[:] = nFC_init_data
            u.subfunctions[2].dat.data[:] = nCX_init_data
            u_prev.assign(u)

        else:
            raise ValueError(
                f"Unknown initial_guess={initial_guess!r}; expected "
                f"'linear', 'pfile', or 'tanh'."
            )

        if v:
            self._plot_profiles(
                x_dofs=x_dofs,
                ne=u.subfunctions[0].dat.data,
                nFC=u.subfunctions[1].dat.data,
                nCX=u.subfunctions[2].dat.data,
                title=f"Initial guess: '{initial_guess}'",
            )

        # ------------------------------------------------------------------
        # Boundary conditions for the coupled (n_e, <n_FC>, <n_CX>) system.
        # IntervalMesh boundary IDs:  1 = left (x = x_inner),  2 = right (x = 0).
        #
        # The system is order 4 overall (n_e is 2nd order, <n_FC> and <n_CX>
        # are 1st order each) so it needs exactly 4 BCs.  Three of them are
        # always Dirichlet at the separatrix (where the physical values are
        # known):
        #
        #   n_e (x = 0)        = ne_x0
        #   <n_FC>(x = 0)      = nFC_x0
        #   <n_CX>(x = 0)      = nCX_x0
        #
        # x = 0 is also the *inflow* boundary for the first-order ODEs (because
        # V_{FC,r}, V_{CX,r} < 0 -- neutrals stream inward), so a single
        # Dirichlet at x = 0 is the well-posed choice for each neutral
        # species.  Putting them at x_inner instead would be ill-posed
        # (outflow boundary).
        #
        # The fourth BC -- on n_e at x_inner -- is the only one where there's
        # a real modelling choice: Neumann (match the core-side particle flux)
        # or Dirichlet (anchor to a known core-side density).
        #
        # NEUMANN vs DIRICHLET AT x_inner -- PROS AND CONS
        # -------------------------------------------------
        # Neumann  (dn_e/dx |_{x_inner} = dne_dx_inner):
        #   + Saarelma-consistent (assumption A7 in docs/derivation_eq{15,16}.tex).
        #     The pedestal-top density emerges from the model rather than being
        #     prescribed, so the solve is genuinely *predictive* for n_e(x_inner).
        #   + Physically matches the pedestal to the core-side particle flux
        #     Gamma_e ~ -D_ped <|grad r|^2> dn_e/dx, which is what you usually
        #     know (or believe) from the core transport / source modelling.
        #   - Sensitive to noise in the p-file gradient at x_inner.
        #   - Two Neumann BCs (one at each end) would be ill-posed; this
        #     setup uses Neumann at x_inner + Dirichlet at x=0, which is fine.
        #
        # Dirichlet (n_e(x_inner) = ne_inner):
        #   + Forces the solve to pass through a known pedestal-top density,
        #     useful when reproducing a specific experimental profile.
        #   + Robust to noise in the inner-boundary *gradient*.
        #   - No longer predictive at the pedestal top; you are inputting one
        #     of the things the model could otherwise tell you.
        #   - Inconsistent with how the parent solver's first_step / Eq.(15),(16)
        #     derivations close the inner boundary.
        #
        # If x_inner is deep enough that the p-file n_e value and the p-file
        # n_e gradient at that point are mutually consistent (and consistent
        # with the model), Neumann and Dirichlet converge to similar profiles.
        # ------------------------------------------------------------------
        ne_x0_c    = Constant(self.ne_x0)
        ne_inner_c = Constant(ne_inner_val)        # used only for Dirichlet
        nFC_x0_c   = Constant(self.nFC_x0)
        nCX_x0_c   = Constant(self.nCX_x0)
        dne_dx_inner_c = Constant(dne_dx_inner_val)  # used only for Neumann

        bcs = [
            DirichletBC(W.sub(0), ne_x0_c,  2),  # n_e (0)        -- separatrix
            DirichletBC(W.sub(1), nFC_x0_c, 2),  # <n_FC>(0)      -- separatrix
            DirichletBC(W.sub(2), nCX_x0_c, 2),  # <n_CX>(0)      -- separatrix
        ]
        if ne_inner_bc == "dirichlet":
            bcs.append(DirichletBC(W.sub(0), ne_inner_c, 1))  # n_e (x_inner)

        # Variational forms for each of the three equations
        #
        # Second-order diffusion equation for n_e.
        # Starting strong form:  d/dx[g D ne'] = -n_e S_i (n_FC + n_CX).
        # Multiply by test function v_e and integrate by parts.  The IBP
        # in 1D gives a boundary contribution +[v_e g D ne'](x_inner)
        # -[v_e g D ne'](0).  v_e = 0 at any *Dirichlet* boundary, so:
        #
        #   * With Dirichlet at both ends (ne_inner_bc == "dirichlet"):
        #     both boundary terms vanish and the residual is
        #         int g D ne' v_e' dx  -  int n_e S_i (n_FC + n_CX) v_e dx = 0.
        #
        #   * With Neumann at x_inner (ne_inner_bc == "neumann"):
        #     v_e(x_inner) is not constrained, so we substitute the
        #     prescribed flux ne'(x_inner) = dne_dx_inner directly into
        #     the boundary integral.  The extra term is
        #         + g(x_inner) * D(x_inner) * dne_dx_inner * v_e(x_inner)
        #     which is written as ``g_fd * D_fd * dne_dx_inner_c * v_e * ds(1)``
        #     in Firedrake (ds(1) is the boundary measure at x_inner).
        #     The Dirichlet at x=0 still kills the boundary term at the
        #     right end.
        #
        # First-order ODE for <n_FC>.  We use the strong
        # residual form (no IBP) so the Galerkin solution converges to
        # the same profile that a forward marching solve from the inflow
        # boundary x = 0 would produce:
        #     int [ |V_FC| d/dx(f_FC g n_FC) - n_e (S_i + S_CX) n_FC ] v_F dx = 0.
        #
        # First-order ODE for <n_CX>, written symmetrically:
        #     int [ |V_CX| d/dx(f_CX g n_CX)
        #           - n_e ( S_i n_CX - (S_CX/2) n_FC ) ] v_C dx = 0,
        # ------------------------------------------------------------------
        if use_picard:
            # Linear Picard: one LinearVariationalProblem per (mesh, BC) layout;
            # lagged factors live in u_prev (updated each iteration).
            a_form, L_form = self._build_picard_bilinear_and_linear_forms(
                W, D_fd, g_fd, Si_fd, Scx_fd, Vcx_fd,
                VFC_const, fFC_const, fCX_const, half,
                u_prev, ne_inner_bc, dne_dx_inner_c,
            )
            picard_params = self._build_picard_linear_solver_parameters(
                linear_solver=linear_solver,
                ksp_rtol=ksp_rtol,
                ksp_max_it=ksp_max_it,
            )
            picard_solver = self._create_picard_linear_solver(
                a_form, L_form, u, bcs, picard_params,
            )

            rel = np.inf
            for k in range(max_picard):
                picard_solver.solve()

                ne_data_new  = u.subfunctions[0].dat.data
                ne_data_old  = u_prev.subfunctions[0].dat.data
                nFC_data_new = u.subfunctions[1].dat.data
                nFC_data_old = u_prev.subfunctions[1].dat.data
                nCX_data_new = u.subfunctions[2].dat.data
                nCX_data_old = u_prev.subfunctions[2].dat.data

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
                if v:
                    import warnings
                    warnings.warn(
                        f"Picard iteration did not reach tol={picard_tol:g} "
                        f"after {max_picard} iterations (last rel = {rel:.3e}).",
                        RuntimeWarning,
                    )
        else:
            # Full Newton via SNES on the nonlinear residual F(u)=0.
            v_e, v_F, v_C = TestFunctions(W)
            ne_curr, nFC_curr, nCX_curr = split(u)

            F1 = (g_fd * D_fd * ne_curr.dx(0) * v_e.dx(0)
                  - ne_curr * Si_fd * (nFC_curr + nCX_curr) * v_e) * dx
            F2 = ((VFC_const * fFC_const * g_fd * nFC_curr).dx(0) * v_F
                  - ne_curr * (Si_fd + Scx_fd) * nFC_curr * v_F) * dx
            F3 = ((Vcx_fd * fCX_const * g_fd * nCX_curr).dx(0) * v_C
                  - ne_curr * (
                        Si_fd * nCX_curr - half * Scx_fd * nFC_curr
                    ) * v_C) * dx
            if ne_inner_bc == "neumann":
                F1 = F1 + g_fd * D_fd * dne_dx_inner_c * v_e * ds(1)

            F = F1 + F2 + F3
            newton_params = self._build_petsc_solver_parameters(
                linear_solver=linear_solver,
                ksp_rtol=ksp_rtol,
                ksp_max_it=ksp_max_it,
            )
            solve(F == 0, u, bcs=bcs, solver_parameters=newton_params)


        # extract the converged profiles from the Firedrake solution and save/return them
        ne_fd, nFC_fd, nCX_fd = u.subfunctions

        sort_x_idx  = np.argsort(x_dofs)
        self.x_sol   = x_dofs[sort_x_idx]
        self.ne_sol  = ne_fd .dat.data[sort_x_idx]
        self.nFC_sol = nFC_fd.dat.data[sort_x_idx]
        self.nCX_sol = nCX_fd.dat.data[sort_x_idx]

        self.u_fd = u
        self.W_fd = W
        self.V_fd = V

        return self.x_sol, self.ne_sol, self.nFC_sol, self.nCX_sol
