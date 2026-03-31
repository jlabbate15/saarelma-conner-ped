import numpy as np
import scipy
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.optimize import fsolve
import matplotlib.pyplot as plt


class saarelma_connor:
    """

    Description
    ----------
    Creates instance of a tokamak pedestal configuration that the Saarelma-Connor (S. Saarelma et al 2024 Nucl. Fusion 64 076025) model can be applied to
    Dependencies:
    - juliacall (install with pip install juliacall) for EPEDNN interfacing
    - EPEDNN (install by git clone)
    - OpenFUSIONToolkit -> TokaMaker

    Uses COCOS 7 coordinate convention (same as TokaMaker) as defined by https://crppwww.epfl.ch/~sauter/cocos/Sauter_COCOS_Tokamak_Coordinate_Conventions.pdf

    Parameters
    ----------
    E_FC : float
        Energy of Franck-Condon neutrals as defined in Mahdavi M.A., Maingi R., Groebner R.J., Leonard A.W., Osborne T.H. and Porter G. 2003 Phys. Plasmas 10 3984 J
    Z_i : int
        Z of ions
    M_i : float
        Ion mass, kg what do you do for DT?????
    M_e : float
        Electron mass, kg
    sigma_i : float
        Ionization cross-section, m^2
    sigma_cx : float
        Charge-exchange cross-section, m^2
    P_tot_e : float
        Total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox, W
    alpha_crit : float
        FREE PARAMETER, Critical alpha value for onset of infinite-n ballooning instability, dimensionless
    C_KBM : float
        FREE PARAMETER, KBM diffusion coefficient, m^2/s CHECK UNITS?????
    De_chie_etg : float
        FREE PARAMETER, ETG diffusion coefficient, m^2/s CHECK UNITS?????
    mhd_loc : string
        Location of MHD equilibrium parameters, currently supporting: Tokamaker eqdsk
    kprof_loc : string
        Location of kinetic equilibrium parameters, currently supporting: p-file
    mhd_fp : string
        Filepath to mhd_loc-type file
    kprof_fp : string
        Filepath to kprof_loc-type file
    T_rat_flag : bool
        True if the temperature ratio is given, False if the temperature ratio is to be calculated
    pol_norm : bool
        True if the poloidal flux is normalized by 2pi, False if the poloidal flux is not normalized by 2pi
    verbose : bool
        True if verbose output is desired, False if verbose output is not desired
    species : string
        Species of ions, currently supporting: D, D-T
    """
    def __init__(
        self,
        E_FC = 3 * 1.60218e-19, # J,
        Z_i = 1, # Z of ions
        M_i = 3.344e-27, # kg, mass of deuterium nuclei
        M_e = 9.109e-31, # kg, mass of electron
        sigma_i = None, # m^2, ionization cross-section
        sigma_cx = None, # m^2, charge-exchange cross-section
        P_tot_e = None, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
        alpha_crit = None,
        C_KBM = None,
        De_chie_etg = None,
        mhd_loc = 'eqdsk', # location of MHD equilibrium parameters, currently supporting: Tokamaker eqdsk
        kprof_loc = 'p', # location of kinetic parameters, currently supporting: p-file
        mhd_fp = None, # filepath to MHD paramter file
        kprof_fp = None, # filepath to kinetic paramter file
        T_rat_flag = True,
        pol_norm = False, # True for when the poloidal flux is not normalized by 2pi
        verbose = False,
        species = 'D', # species of ions, currently supporting: D, D-T
    ):

        self.T_rat_flag = T_rat_flag
        self.verbose = verbose

        self.mhd_load(mhd_loc,mhd_fp) # load in MHD quantities
        self.kprof_load(kprof_loc,kprof_fp) # load in kinetic quantities

        if species == 'D':
            self.M_eff = 2.0
        elif species == 'D-T':
            self.M_eff = 2.5
        else:
            assert True, 'species must be D or D-T'

        self.Z_i = Z_i
        self.e_i = Z_i * 1.602e-19 # C
        self.V_th_i = np.sqrt(2*self.T_i/M_i) # m/s
        self.V_th_e = np.sqrt(2*self.T_e/M_e) # m/s

        self.S_i = sigma_i * self.V_th_e # m^3/s
        self.S_cx = sigma_cx * self.V_th_i # m^3/s

        self.V_FC = np.sqrt(8*E_FC/((np.pi^2) * M_i)) # m/s
        self.V_cx = np.sqrt(2*self.T_i/(np.pi * M_i)) # m/s

        # Diffusion coefficient setup
        c_s = self.fsa() # m/s, https://www.osti.gov/servlets/purl/4315023
        rho_s = self.fsa(  self.V_th_i*M_i / (self.e_i * self.B) ) # m
        mu0 = 4 * np.pi * 10**-7 # N/A**2, vacuum magnetic permeability constant
        alpha = (2 * np.gradient(self.V_plasma) / ((2*np.pi)**2)) * mu0 * np.gradient(self.P) * np.sqrt(self.V_plasma / (2*self.R0*np.pi**2))

        # Diffusion coefficient computation
        D_KBM = np.where(
            alpha > alpha_crit,
            C_KBM*(alpha-alpha_crit)*(c_s*rho_s**2)/self.a,
            0)
        D_ETG = De_chie_etg * P_tot_e / (self.S_plasma * np.gradient(self.T_e) ) # Need to be careful with gradient of T_e, make sure this is right
        D_NEO = 0.05 * (c_s * rho_s**2) / self.a
        self.D_ped = D_KBM + D_ETG + D_NEO

    def find_separatrix_points(eq, psi_tol=0.3):
        """Find X-points (if they exist) or the top/bottom extrema of the separatrix.

        Handles single-null, double-null, and limiter (no null) configurations.
        Uses 2D spline interpolation of psi(R,Z) to locate critical points of the
        poloidal flux, classifies them via the Hessian determinant, then selects
        the best X-point in each hemisphere (above/below magnetic axis) by ranking
        how close each saddle point's psi is to psi_boundary.

        Parameters
        ----------
        eq : dict
            Equilibrium dictionary returned by ``read_eqdsk``.  Must contain
            keys: ``nr``, ``nz``, ``rleft``, ``rdim``, ``zmid``, ``zdim``,
            ``psirz``, ``raxis``, ``zaxis``, ``psimag``, ``psibry``.
            Optionally ``rzout`` (boundary R,Z points).
        psi_tol : float
            Maximum allowed ``|psi_norm - 1|`` for an X-point to be considered
            a separatrix X-point at all.  Acts as a loose sanity-check; within
            each hemisphere the candidate closest to psi_boundary is always
            preferred.  Default 0.3.

        Returns
        -------
        result : dict
            ``'type'``     – ``'double_null'``, ``'single_null'``, or ``'limiter'``
            ``'xpoints'``  – list of ``(R, Z)`` tuples for separatrix X-points
            ``'top'``      – ``(R, Z)`` of the upper X-point or upper separatrix extremum
            ``'bottom'``   – ``(R, Z)`` of the lower X-point or lower separatrix extremum
        """
        nr = eq['nr']
        nz = eq['nz']
        r = np.linspace(eq['rleft'], eq['rleft'] + eq['rdim'], nr)
        z = np.linspace(eq['zmid'] - eq['zdim'], eq['zmid'] + eq['zdim'], nz)
        psi = eq['psirz']
        zaxis = eq['zaxis']
        psimag = eq['psimag']
        psibry = eq['psibry']

        # ---- 2-D spline interpolation  (psi has shape (nz, nr) -> x=z, y=r) ----
        spl = RectBivariateSpline(z, r, psi)

        def _grad(point):
            """Return [dpsi/dR, dpsi/dZ] at a single (R, Z) point."""
            R, Z = point
            dR = float(spl(Z, R, dx=0, dy=1, grid=False))
            dZ = float(spl(Z, R, dx=1, dy=0, grid=False))
            return [dR, dZ]

        # ---- locate candidate critical points via sign changes on the grid ----
        dpsidR = spl(z, r, dx=0, dy=1)   # shape (nz, nr)
        dpsidZ = spl(z, r, dx=1, dy=0)

        candidates = []
        for iz in range(nz - 1):
            for ir in range(nr - 1):
                corners_dR = [dpsidR[iz, ir], dpsidR[iz, ir+1],
                            dpsidR[iz+1, ir], dpsidR[iz+1, ir+1]]
                corners_dZ = [dpsidZ[iz, ir], dpsidZ[iz, ir+1],
                            dpsidZ[iz+1, ir], dpsidZ[iz+1, ir+1]]
                signs_dR = set(np.sign(v) for v in corners_dR if v != 0.0)
                signs_dZ = set(np.sign(v) for v in corners_dZ if v != 0.0)
                if len(signs_dR) > 1 and len(signs_dZ) > 1:
                    candidates.append((0.5*(r[ir]+r[ir+1]),
                                    0.5*(z[iz]+z[iz+1])))

        # ---- refine & classify each candidate ----
        xpoints = []
        opoints = []
        found = []

        for R0, Z0 in candidates:
            try:
                sol, _, ier, _ = fsolve(_grad, [R0, Z0], full_output=True)
            except Exception:
                continue
            if ier != 1:
                continue
            Rs, Zs = sol
            if Rs < r[0] or Rs > r[-1] or Zs < z[0] or Zs > z[-1]:
                continue
            residual = _grad(sol)
            if abs(residual[0]) > 1e-6 or abs(residual[1]) > 1e-6:
                continue
            if any(abs(Rs - Rf) < 0.01 and abs(Zs - Zf) < 0.01
                for Rf, Zf in found):
                continue
            found.append((Rs, Zs))

            # Hessian determinant distinguishes saddle (X) from extremum (O)
            d2R2 = float(spl(Zs, Rs, dx=0, dy=2, grid=False))
            d2Z2 = float(spl(Zs, Rs, dx=2, dy=0, grid=False))
            d2RZ = float(spl(Zs, Rs, dx=1, dy=1, grid=False))
            det_H = d2R2 * d2Z2 - d2RZ**2

            if det_H < 0:
                xpoints.append((Rs, Zs))
            else:
                opoints.append((Rs, Zs))

        # ---- rank X-points by proximity to psi_boundary, per hemisphere ----
        psi_range = abs(psibry - psimag)
        upper_xpts = []   # (R, Z, psi_error)  for Z > zaxis
        lower_xpts = []   # (R, Z, psi_error)  for Z < zaxis

        for Rx, Zx in xpoints:
            psi_norm = abs(float(spl(Zx, Rx, grid=False)) - psimag) / psi_range
            psi_err = abs(psi_norm - 1.0)
            if psi_err > psi_tol:
                continue
            if Zx > zaxis:
                upper_xpts.append((Rx, Zx, psi_err))
            else:
                lower_xpts.append((Rx, Zx, psi_err))

        best_upper = min(upper_xpts, key=lambda x: x[2]) if upper_xpts else None
        best_lower = min(lower_xpts, key=lambda x: x[2]) if lower_xpts else None

        sep_xpoints = []
        if best_upper is not None:
            sep_xpoints.append((best_upper[0], best_upper[1]))
        if best_lower is not None:
            sep_xpoints.append((best_lower[0], best_lower[1]))

        # ---- helper: extrema from boundary contour or rzout ----
        def _boundary_extrema():
            """Return top, bottom, outboard, inboard (R,Z) extrema of the separatrix."""
            if 'rzout' in eq and eq['rzout'] is not None and len(eq['rzout']) > 0:
                bdy = eq['rzout']
            else:
                import matplotlib
                matplotlib.use('Agg')
                fig, ax = plt.subplots()
                cs = ax.contour(r, z, psi, levels=[psibry])
                bdy = np.vstack(cs.allsegs[0])
                plt.close(fig)
            itop = np.argmax(bdy[:, 1])
            ibot = np.argmin(bdy[:, 1])
            iout = np.argmax(bdy[:, 0])
            iin  = np.argmin(bdy[:, 0])
            top      = (bdy[itop, 0], bdy[itop, 1])
            bottom   = (bdy[ibot, 0], bdy[ibot, 1])
            outboard = (bdy[iout, 0], bdy[iout, 1])
            inboard  = (bdy[iin,  0], bdy[iin,  1])
            return top, bottom, outboard, inboard

        # ---- assemble result ----
        result = {'xpoints': sep_xpoints, 'opoints': opoints}
        top_bdy, bot_bdy, out_bdy, in_bdy = _boundary_extrema()
        result['outboard'] = out_bdy
        result['inboard']  = in_bdy

        if len(sep_xpoints) == 0:
            result['type'] = 'limiter'
            result['top'] = top_bdy
            result['bottom'] = bot_bdy

        elif len(sep_xpoints) == 1:
            result['type'] = 'single_null'
            xpt = sep_xpoints[0]
            if xpt[1] > zaxis:
                result['top'] = xpt
                result['bottom'] = bot_bdy
            else:
                result['bottom'] = xpt
                result['top'] = top_bdy

        else:
            result['type'] = 'double_null'
            sep_xpoints.sort(key=lambda p: p[1])
            result['bottom'] = sep_xpoints[0]
            result['top'] = sep_xpoints[-1]

        return result

    def plasma_surface_area_and_volume(self,eq):
        """Compute the surface area and volume of the plasma bounded by the separatrix.

        Treats the boundary as a closed polygon in (R, Z) revolved around the
        Z-axis (toroidal symmetry).  Uses eq['rzout'] as the boundary points.

        Uses Pappus' theorem to compute the surface area of the plasma.
        Takes exact integral of pi*R^2 dZ for piecewise-linear boundary segments to compute the volume of the plasma.

        Parameters
        ----------
        eq : dict
            Equilibrium dictionary from read_eqdsk (must contain 'rzout').

        """
        bdy = eq['rzout']
        R = bdy[:, 0]
        Z = bdy[:, 1]

        # Close the polygon if the first and last points don't match
        if not (np.isclose(R[0], R[-1]) and np.isclose(Z[0], Z[-1])):
            R = np.append(R, R[0])
            Z = np.append(Z, Z[0])

        dZ = np.diff(Z)
        dR = np.diff(R)

        R_i  = R[:-1]
        R_ip = R[1:]

        # Poloidal cross-section area  (shoelace formula - general to any polygon)
        # cross_section_area = 0.5 * abs(np.sum(R_i * Z[1:] - R_ip * Z[:-1]))

        # Toroidal volume:  V = (pi/3) |sum (Z_{i+1}-Z_i)(R_i^2 + R_i*R_{i+1} + R_{i+1}^2)|
        # Exact integral of pi*R^2 dZ for piecewise-linear boundary segments
        volume = (np.pi / 3.0) * abs(np.sum(dZ * (R_i**2 + R_i * R_ip + R_ip**2)))

        # Toroidal surface area:  S = 2*pi * sum  R_avg * dl   (Pappus' theorem)
        dl = np.sqrt(dR**2 + dZ**2)
        surface_area = 2.0 * np.pi * np.sum(0.5 * (R_i + R_ip) * dl)

        self.S_plasma = surface_area # m^2, total surface area of plasma
        self.V_plasma = volume # m^3, volume enclosed by the plasma per poloidal flux

    def calc_B(self,eq,R_eval,Z_eval):
        """Calculate magnetic field at some point in the plasma
            Always use (rho,theta,var_zeta) coordinate convention as defined by https://crppwww.epfl.ch/~sauter/cocos/Sauter_COCOS_Tokamak_Coordinate_Conventions.pdf

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        eq : dict
            Equilibrium dictionary from read_eqdsk
        R_eval : float or array
            radial location at which to evaluate the magnetic field
        Z_eval : float or array
            vertical location at which to evaluate the magnetic field
        """

        F = eq['fpol']
        B = np.array([0,0,0])

        sigma_Bp = 1 if self.helicity > 0 else -1 # does this change sign in a single equilibrium?
        e_Bp = 1 if self.pol_norm else 0

        r = self.rgrid
        z = self.zgrid
        spl = RectBivariateSpline(z, r, eq['psirz'])
        dpsi_dR = spl(Z_eval, R_eval, dx=0, dy=1, grid=False)
        dpsi_dZ = spl(Z_eval, R_eval, dx=1, dy=0, grid=False)
        B[0] = (sigma_Bp / ((2*np.pi)**e_Bp)) * dpsi_dZ / R_eval # R component of the magnetic field
        B[1] = (sigma_Bp / ((2*np.pi)**e_Bp)) * -dpsi_dR / R_eval # Z component of the magnetic field
        B[2] = (F / self.rgrid) # T, toroidal magnetic field

        self.B = B # T, total magnetic field (vector)

    def mhd_load(self,mhd_loc,fp):
        """Load MHD equilibrium parameters using method specified by mhd_eq_loc flag. 
        Parameters that will be loaded include: R0, a, r_sep(theta), r, P(psi), psi_pol(r)
        Calculates: Plasma volume, plasma surface area
        
        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        mhd_loc : string
            which method to use to load MHD parameters.
        fp : string
            filepath to file with MHD parameters.
             
        """

        if mhd_loc == 'eqdsk':
            from OpenFUSIONToolkit.TokaMaker.util import read_eqdsk
            self.eq = read_eqdsk(fp)

            # Find separatrix points
            sep = self.find_separatrix_points(self.eq)
            if self.verbose:
                print(f"Configuration type: {sep['type']}")
                print(f"Top    (R, Z) = ({sep['top'][0]:.4f}, {sep['top'][1]:.4f})")
                print(f"Bottom (R, Z) = ({sep['bottom'][0]:.4f}, {sep['bottom'][1]:.4f})")
                if sep['xpoints']:
                    for i, xpt in enumerate(sep['xpoints']):
                        print(f"X-point {i+1}: (R, Z) = ({xpt[0]:.4f}, {xpt[1]:.4f})")

                r_grid = np.linspace(eq["rleft"], eq["rleft"] + eq["rdim"], eq["nr"])
                z_grid = np.linspace(eq["zmid"] - eq["zdim"], eq["zmid"] + eq["zdim"], eq["nz"])

                fig, ax = plt.subplots()
                ax.contour(r_grid, z_grid, eq["psirz"], levels=30, colors='gray', linewidths=0.5)
                ax.contour(r_grid, z_grid, eq["psirz"], levels=[eq['psibry']], colors='blue', linewidths=1.5)
                ax.plot(eq['raxis'], eq['zaxis'], 'go', markersize=8, label='O-point (axis)')

                ax.plot(*sep['top'], 'rv', markersize=10, label=f"Top ({sep['type']})")
                ax.plot(*sep['bottom'], 'r^', markersize=10, label=f"Bottom ({sep['type']})")

                for xpt in sep['xpoints']:
                    ax.plot(*xpt, 'rx', markersize=12, markeredgewidth=2)

                ax.plot(sep['outboard'][0], sep['outboard'][1], 'rx', markersize=10, label='Outboard')
                ax.plot(sep['inboard'][0], sep['inboard'][1], 'gx', markersize=10, label='Inboard')

                ax.set_xlabel('R (m)')
                ax.set_ylabel('Z (m)')
                ax.set_aspect('equal')
                ax.legend()
                plt.title(f"Separatrix analysis: {sep['type']}")
                plt.show()
            rmax_top = sep['top'][0]
            rmax_bottom = sep['bottom'][0]
            zmax_top = sep['top'][1]
            zmax_bottom = sep['bottom'][1]
            rmax_outboard = sep['outboard'][0]
            rmax_inboard = sep['inboard'][0]
            # zmax_outboard = sep['outboard'][1]
            # zmax_inboard = sep['inboard'][1]

            # Geometric parameters
            self.Raxis = self.eq['raxis'] # m, location of magnetic axis relative to device rotational line of toroidal symmetry
            self.Rmajor = (rmax_outboard + rmax_inboard) / 2 # m
            self.a = (rmax_outboard - rmax_inboard) / 2 # minor radius # m
            delta_u = (self.Rmajor - rmax_top) / self.a
            delta_l = (self.Rmajor - rmax_bottom) / self.a
            self.delta = (delta_u + delta_l) / 2 # dimensionless, total triangularity
            self.kappa = (zmax_top - zmax_bottom) / (2*self.a) # dimensionless, elongation
            self.plasma_surface_area_and_volume(self.eq)

            # Plasma parameters
            self.Ip = self.eq['ip'] # MA, Plasma current
            self.Bt = self.eq['bcentr'] # T, Toroidal magnetic field at the center of the plasma (in vacuum field)
            self.Ppsi = self.eq['pres'] # pressure evaluated at each psi_N
            
            # Grids
            self.rgrid = np.linspace(self.eq['rleft'],self.eq['rleft']+self.eq['rdim'],self.eq['nr']) # m, 1D R grid
            self.zgrid = np.linspace(self.eq['zmid']-self.eq['zdim'],self.eq['zmid']+self.eq['zdim'],self.eq['nz']) # m, 1D Z grid
            self.psi_RZ = self.eq['psirz'] # 2D poloidal flux array at each RZ grid point

    def kprof_load(self,kprof_loc='p',kprof_fp=None):
        """Load kinetic equilibrium parameters using method specified by kprof_loc flag. 
        Parameters that will be loaded include: T_e, n_e
        Calculates: dn_e/dx|x=-inf, T_i
        
        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        kprof_loc : string
            which method to use to load kinetic parameters.
        kprof_fp : string
            filepath to kprof_loc-type file with kinetic parameters.
             
        """

        if kprof_loc == 'p':

            def read_pfile(path):
                data = {}
                key = ''
                with open(path) as f:
                    for line in f:
                        if '3 N Z A' in line:
                            break
                        if line.startswith('201'):
                            key = line.split()[2]
                            data[key] = np.array([])
                            data[f'{key}_psi'] = np.array([])
                        else:
                            psi, dat, _ = line.split()
                            psi = float(psi)
                            dat = float(dat)
                            data[key] = np.append(data[key], dat)
                            data[f'{key}_psi'] = np.append(data[f'{key}_psi'], psi)
                return data

            # Extract profiles
            pf = read_pfile(kprof_fp)
            self.n_e = pf['ne(10^20/m^3)'] # n_e values (10^20/m^3) evaluated at psi_ne_eval
            self.T_e = pf['te(KeV)'] # T_e values (KeV) evaluated at psi_Te_eval
            self.psi_ne_eval = pf['ne(10^20/m^3)_psi'] # psi_N values at which n_e is evaluated
            self.psi_Te_eval = pf['te(KeV)_psi'] # psi_N values at which T_e is evaluated

            # Calculate boundary condition from profiles at psi_N = 0.85
            self.dne_dx = np.gradient(self.n_e, self.psi_ne_eval) # (particles/m^3) / m, electron density gradient

            # Create the interpolation function
            dne_dx_interp = interp1d(self.psi_ne_eval, self.dne_dx, kind='linear')
            self.dne_dx_neginf = dne_dx_interp(0.85) # hard-coded to psi_N = 0.85, would love to change to a better boundary condition
        else:
            assert True, 'kprof_loc method not supported'

        if self.T_rat_flag:
            self.T_i = self.T_e * self.T_rat

    def first_step(self):
        """Solve Equation (16) in S. Saarelma et al 2023 Nucl. Fusion 63 052002

        Parameters
        ----------
        parm : bool, optional
            info.
             
        """   

    def solve(self):
        """Iteratively solve Equation (15) in S. Saarelma et al 2023 Nucl. Fusion 63 052002

        Parameters
        ----------
        parm : bool, optional
            info.
             
        """

        f_FC = form_factor('FC')
        f_cx = form_factor('cx')

        step_one = first_step()

        self.neped =  # solution of SC model

    def form_factor(self,type = 'ex'):
        assert type != 'FC' or type != 'cx', 'form factor must be for FC or cx'


    def fsa(self,A,theta,R):
        """Flux surface average a quantity as defined by ⟨A⟩= int(R^2Adθ)/ int(R^22dθ) in S. Saarelma et al 2023 Nucl. Fusion 63 052002

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        A : array
            any array with an axis spanning the theta direction.
        theta: array
            array of theta values at which A is evaluated.
        R: array
            value of R at each theta along magnetic surface of interest.
             
        """

        # Interpolate to theta=0 and theta=2*pi if theta doesn't go to 0,2*np.pi
        if 0 not in theta or 2*np.pi not in theta:
            spline_func = scipy.interpolate.CubicSpline(theta, A, bc_type='natural')
            A_at_0 = spline_func(0)
            A_at_2pi = spline_func(2 * np.pi)
            theta = np.insert(theta, 0, 0.0) # Add 0 at the beginning
            theta = np.append(theta, 2 * np.pi) # Add 2*pi at the end
            A = np.insert(A, 0, A_at_0) # Add f(0) at the beginning
            A = np.append(A, A_at_2pi) # Add f(2*pi) at the end

        num = scipy.integrate.simpson(y=A*R**2, x=theta)
        den = scipy.integrate.simpson(y=R**2, x=theta)
        
        return num / den
    
    def feed_eped(self):
        """Once a pedestal density profile is calculated, feed this profile to EPED and run EPED or EPEDNN to determine pedestal pressure height and width

        Parameters
        ----------
        self : object
            instance of saarelma_connor class

        Returns
        -------
        pedestal_pressure : float
            Pedestal pressure (MPa)
        pedestal_width : float
            Pedestal width (normalized poloidal flux)
        """

        # Requires dependency "juliacall" to translate Python inputs to FUSE EPED.jl
        # Requires dependency EPEDNN

        from juliacall import Main as jl

        # 1. Load the Julia EPEDNN module (Assuming EPEDNN is already installed in your Julia environment)
        '''
            # To install EPEDNN, run the following command in your terminal:
            conda activate sc_ped
            cd /Users/nelsonlab/codes/saarelma-conner-ped
            julia

            # if you need to install julia, run the following command in your terminal:
            curl -fsSL https://install.julialang.org | sh
            # then restart terminal
            julia --version
            # if this doesn't work, you could try the following although I did not verify this works:
            echo 'export PATH="$HOME/.juliaup/bin:$PATH"' >> ~/.zshrc
            source ~/.zshrc
            julia --version

            # Then in Julia:
            using Pkg
            Pkg.activate(".")  # optional but recommended: use this repo as the active Julia project
            Pkg.develop(path="dependencies/EPEDNN.jl")
            Pkg.instantiate()

            #Then in Julia: 
            using EPEDNN
        '''
        # you can run Julia commands in Python using jl.seval('command')
        jl.seval('using Pkg')
        jl.seval('Pkg.activate(".")  # optional but recommended: use this repo as the active Julia project')
        jl.seval('Pkg.develop(path="../dependencies/EPEDNN.jl")')
        jl.seval('Pkg.instantiate()')
        jl.seval('using EPEDNN')

        # 2. Load the pre-trained EPED neural network model
        # This mimics the EPEDNN.loadmodelonce("EPED1NNmodel.bson") step
        model_filename = "EPED1NNmodel.bson" 
        eped_model = jl.EPEDNN.loadmodelonce(model_filename)

        # 3. Define your inputs in Python
        # These map exactly to the InputEPED struct we saw in the Julia code
        inputs = {
            "a": self.a,           # Minor radius (m)
            "betan": self.betan,       # Normalized beta
            "bt": self.Bt,          # Toroidal magnetic field (T)
            "delta": self.delta,       # Effective triangularity
            "ip": self.Ip,          # Plasma current (MA)
            "kappa": self.kappa,       # Elongation
            "m": self.M_eff,           # Effective mass (must be 2.0 for D or 2.5 for D-T)
            "neped": self.neped,       # Pedestal density (in 10^19 m^-3)
            "r": self.R0,           # Major radius (m)
            "zeffped": self.Z_i      # Effective charge
        }

        # 4. Call the Julia model using the Python inputs
        # We pass the inputs into the Julia function, along with the keyword arguments
        solution = eped_model(
            inputs["a"], 
            inputs["betan"], 
            inputs["bt"], 
            inputs["delta"],
            inputs["ip"], 
            inputs["kappa"], 
            inputs["m"], 
            inputs["neped"],
            inputs["r"], 
            inputs["zeffped"],
            only_powerlaw=False,        # Set to True if you only want the scaling law
            warn_nn_train_bounds=True   # Warns if inputs are outside the training data
        )

        # 5. Extract the results back into Python
        # The solution structure has pressure and width for different modes (GH, G, H)
        self.pedestal_pressure = solution.pressure.GH.H  # in MPa
        self.pedestal_width = solution.width.GH.H        # in normalized poloidal flux

        return {"pedestal_pressure": self.pedestal_pressure, "pedestal_width": self.pedestal_width}


    
