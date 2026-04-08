# from typing import Self
import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
from scipy.integrate import simpson, solve_bvp, cumulative_trapezoid
import matplotlib.pyplot as plt
from adas.adas_ionisation import scd_adas


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
        Proton mass, kg
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
    ne_x0 : float
        m^-3, electron density at the separatrix (boundary condition)
    dne_dx0 : float
        m^-3, gradient of electron density at the separatrix (boundary condition)
    nFC_x0 : float
        m^-3, Franck-Condon neutral density at the separatrix (boundary condition)
    psi_N_inner_boundary : float
        normalized poloidal flux at the inner boundary (boundary condition)
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
    T_rat : float
        Temperature ratio between ions and electrons, dimensionless
        Ignored if T_rat_flag is False
        Default is 1
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
        M_i = 1.673e-27, # kg, mass of hydrogen nuclei
        M_e = 9.109e-31, # kg, mass of electron
        sigma_i = None, # m^2, ionization cross-section
        sigma_cx = None, # m^2, charge-exchange cross-section
        P_tot_e = None, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
        alpha_crit = None,
        C_KBM = None,
        De_chie_etg = None,
        ne_x0 = 1e19, # m^-3,electron density at the separatrix
        dne_dx0 = 1e-5, # m^-3, gradient of electron density at the separatrix
        nFC_x0 = 0.1e19, # m^-3, Franck-Condon neutral density at the separatrix
        psi_N_inner_boundary = 0.85, # normalized poloidal flux at the inner boundary (boundary condition)
        mhd_loc = 'eqdsk', # location of MHD equilibrium parameters, currently supporting: Tokamaker eqdsk
        kprof_loc = 'p', # location of kinetic parameters, currently supporting: p-file
        mhd_fp = None, # filepath to MHD paramter file
        kprof_fp = None, # filepath to kinetic paramter file
        T_rat_flag = True,
        T_rat = 1,
        pol_norm = False, # True for when the poloidal flux is not normalized by 2pi. COCOS 7 convention is pol_norm=False, so poloidal flux is normalized by 2pi
        verbose = False,
        species = 'D', # species of ions, currently supporting: D, D-T
    ):

        self.T_rat_flag = T_rat_flag
        self.T_rat = T_rat
        self.verbose = verbose
        self.pol_norm = pol_norm
        self.psi_N_inner_boundary = psi_N_inner_boundary

        self.M_i = M_i
        if species == 'D':
            self.M_eff = 2.0
        elif species == 'D-T':
            self.M_eff = 2.5
        else:
            assert False, 'species must be D or D-T'

        self.mhd_load(mhd_loc,mhd_fp) # load in MHD quantities
        self.kprof_load(kprof_loc,kprof_fp) # load in kinetic quantities
        self.calc_B(self.rgrid,self.zgrid) # calculate the magnetic field at each RZ grid point, sets self.B

        self.Z_i = Z_i
        self.e_i = Z_i * 1.602e-19 # C
        self.V_th_i = np.sqrt(2*self.T_i/(M_i*self.M_eff)) # m/s, per psi_N_eval for Ti
        self.V_th_e = np.sqrt(2*self.T_e/M_e) # m/s, per psi_N_eval for Te

        self.V_FC = np.sqrt(8*E_FC/((np.pi**2) * M_i*self.M_eff)) # m/s
        self.V_cx = np.sqrt(2*self.T_i/(np.pi * M_i*self.M_eff)) # m/s, per psi_N_eval for Ti

        self.cross_sections(species) # load in cross-sections

        self.S_i = self.sigma_i * self.V_th_e # m^3/s, per psi_N_eval for Te
        self.S_cx = self.sigma_cx * self.V_th_i # m^3/s, per psi_N_eval for Te

        # Diffusion coefficient setup
        c_s = (self.e_i * self.T_e / (M_i * self.M_eff)) ** 0.5 # m/s, cs = (e*T_e/mD)^1/2 as defined in W. Guttenfelder et al 2021 Nucl. Fusion 61 056005
        V_th_i_rz = self.psi_rz_expand(self.V_th_i, psi_N_A='T_e')
        rho_s = V_th_i_rz*M_i*self.M_eff / (self.e_i * self.B) # m, known on each RZ grid point
        rho_s = self.fsa(rho_s,flux_surfaces='T_e') # m, known on each flux surface
        self.mu0 = 4 * np.pi * 10**-7 # N/A**2, vacuum magnetic permeability constant
        alpha = (2 * np.gradient(self.V_plasma) / ((2*np.pi)**2)) * self.mu0 * np.gradient(self.pres) * np.sqrt(self.V_plasma / (2*self.Rmajor*np.pi**2)) # evaluated at each psi_N = np.linspace(eq['psimag'], eq['psibry'], len(self.pres))

        # Interpolate T_e, c_s, rho_s from psi_Te_eval onto the pressure psi_N grid
        T_e_pres = interp1d(self.psi_Te_eval, self.T_e, kind='linear',
                            bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        c_s = interp1d(self.psi_Te_eval, c_s, kind='linear',
                       bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        rho_s = interp1d(self.psi_Te_eval, rho_s, kind='linear',
                         bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)

        # Diffusion coefficient computation
        D_KBM = np.where(
            alpha > alpha_crit,
            C_KBM*(alpha-alpha_crit)*(c_s*rho_s**2)/self.a,
            0)
        D_ETG = De_chie_etg * P_tot_e / (self.S_plasma[-1] * abs(np.gradient(T_e_pres,self.psi_N_pres)) ) # evaluated at each psi_N_pres
        D_NEO = 0.05 * (c_s * rho_s**2) / self.a
        self.D_ped = D_KBM + D_ETG + D_NEO

        # boundary conditions - will be updated with a more comprehensive model in the future
        self.ne_x0 = ne_x0
        self.dne_dx0 = dne_dx0
        self.nFC_x0 = nFC_x0

    def find_boundary_points(self,eq):
        """Find the top/bottom/inboard/outboard extrema of the separatrix.

        Parameters
        ----------
        eq : dict
            Equilibrium dictionary returned by ``read_eqdsk``.  Must contain
            keys: ``nr``, ``nz``, ``rleft``, ``rdim``, ``zmid``, ``zdim``,
            ``psirz``, ``raxis``, ``zaxis``, ``psimag``, ``psibry``.
            Optionally ``rzout`` (boundary R,Z points).

        Returns
        -------
        result : dict
            ``'top'``      – ``(R, Z)`` of the upper boundary point
            ``'bottom'``   – ``(R, Z)`` of the lower boundary point
            ``'outboard'``  – ``(R, Z)`` of the outboard boundary point
            ``'inboard'``   – ``(R, Z)`` of the inboard boundary point
        """

        psi = eq['psirz']
        psibry = eq['psibry']

        if 'rzout' in eq and eq['rzout'] is not None and len(eq['rzout']) > 0:
            bdy = eq['rzout']
        else:
            nr = eq['nr']
            nz = eq['nz']
            r = np.linspace(eq['rleft'], eq['rleft'] + eq['rdim'], nr)
            z = np.linspace(eq['zmid'] - eq['zdim']/2, eq['zmid'] + eq['zdim']/2, nz)
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

        return {
            'top': top,
            'bottom': bottom,
            'outboard': outboard,
            'inboard': inboard,
        }

    def plasma_surface_area_and_volume(self):
        """Compute the plasma surface area and enclosed volume at each flux surface.

        For each psi in np.linspace(psimag, psibry, len(self.pres)), extracts
        the flux surface contour from the 2D psi grid, then computes the
        toroidal surface area (Pappus' theorem) and volume (exact
        piecewise-linear revolution integral) of the surface of revolution.

        Parameters
        ----------
        self : object
            instance of saarelma_connor class

        Sets
        ----
        self.S_plasma : ndarray, shape (n_psi,)
            Toroidal surface area (m^2) at each flux surface.
        self.V_plasma : ndarray, shape (n_psi,)
            Enclosed toroidal volume (m^3) at each flux surface.
        """

        n_psi = len(self.pres)

        self.S_plasma = np.zeros(n_psi)
        self.V_plasma = np.zeros(n_psi)

        # Extract the flux surface contour from the 2D psi grid
        fig, ax = plt.subplots()
        for i in range(n_psi):
            ax.cla()
            cs = ax.contour(self.rgrid, self.zgrid, self.psi_RZ,
                            levels=[self.psi_pres[i]])

            segs = cs.allsegs[0]
            if not segs:
                self.S_plasma[i] = np.nan
                self.V_plasma[i] = np.nan
                continue

            # Longest contour = the real flux surface, not islands
            seg = max(segs, key=lambda s: len(s))
            R = seg[:, 0]
            Z = seg[:, 1]

            # Close the contour so the integral spans a full 2*pi
            if not (np.isclose(R[0], R[-1]) and np.isclose(Z[0], Z[-1])):
                R = np.append(R, R[0])
                Z = np.append(Z, Z[0])

            dZ = np.diff(Z)
            dR = np.diff(R)
            R_i  = R[:-1]
            R_ip = R[1:]

            # Toroidal volume:  V = (pi/3) |sum (Z_{i+1}-Z_i)(R_i^2 + R_i*R_{i+1} + R_{i+1}^2)|
            # Exact integral of pi*R^2 dZ for piecewise-linear boundary segments
            self.V_plasma[i] = (np.pi / 3.0) * abs(np.sum(dZ * (R_i**2 + R_i * R_ip + R_ip**2))) # m^3, volume enclosed by the plasma per poloidal flux

            # Poloidal cross-section area: Shoelace formula - general to any polygon (Pappus' theorem)
            dl = np.sqrt(dR**2 + dZ**2)
            self.S_plasma[i] = 2.0 * np.pi * np.sum(0.5 * (R_i + R_ip) * dl) # m^2, total surface area of plasma

        plt.close(fig)

    def calc_B(self,R_eval,Z_eval):
        """Calculate magnetic field at some point in the plasma
            Always use (rho,theta,var_zeta) coordinate convention as defined by https://crppwww.epfl.ch/~sauter/cocos/Sauter_COCOS_Tokamak_Coordinate_Conventions.pdf

           Note: sigma_Bb is not important for this model, we will always use sigma_Bp=1.
        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        R_eval : float or array
            radial location at which to evaluate the magnetic field
        Z_eval : float or array
            vertical location at which to evaluate the magnetic field
        """

        F = self.eq['fpol']
        psi_F = np.linspace(self.eq['psimag'], self.eq['psibry'], len(F))

        e_Bp = 1 if self.pol_norm else 0

        r = self.rgrid
        z = self.zgrid
        spl = RectBivariateSpline(z, r, self.eq['psirz'])

        R_eval_arr = np.atleast_1d(R_eval)
        Z_eval_arr = np.atleast_1d(Z_eval)

        psi = spl(Z_eval_arr, R_eval_arr, grid=False)
        dpsi_dR = spl(Z_eval_arr, R_eval_arr, dx=0, dy=1, grid=False) # specifying dx, dy specifies the derivative order in the respective direction
        dpsi_dZ = spl(Z_eval_arr, R_eval_arr, dx=1, dy=0, grid=False)
        F_interp = interp1d(psi_F, F, kind='linear', bounds_error=False, fill_value="extrapolate")
        B_R = (1 / ((2*np.pi)**e_Bp)) * dpsi_dZ / R_eval_arr # R component of the magnetic field
        B_Z = (1 / ((2*np.pi)**e_Bp)) * -dpsi_dR / R_eval_arr # Z component of the magnetic field
        B_phi = (F_interp(psi) / R_eval_arr) # T, toroidal magnetic field

        self.B = np.sqrt(B_R**2 + B_Z**2 + B_phi**2) # T, total magnetic field at each R_eval, Z_eval
        return self.B, [B_R, B_Z, B_phi]

    def mhd_load(self,mhd_loc,fp):
        """Load and calculate various MHD equilibrium parameters using method specified by mhd_eq_loc flag. 
        
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

            bdry = self.find_boundary_points(eq=self.eq)

            rmax_top = bdry['top'][0]
            rmax_bottom = bdry['bottom'][0]
            zmax_top = bdry['top'][1]
            zmax_bottom = bdry['bottom'][1]
            rmax_outboard = bdry['outboard'][0]
            rmax_inboard = bdry['inboard'][0]
            z_outboard = bdry['outboard'][1]
            # zmax_inboard = bdry['inboard'][1]

            # Geometric parameters
            self.Raxis = self.eq['raxis'] # m, location of magnetic axis relative to device rotational line of toroidal symmetry
            self.Rmajor = (rmax_outboard + rmax_inboard) / 2 # m
            self.a = (rmax_outboard - rmax_inboard) / 2 # minor radius # m
            delta_u = (self.Rmajor - rmax_top) / self.a
            delta_l = (self.Rmajor - rmax_bottom) / self.a
            self.delta = (delta_u + delta_l) / 2 # dimensionless, total triangularity
            self.kappa = (zmax_top - zmax_bottom) / (2*self.a) # dimensionless, elongation

            # Plasma parameters (skip the magnetic axis to avoid degenerate zero-area/volume flux surface)
            self.Ip = self.eq['ip'] # MA, Plasma current
            self.pres = self.eq['pres'][1:]
            self.psi_pres = np.linspace(self.eq['psimag'], self.eq['psibry'], len(self.eq['pres']))[1:]
            self.psi_N_pres = (self.psi_pres - self.eq['psimag']) / (self.eq['psibry'] - self.eq['psimag'])

            # Grids
            self.rgrid = np.linspace(self.eq['rleft'],self.eq['rleft']+self.eq['rdim'],self.eq['nr']) # m, 1D R grid
            self.zgrid = np.linspace(self.eq['zmid']-self.eq['zdim']/2,self.eq['zmid']+self.eq['zdim']/2,self.eq['nz']) # m, 1D Z grid
            self.psi_RZ = self.eq['psirz'] # 2D poloidal flux array at each RZ grid point
            self.psi_RZ_N = (self.psi_RZ - self.eq['psimag']) / (self.eq['psibry'] - self.eq['psimag']) # normalized poloidal flux at each RZ grid point
            # self.rsep_mid = (((rmax_outboard - self.Raxis)**2) + ((z_outboard - self.eq['zaxis'])**2))**5 # separatrix radius at midplane

            self.plasma_surface_area_and_volume()

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
            self.dne_dx_neginf = dne_dx_interp(self.psi_N_inner_boundary) # hard-coded to psi_N = 0.85, would love to change to a better boundary condition
        else:
            assert False, 'kprof_loc method not supported'

        if self.T_rat_flag:
            self.T_i = self.T_e * self.T_rat

    def cross_sections(self,species='D'):
        """Calculate the cross-sections for the ionization and charge-exchange cross-sections
        Uses ADAS ADF01 qcx#h0_ex3#h1.dat to interpolate the charge-exchange cross-sections as a function of energy
        Uses ADAS ADF23  to interpolate the ionization cross-sections as a function of energy
        This is for deuterium only

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        species : string
            species of ions, currently supporting: D
             
        """     

        if species == 'D':
            
            # charge-exchange cross-section
            sigma_cx_perE = np.array([3.81*10**(-18), 3.85*10**(-18), 3.44*10**(-18), 2.71*10**(-18), 1.74*10**(-18), 8.10*10**(-20), 9.56*10**(-22), 1.46*10**(-23)]) # m^2
            E = np.array([3.23*10**(-16), 9.68*10**(-16), 3.23*10**(-15), 6.45*10**(-15), 9.68*10**(-15), 3.23*10**(-14), 9.68*10**(-14), 2.26*10**(-13)]) # J
            sigma_cx_interp = interp1d(E, sigma_cx_perE, kind='linear',fill_value='extrapolate',bounds_error=False)
            self.sigma_cx = sigma_cx_interp(0.5 * (self.M_i*self.M_eff) * self.V_cx[-1]**2) # m^2, charge-exchange cross-section evaluated at psi_N ~ 1
        
            # ionization cross-section
            self.sigma_i = scd_adas(self.n_e[-1], self.T_e[-1]) # m^2, ionization cross-section evaluated at psi_N ~ 1
        
        else:
            assert False, 'species not supported'

    def calc_volavgP(self):
        """Calculate the volume-averaged pressure

        Parameters
        ----------
        self : object
            instance of saarelma_connor class

        Sets
        ----
        self.volavgP : float
            Volume-averaged pressure (same units as self.pres).
        """
        dV_dpsi = np.gradient(self.V_plasma, self.psi_N_pres)
        self.volavgP = (simpson(self.pres * dV_dpsi, self.psi_N_pres)
                        / simpson(dV_dpsi, self.psi_N_pres))

    def calc_betan(self):
        """Calculate the normalized beta

        Parameters
        ----------
        self : object
            instance of saarelma_connor class

        Sets
        -------
        self.Betan : float
            Normalized beta, dimensionless
        """

        self.calc_volavgP()

        _, [B_R, B_Z, _] = self.calc_B(self.eq['rzout'][:, 0], self.eq['rzout'][:, 1])
        bp_lcfs = np.sqrt(B_R**2 + B_Z**2)
        bp_avg = np.mean(bp_lcfs)

        betat = self.volavgP / (self.bt**2 / (2 * self.mu0))
        betap = self.volavgP / (bp_avg**2 / (2 * self.mu0))

        beta = ((1/betat) - (1/betap))**(-1)
        self.betan = beta * (self.a*self.bt/self.Ip)

    def form_factor(self,type = 'ex'):
        """Calculate the form factor for FC or charge-exchange cases
        Currently just sets to 1, but can be updated to use a more sophisticated to account for poloidal asymmetries in the FC and CX neutral profiles.

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        type : string
            type of form factor to calculate, supporting: FC, cx
        """
        assert type != 'FC' or type != 'cx', 'form factor must be for FC or cx'

        # grad(r) and nFC or nCX are needed

        if type == 'FC':
            self.fFC = 1
        elif type == 'cx':
            self.fCX = 1


    def setup_solver_grids(self,res = 100):
        """Setup the grids for the solver and calculates the flux surface-averaged |grad(r)| and |grad(r)|^2
        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        res : int, optional
            number of points to use in the radial grid if using the first method of defining the radial grid

        Returns
        -------
        self.x_prev : ndarray, shape (n_psi,)
            Radial grid (shifted so zero is at the separatrix) only at midplane, defined on the psi_N_pres grid.
        self.gradr_fsa : ndarray, shape (n_psi,)
            Flux-surface-averaged |grad(r)| at each psi_N_pres surface.
        self.gradr2_fsa : ndarray, shape (n_psi,)
            Flux-surface-averaged |grad(r)|^2 at each psi_N_pres surface.
        self.S_i_pres : ndarray, shape (n_psi,)
            Ionization cross-section at each psi_N_pres surface.
        self.S_cx_pres : ndarray, shape (n_psi,)
            Charge-exchange cross-section at each psi_N_pres surface.
        self.V_cx_pres : ndarray, shape (n_psi,)
            Volume of the charge-exchange neutral at each psi_N_pres surface.
        """

        # calculate the flux surface-averaged |grad(r)| and |grad(r)|^2
        self.calc_gradr()

        # one method of defining the radial grid, requires uncommenting the rsep_mid definition in the mhd_load function
        # self.rmid = np.linspace(0, self.rsep_mid, res) # m, radial grid (shifted so zero is at the magnetic axis)
        # self.xmid = self.rmid - self.rsep_mid # m, radial grid (shifted so zero is at the separatrix)

        # another method of defining the radial grid
        # self.r_psi is the outboard midplane minor radius for each flux surface for the psi_N_pres grid
        self.x_init = self.r_psi - self.r_psi[-1] # m, radial grid (shifted so zero is at the separatrix) only at midplane, defined on the psi_N_pres grid
        self.x_prev = self.x_init.copy() # m, radial grid (shifted so zero is at the separatrix) only at midplane, defined on the psi_N_pres grid

        # interpolate quantities on Te grid to psi_N_pres grid
        self.S_i_pres = interp1d(self.psi_Te_eval, self.S_i, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.S_cx_pres = interp1d(self.psi_Te_eval, self.S_cx, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)
        self.V_cx_pres = interp1d(self.psi_Te_eval, np.abs(self.V_cx), kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_pres)

        # note all 1D quantities are now defined on the psi_N_pres grid, which is the same as the x_prev grid

        self.x_inner = interp1d(self.psi_N_pres, self.x_prev, kind='linear', bounds_error=False, fill_value='extrapolate')(self.psi_N_inner_boundary)

    def calc_gradr(self):
        """Compute <|grad(r)|> at each flux surface.

        r is defined as the outboard-midplane minor radius for each flux
        surface, making it a proper flux-surface label (one value per surface).
        By the chain rule:
            |grad(r)| = |dr/dpsi| * |grad(psi)|
        This varies poloidally because |grad(psi)| is larger where flux
        surfaces are compressed (inboard side) and smaller where they are
        spread apart (outboard side).

        The flux surface average is:
            <|grad(r)|> = ∮ R² |grad(r)| dθ / ∮ R² dθ

        Sets
        ----
        self.r_psi : ndarray, shape (n_psi,)
            Outboard midplane minor radius for each flux surface (m).
        self.gradr_c : ndarray, shape (n_psi,)
            |grad(r)| at each contour point on each flux surface.
        self.gradr_fsa : ndarray, shape (n_psi,)
            Flux-surface-averaged |grad(r)| at each psi_N_pres surface.
        self.gradr2_fsa : ndarray, shape (n_psi,)
            Flux-surface-averaged |grad(r)|^2 at each psi_N_pres surface.
        """
        R_axis = self.eq['raxis']
        Z_axis = self.eq['zaxis']

        psi_spl = RectBivariateSpline(self.zgrid, self.rgrid, self.psi_RZ)
        n_psi = len(self.psi_N_pres)

        # r(psi): find outboard midplane crossing for each flux surface
        R_out = np.linspace(R_axis, self.rgrid[-1], 500)
        Z_mid = np.full_like(R_out, Z_axis)
        psi_mid = psi_spl(Z_mid, R_out, grid=False)
        sort_idx = np.argsort(psi_mid)
        psi_to_R = interp1d(psi_mid[sort_idx], R_out[sort_idx], kind='linear',
                            bounds_error=False, fill_value=np.nan)

        self.r_psi = np.zeros(n_psi)
        for i, psi_val in enumerate(self.psi_pres):
            self.r_psi[i] = psi_to_R(psi_val) - R_axis

        dr_dpsi = np.gradient(self.r_psi, self.psi_N_pres) # (m) / (dimensionless), change in r_midplane(psi_N) over psi_N

        self.gradr_fsa = np.zeros(n_psi)
        self.gradr2_fsa = np.zeros(n_psi)
        fig, ax = plt.subplots()
        for i, psi_val in enumerate(self.psi_pres):
            ax.cla()
            cs = ax.contour(self.rgrid, self.zgrid, self.psi_RZ,
                            levels=[psi_val])
            segs = cs.allsegs[0]
            if not segs:
                self.gradr_fsa[i] = np.nan
                self.gradr2_fsa[i] = np.nan
                continue

            seg = max(segs, key=lambda s: len(s)) # longest contour = the real flux surface, not islands
            R_c, Z_c = seg[:, 0], seg[:, 1]

            theta_c = np.arctan2(Z_c - Z_axis, R_c - R_axis) # theta at all points on contour
            idx = np.argsort(theta_c)
            theta_c, R_c, Z_c = theta_c[idx], R_c[idx], Z_c[idx]

            theta_c = np.append(theta_c, theta_c[0] + 2 * np.pi)
            R_c = np.append(R_c, R_c[0])
            Z_c = np.append(Z_c, Z_c[0])

            # |grad(psi)| at each contour point from the equilibrium spline
            dpsi_dR = psi_spl(Z_c, R_c, dx=0, dy=1, grid=False) # value at each point on the contour
            dpsi_dZ = psi_spl(Z_c, R_c, dx=1, dy=0, grid=False) # value at each point on the contour
            grad_psi_mag = np.sqrt(dpsi_dR**2 + dpsi_dZ**2) # value at each point on the contour

            # |grad(r)| = |dr/dpsi| * |grad(psi)| at each contour point on each flux surface i
            gradr_c = np.abs(dr_dpsi[i]) * grad_psi_mag

            den = simpson(R_c**2, theta_c)
            self.gradr_fsa[i] = simpson(R_c**2 * gradr_c, theta_c) / den
            self.gradr2_fsa[i] = simpson(R_c**2 * gradr_c**2, theta_c) / den
        plt.close(fig)


    def first_step(self,resolution=200):
        """Solve Equation (16) in S. Saarelma et al 2023 Nucl. Fusion 63 052002
        This is the simplified BVP (no charge-exchange neutrals) used as the
        initial guess for the full iterative solve of Equation (15).

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        resolution : int, optional
            number of points to use in the radial grid solve_bvp() call

        Boundary conditions:
            dn_e/dx = dne_dx_neginf  at x_inner (psi_N = 0.85)
            n_e     = ne_x0          at x = 0   (separatrix)

        Uses coefficient interpolators set up by solve() and stored as
        self._D_ped_x, self._gradr2_x, self._Si_x, self._Scx_x,
        self._P_x, self._dPdx_x, self._x_inner.

        Sets
        ----
        self.ne_first_sol : BVP solution object from solve_bvp.
        """
        x_inner = self.x_inner

        D_ped_x = interp1d(self.x_prev, self.D_ped, kind='linear', bounds_error=False, fill_value='extrapolate')
        Si_x = interp1d(self.x_prev, self.S_i_pres, kind='linear', bounds_error=False, fill_value='extrapolate')
        Scx_x = interp1d(self.x_prev, self.S_cx_pres, kind='linear', bounds_error=False, fill_value='extrapolate')

        # f(x) = <|grad(r)|^2> * D_ped
        f_arr = self.gradr2_fsa * self.D_ped
        df_arr = np.gradient(f_arr, self.x_prev)
        f_x = interp1d(self.x_prev, f_arr, kind='linear', bounds_error=False, fill_value='extrapolate')
        df_dx = interp1d(self.x_prev, df_arr, kind='linear', bounds_error=False, fill_value='extrapolate')

        def ode(x, Y):
            ne, dnedx = Y
            D = D_ped_x(x)
            Si = Si_x(x)
            Scx = Scx_x(x)
            f = f_x(x)
            dfdx = df_dx(x)
            rhs = (ne * D * (Si + Scx) / self.V_FC) * (dnedx - self.dne_dx_neginf)
            d2nedx2 = (rhs - dfdx * dnedx) / f
            return np.vstack([dnedx, d2nedx2])

        def bc(Ya, Yb):
            return np.array([
                Ya[1] - self.dne_dx_neginf, # Neumann boundary condition at x_inner
                Yb[0] - self.ne_x0, # Dirichlet boundary condition at x = 0
            ])

        x_grid = np.linspace(x_inner, 0, resolution)
        ne_guess = self.ne_x0 + self.dne_dx_neginf * x_grid
        ne_guess = np.maximum(ne_guess, self.ne_x0 * 0.1)
        dne_guess = np.full_like(x_grid, self.dne_dx_neginf)
        Y_guess = np.vstack([ne_guess, dne_guess])

        sol = solve_bvp(ode, bc, x_grid, Y_guess)
        if not sol.success:
            raise RuntimeError(f"first_step BVP failed: {sol.message}")

        self.sol = sol

    def solve(self,soln_method='sc_2order',tol=1e-4,max_iter=50,x_res=100):
        """Iteratively solve Equation (15) in S. Saarelma et al 2023 Nucl. Fusion 63 052002

        1. Sets up coefficient interpolators mapping psi_N quantities to x.
        2. Calls first_step() to solve the simplified Eq 16 (no CX neutrals).
        3. Iterates the full Eq 15 (with CX neutrals and the integral term)
           until the n_e profile converges.

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        soln_method : string
            method to use to determine the solution, supporting: sc_2order
        tol : float
            Relative convergence tolerance on n_e for solve_bvp() call.
        max_iter : int
            Maximum number of Picard iterations for solve_bvp() call.
        x_res : int
            Number of points to use in the radial grid for solve_bvp() call.

        Sets
        ----
        self.ne_sol : ndarray
            Converged electron density profile on self.x_sol.
        self.x_sol : ndarray
            Radial grid (x = 0 at separatrix, x < 0 inside).
        """

        # form factors are currently set to 1, assuming poloidal symmetry as done in S. Saarelma et al 2024 Nucl. Fusion 64 076025 Section 3
        self.form_factor(type='FC')
        self.form_factor(type='cx')
        self.setup_solver_grids(res=x_res)

        if soln_method == 'sc_2order':
            # --- Step 1: first step (Eq 16, no CX neutrals) ---
            self.first_step()

            # --- Step 2: iterate full Eq 15 ---
            for i in range(max_iter):

                # Interpolate required parameters to x_sol_prev grid (interpolating from initial grid to minimize errors)
                self.x_prev = self.sol.x # not gauranteed to be the same grid as inputted to solve_bvp()
                ne_sol_prev = self.sol.y[0] # not gauranteed to be the same grid as inputted to solve_bvp()

                D_ped_int = interp1d(self.x_init, self.D_ped, kind='linear', bounds_error=False, fill_value='extrapolate')
                S_i_int = interp1d(self.x_init, self.S_i_pres, kind='linear', bounds_error=False, fill_value='extrapolate')
                S_cx_int = interp1d(self.x_init, self.S_cx_pres, kind='linear', bounds_error=False, fill_value='extrapolate')
                gradr_fsa_int = interp1d(self.x_init, self.gradr_fsa, kind='linear', bounds_error=False, fill_value='extrapolate')
                V_CX_int = interp1d(self.x_init, self.V_cx_pres, kind='linear', bounds_error=False, fill_value='extrapolate')
                    
                integrand = (ne_sol_prev * S_i_int(self.x_prev) + S_cx_int(self.x_prev)) / (self.fFC * self.V_FC) # iterative integral: int_0^x n_e(x')*(S_i+S_cx)/(f_FC*V_FC) dx'
                int_from_left = cumulative_trapezoid(integrand, self.x_prev, initial=0) # want integral output to be calculated from 0 to x
                integral_from_0 = int_from_left - int_from_left[-1] # want integral output to be calculated from 0 to x and cumulative_trapezoid gives integral from x_sol to 0 since x_sol<0
                exp_term_prev = np.exp(integral_from_0)

                # f(x) = <|grad(r)|^2> * D_ped
                f_x = gradr_fsa_int(self.x_prev) * D_ped_int(self.x_prev)
                df_dx = np.gradient(f_x, self.x_prev) # df/dx

                # change to callables to input to solve_bvp()
                f_x = interp1d(self.x_prev, f_x, kind='linear', bounds_error=False, fill_value='extrapolate')
                df_dx = interp1d(self.x_prev, df_dx, kind='linear', bounds_error=False, fill_value='extrapolate')
                exp_term_prev = interp1d(self.x_prev, exp_term_prev, kind='linear', bounds_error=False, fill_value='extrapolate')

                def ode(x, Y):
                    ne, dnedx = Y
                    D = D_ped_int(x)
                    Si = S_i_int(x)
                    Scx = S_cx_int(x)
                    Vcx = V_CX_int(x)
                    f = f_x(x)
                    dfdx = df_dx(x)
                    exp_term = exp_term_prev(x)

                    rhs = -ne*Si * ( -D*(dnedx-self.dne_dx_neginf) + ( 1 - (abs(self.V_FC)*self.fFC/(Vcx*self.fCX)) * ((Si+Scx/2)/(Si+Scx)) )*(self.nFC_x0*exp_term) )
                    d2nedx2 = (rhs - (dfdx * dnedx)) / f
                    return np.vstack([dnedx, d2nedx2])

                def bc(Ya, Yb):
                    return np.array([
                        Ya[1] - self.dne_dx_neginf, # Neumann boundary condition at x_inner
                        Yb[0] - self.ne_x0, # Dirichlet boundary condition at x = 0
                    ])

                x_grid = np.linspace(self.x_inner, 0, x_res)
                ne_guess = ne_sol_prev
                dne_guess = np.gradient(ne_guess, x_grid)
                Y_guess = np.vstack([ne_guess, dne_guess])

                self.sol = solve_bvp(ode, bc, x_grid, Y_guess)
                if not self.sol.success:
                    raise RuntimeError(f"step {i} BVP failed: {self.sol.message}")

                ne_sol_prev = interp1d(self.x_prev, ne_sol_prev, kind='linear', bounds_error=False, fill_value='extrapolate')(self.sol.x)
                residual = np.max(np.abs(self.sol.y[0] - ne_sol_prev)) / np.max(np.abs(self.sol.y[0]))
                print(f"  Eq 15 iteration {i}: residual = {residual:.2e}")

                if residual < tol:
                    break

            self.x_sol = self.sol.x
            self.ne_sol = self.sol.y
            self.dne_dx_sol = self.sol.yp


    def fsa(self,A,flux_surfaces='T_e'):
        """Flux surface average a quantity as defined by ⟨A⟩= int(R^2Adθ)/ int(R^2dθ) in S. Saarelma et al 2023 Nucl. Fusion 63 052002

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        A : array
            2D array of A values at each R_grid, Z_grid.
        flux_surfaces : string
            which flux surface to average over, supporting: T_e, psi_N_pres
             
        """

        R_axis = self.eq['raxis']
        Z_axis = self.eq['zaxis']

        if flux_surfaces == 'T_e':
            psi_N_vals = self.psi_Te_eval
        elif flux_surfaces == 'psi_N_pres':
            psi_N_vals = self.psi_N_pres
        else:
            assert False, 'valid flux_surfaces method must be provided'

        A_spl = RectBivariateSpline(self.zgrid, self.rgrid, A)
        fsa_A = np.full(len(psi_N_vals), np.nan)

        fig, ax = plt.subplots()
        for i, psi_val in enumerate(psi_N_vals):
            if psi_val <= 0.01 or psi_val >= 0.99:
                continue

            ax.cla()
            cs = ax.contour(self.rgrid, self.zgrid, self.psi_RZ_N,
                            levels=[psi_val])

            segs = cs.allsegs[0]
            if not segs:
                continue

            # longest contour = the real flux surface, not islands
            seg = max(segs, key=lambda s: len(s))
            R_c, Z_c = seg[:, 0], seg[:, 1] # R, Z coordinates of the contour

            # poloidal angle measured from the magnetic axis
            # R_c_ax = (((R_c - R_axis)**2) + ((Z_c - Z_axis)**2))**0.5
            # theta_c = np.arcsin( (Z_c - Z_axis) / R_c_ax )
            theta_c = np.arctan2(Z_c - Z_axis, R_c - R_axis) # theta at all points on contour
            idx = np.argsort(theta_c)
            theta_c, R_c, Z_c = theta_c[idx], R_c[idx], Z_c[idx]

            # close the contour so the integral spans a full 2*pi
            theta_c = np.append(theta_c, theta_c[0] + 2 * np.pi)
            R_c = np.append(R_c, R_c[0])
            Z_c = np.append(Z_c, Z_c[0])

            A_c = A_spl(Z_c, R_c, grid=False)

            den = simpson(R_c**2, theta_c)
            if abs(den) < 1e-30:
                continue
            fsa_A[i] = simpson(R_c**2 * A_c, theta_c) / den

        plt.close(fig)

        return fsa_A

    def psi_rz_expand(self,A,psi_N_A='T_e'):
        """For A defined for each psi_N, expand to all R_grid, Z_grid.

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        A : array
            1D array of A values at each psi_N.
        psi_N_A : array
            1D array of psi_N values at which A is defined.

        Returns
        -------
        A_expanded : array
            2D array of A values at each R_grid, Z_grid.
             
        """

        # psi_N values at which A is defined
        if psi_N_A == 'T_e':
            psi_N_A = self.psi_Te_eval # 1D array of psi_N values at which T_e is evaluated
        else:
            assert False, 'valid psi_N_A method must be provided'

        # Interpolate: psi_N -> A, then evaluate on the 2D psi_N map
        A_interp = interp1d(psi_N_A, A, kind='linear',
                            bounds_error=False, fill_value=np.nan)
        return A_interp(self.psi_RZ_N)
    
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
        self.neped = interp1d(self.x_prev, self.ne_sol, kind='linear', bounds_error=False, fill_value='extrapolate')(self.x_inner) / (1e19) # m^-3 -> 10^19 m^-3
        self.bt = self.calc_B(self.eq['raxis'],self.eq['zaxis'])[1][2]
        self.calc_betan()
        inputs = {
            "a": self.a,           # Minor radius (m)
            "betan": self.betan,       # Normalized beta
            "bt": self.bt,        # Toroidal magnetic field at the magnetic axis (T)
            "delta": self.delta,       # Effective triangularity
            "ip": self.Ip,          # Plasma current (MA)
            "kappa": self.kappa,       # Elongation
            "m": self.M_eff,           # Effective mass (must be 2.0 for D or 2.5 for D-T)
            "neped": self.neped,       # Pedestal density (in 10^19 m^-3)
            "r": self.Rmajor,           # Major radius (m)
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
