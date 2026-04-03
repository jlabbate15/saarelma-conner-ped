import numpy as np
from scipy.interpolate import RectBivariateSpline, interp1d
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
        self.verbose = verbose

        self.mhd_load(mhd_loc,mhd_fp) # load in MHD quantities
        self.kprof_load(kprof_loc,kprof_fp) # load in kinetic quantities
        self.cross_sections(species) # load in cross-sections
        self.calc_B(self.rgrid,self.zgrid) # calculate the magnetic field at each RZ grid point, sets self.B

        if species == 'D':
            self.M_eff = 2.0
        elif species == 'D-T':
            self.M_eff = 2.5
        else:
            assert True, 'species must be D or D-T'

        self.Z_i = Z_i
        self.e_i = Z_i * 1.602e-19 # C
        self.V_th_i = np.sqrt(2*self.T_i/(M_i*self.M_eff)) # m/s, per psi_N_eval for Ti
        self.V_th_e = np.sqrt(2*self.T_e/M_e) # m/s, per psi_N_eval for Te

        self.S_i = self.sigma_i * self.V_th_e # m^3/s, per psi_N_eval for Te
        self.S_cx = self.sigma_cx * self.V_th_i # m^3/s, per psi_N_eval for Te

        self.V_FC = np.sqrt(8*E_FC/((np.pi^2) * M_i*self.M_eff)) # m/s
        self.V_cx = np.sqrt(2*self.T_i/(np.pi * M_i*self.M_eff)) # m/s, per psi_N_eval for Ti

        # Diffusion coefficient setup
        c_s = (self.e_i * self.T_e / (M_i * self.M_eff)) ** 0.5 # m/s, cs = (e*T_e/mD)^1/2 as defined in W. Guttenfelder et al 2021 Nucl. Fusion 61 056005
        V_th_i_rz = self.psi_rz_expand(self.V_th_i, psi_N_A='T_e')
        rho_s = V_th_i_rz*M_i*self.M_eff / (self.e_i * self.B) # m, known on each RZ grid point
        rho_s = self.fsa(rho_s,flux_surfaces='T_e') # m, known on each flux surface
        mu0 = 4 * np.pi * 10**-7 # N/A**2, vacuum magnetic permeability constant
        alpha = (2 * np.gradient(self.V_plasma) / ((2*np.pi)**2)) * mu0 * np.gradient(self.pres) * np.sqrt(self.V_plasma / (2*self.R0*np.pi**2)) # evaluated at each psi_N = np.linspace(eq['psimag'], eq['psibry'], len(self.pres))

        # Diffusion coefficient computation
        D_KBM = np.where(
            alpha > alpha_crit,
            C_KBM*(alpha-alpha_crit)*(c_s*rho_s**2)/self.a,
            0)
        D_ETG = De_chie_etg * P_tot_e / (self.S_plasma * np.gradient(self.T_e,self.psi_Te_eval) ) # evaluated at each psi_Te_eval
        D_NEO = 0.05 * (c_s * rho_s**2) / self.a
        self.D_ped = D_KBM + D_ETG + D_NEO

    def find_boundary_points(eq):
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
        psi_F = np.linspace(eq['psimag'], eq['psibry'], len(F))

        e_Bp = 1 if self.pol_norm else 0

        r = self.rgrid
        z = self.zgrid
        spl = RectBivariateSpline(z, r, eq['psirz'])

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

            bdry = self.find_boundary_points(self.eq)

            rmax_top = bdry['top'][0]
            rmax_bottom = bdry['bottom'][0]
            zmax_top = bdry['top'][1]
            zmax_bottom = bdry['bottom'][1]
            rmax_outboard = bdry['outboard'][0]
            rmax_inboard = bdry['inboard'][0]
            # zmax_outboard = bdry['outboard'][1]
            # zmax_inboard = bdry['inboard'][1]

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
            self.pres = self.eq['pres'] # pressure evaluated at each psi_N = np.linspace(eq['psimag'], eq['psibry'], len(self.pres))

            # Grids
            self.rgrid = np.linspace(self.eq['rleft'],self.eq['rleft']+self.eq['rdim'],self.eq['nr']) # m, 1D R grid
            self.zgrid = np.linspace(self.eq['zmid']-self.eq['zdim']/2,self.eq['zmid']+self.eq['zdim']/2,self.eq['nz']) # m, 1D Z grid
            self.psi_RZ = self.eq['psirz'] # 2D poloidal flux array at each RZ grid point
            self.psi_RZ_N = (self.psi_RZ - self.eq['psimag']) / (self.eq['psibry'] - self.eq['psimag']) # normalized poloidal flux at each RZ grid point

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
            self.sigma_cx = sigma_cx_interp(0.5 * self.M_i * self.V_cx[-1]**2) # m^2, charge-exchange cross-section evaluated at psi_N ~ 1
        
            # ionization cross-section
            self.sigma_i = scd_adas(self.n_e[-1], self.T_e[-1]) # m^2, ionization cross-section evaluated at psi_N ~ 1
        
        else:
            assert True, 'species not supported'


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

        # form factors are currently set to 1, assuming poloidal symmetry as done in S. Saarelma et al 2024 Nucl. Fusion 64 076025 Section 3
        self.form_factor('FC')
        self.form_factor('cx')

        self.first_step()

        self.neped =  # solution of SC model

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

    def fsa(self,A,flux_surfaces='T_e'):
        """Flux surface average a quantity as defined by ⟨A⟩= int(R^2Adθ)/ int(R^2dθ) in S. Saarelma et al 2023 Nucl. Fusion 63 052002

        Parameters
        ----------
        self : object
            instance of saarelma_connor class
        A : array
            2D array of A values at each R_grid, Z_grid.
        flux_surfaces : string
            which flux surface to average over, supporting: T_e
             
        """

        R_axis = self.eq['raxis']
        Z_axis = self.eq['zaxis']

        if flux_surfaces == 'T_e':
            psi_N_vals = self.psi_Te_eval
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
            R_c_ax = (((R_c - R_axis)**2) + ((Z_c - Z_axis)**2))**0.5
            theta_c = np.arcsin( (Z_c - Z_axis) / R_c_ax )
            idx = np.argsort(theta_c)
            theta_c, R_c, Z_c = theta_c[idx], R_c[idx], Z_c[idx]

            # close the contour so the integral spans a full 2*pi
            theta_c = np.append(theta_c, theta_c[0] + 2 * np.pi)
            R_c = np.append(R_c, R_c[0])
            Z_c = np.append(Z_c, Z_c[0])

            A_c = A_spl(Z_c, R_c, grid=False)

            den = np.simpson(R_c**2, theta_c)
            if abs(den) < 1e-30:
                continue
            fsa_A[i] = np.simpson(R_c**2 * A_c, theta_c) / den

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
        if psi_N_A is 'T_e':
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
