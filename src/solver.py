import numpy as np
import scipy

class saarelma_connor:
    """

    Description
    ----------
    Creates instance of a tokamak pedestal configuration that the Saarelma-Connor (S. Saarelma et al 2024 Nucl. Fusion 64 076025) model can be applied to
    Dependencies:
    - bouquet (install with pip install bouquet)

    Parameters
    ----------
    info : String, optional
        Any machine or shot data available for pedestal instance
    r : np array
        Span of r for relavant machine
    rmax : float
        Max of r for relevant machine
    r_res : int
    E_FC : float
        Energy of Franck-Condon neutrals as defined in Mahdavi M.A., Maingi R., Groebner R.J., Leonard A.W., Osborne T.H. and Porter G. 2003 Phys. Plasmas 10 3984
    M_i : float
        Ion mass (I think it is hydrogrenic ions)

    """
    def __init__(
        self,
        info = None,
        r = None, # ???, radius or flux at separatrix? also specify in 3D or per flux surface?
        rmax = None,
        r_res = 30,
        E_FC = 3 * 1.60218e-19, # J,
        Z_i = 1, # Z of ions
        M_i = 3.344e-27, # kg, mass of deuterium nuclei
        M_e = 9.109e-31, # kg, mass of electron
        T_i = np.linspace(1000,5000,10), # K, ion temperature over r (possibly could modify to take in 3D T and then flux-surface average)
        T_e = np.linspace(1000,5000,10), # K, electron temperature over r (possibly could modify to take in 3D T and then flux-surface average)
        T_rat = 1, # dim, T_i / T_e
        sigma_i = None, # m^2, ionization cross-section
        sigma_cx = None, # m^2, charge-exchange cross-section
        dne_dx_ninf = None, # (particles/m^3) / m, electron density gradient in the core of the plasma
        a = None, # m, minor radius
        P_tot_e = None, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002)
        S_plasma = None, # m^2, total surface area of plasma
        V_plasma = None, # m^3, volume enclosed by the plasma per poloidal flux
        R0 = None, # m, location of magnetic axis relative to device rotational line of toroidal symmetry
        P = None, # Pa, pressure per poloidal flux
        r_sep = None, # ???, radius or flux at separatrix?
        alpha_crit = None,
        C_KBM = None,
        De_chie_etg = None,
        T_rat_flag = False,
        mhd_eq_loc = 'g', # location of MHD equilibrium parameters, currently supporting: g-file
        kprof_loc = 'p', # location of kinetic parameters, currently supporting: p-file
    ):
        if r == None:
            r = np.linspace(0,rmax,r_res)
        x = r - r_sep # ???, radial coordinate with origin at separatrix, x is a flux coordinate I think

        self.geo_load(mhd_eq_loc) # load in geometric quantities

        self.e_i = Z_i * 1.602e-19 # C
        self.V_th_i = np.sqrt(2*T_i/M_i) # m/s
        self.V_th_e = np.sqrt(2*T_e/M_e) # m/s

        self.S_i = sigma_i * self.V_th_e # m^3/s
        self.S_cx = sigma_cx * self.V_th_i # m^3/s

        self.V_FC = np.sqrt(8*E_FC/((np.pi^2) * M_i)) # m/s
        self.V_cx = np.sqrt(2*T_i/(np.pi * M_i)) # m/s

        self.dne_dx_ninf = dne_dx_ninf

        if T_rat_flag:
            T_i = T_e * T_rat

        # Diffusion coefficient setup
        c_s =  # m/s, https://www.osti.gov/servlets/purl/4315023
        rho_s = self.V_th_i*M_i / (self.e_i * B) # m, how do you find B?
        mu0 = 4 * np.pi * 10**-7 # N/A**2, vacuum magnetic permeability constant
        alpha = (2 * np.gradient(V_plasma) / ((2*np.pi)**2)) * mu0 * np.gradient(P) * np.sqrt(V_plasma / (2*R0*np.pi**2))

        # Diffusion coefficient computation
        D_KBM = np.where(
            alpha > alpha_crit,
            C_KBM*(alpha-alpha_crit)*(c_s*rho_s**2)/a,
            0)
        D_ETG = De_chie_etg * P_tot_e / (S_plasma * np.gradient(T)) # Is T just the electron temperature here? Do we give a 1D profile or 3D? Probably 3D if we want a true gradient, but we are looking for the magnitude of the gradient I think
        D_NEO = 0.05 * (c_s * rho_s**2) / a
        self.D_ped = D_KBM + D_ETG + D_NEO

    def mhd_load(self,mhd_loc):
        """Load MHD equilibrium parameters using method specified by mhd_eq_loc flag. 
        Parameters that will be loaded include: R0, a, r_sep(theta), r, P(psi), psi_pol(r)
        Calculates: Plasma volume, plasma surface area
        
        Parameters
        ----------
        mhd_loc : string
            which method to use to load geometric parameters.
             
        """

        from bouquet import GEQDSKEquilibrium # need bouquet as dependency, "pip install bouquet"

        # Load a g-file
        eq = GEQDSKEquilibrium("g123456.01000")
        print(f"Ip = {eq.Ip/1e6:.3f} MA")
        print(f"q95 = {eq.q_profile[-1]:.2f}")
        print(f"li(1) = {eq.li1:.3f}")

        # Access flux-surface geometry
        geo = eq.geometry
        print(f"Elongation at boundary: {geo['kappa'][-1]:.2f}")

        # Exact outboard-midplane profiles
        mid = eq.midplane
        print(f"R_mid at boundary: {mid['R'][-1]:.4f} m")

    def kprof_load(self,kprof_loc,T_rat=None):
        """Load kinetic equilibrium parameters using method specified by mhd_eq_loc flag. 
        Parameters that will be loaded include: T_e, n_e
        Calculates: dn_e/dx|x=-inf, T_i
        
        Parameters
        ----------
        kprof_loc : string
            which method to use to load geometric parameters.
             
        """

        from bouquet import read_pfile # need bouquet as dependency, "pip install bouquet"

        # Load a p-file
        pf = read_pfile("p123456.01000")
        ne = pf.get("ne")       # returns (psinorm, values, derivatives) tuple
        Te = pf.get("te")

    
    def first_step():
        """Solve Equation (16) in S. Saarelma et al 2023 Nucl. Fusion 63 052002

        Parameters
        ----------
        parm : bool, optional
            info.
             
        """   

    def solve():
        """Iteratively solve Equation (15) in S. Saarelma et al 2023 Nucl. Fusion 63 052002

        Parameters
        ----------
        parm : bool, optional
            info.
             
        """

        f_FC = form_factor('FC')
        f_cx = form_factor('cx')

        step_one = first_step()

    def form_factor(type = 'ex'):
        assert type != 'FC' or type != 'cx', 'form factor must be for FC or cx'


    def fsa(A,theta,R):
        """Flux surface average a quantity as defined by ⟨A⟩= int(R^2Adθ)/ int(R^22dθ) in S. Saarelma et al 2023 Nucl. Fusion 63 052002

        Parameters
        ----------
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