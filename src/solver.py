import numpy as np
import scipy

class saarelma_connor:
    """

    Description
    ----------
    Creates instance of a tokamak pedestal configuration that the Saarelma-Connor (S. Saarelma et al 2024 Nucl. Fusion 64 076025) model can be applied to
    Dependencies:
    - bouquet (install with pip install bouquet) for equilibrium inputs
    - juliacall (install with pip install juliacall) for EPEDNN interfacing
    - EPEDNN (install by git clone)

    Parameters
    ----------
    E_FC : float
        Energy of Franck-Condon neutrals as defined in Mahdavi M.A., Maingi R., Groebner R.J., Leonard A.W., Osborne T.H. and Porter G. 2003 Phys. Plasmas 10 3984
    M_i : float
        Ion mass (I think it is hydrogrenic ions)

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
        mhd_loc = 'g', # location of MHD equilibrium parameters, currently supporting: g-file
        kprof_loc = 'p', # location of kinetic parameters, currently supporting: p-file
        mhd_fp = None, # filepath to MHD paramter file
        kprof_fp = None, # filepath to kinetic paramter file
        T_rat_flag = False,
        species = 'D', # species of ions, currently supporting: D, D-T
    ):

        self.T_rat_flag = T_rat_flag

        self.mhd_load(mhd_loc,mhd_fp) # load in MHD quantities
        self.kprof_load(kprof_loc,kprof_fp) # load in kinetic quantities

        if species == 'D':
            self.M_eff = 2.0
        elif species == 'D-T':
            self.M_eff = 2.5
        else:
            assert True, 'species must be D or D-T'

        # if r == None:
        #     r = np.linspace(0,rmax,r_res)
        #     x = r - self.r_sep # ???, radial coordinate with origin at separatrix, x is a flux coordinate I think

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

        if mhd_loc == 'g':
            from bouquet import GEQDSKEquilibrium # need bouquet as dependency, "pip install bouquet"

            # Load a g-file
            eq = GEQDSKEquilibrium(fp)
            self.R_grid = eq.R_grid # m, 1D R grid
            self.psi_RZ = eq.psi_RZ # 2D poloidal flux array at each RZ grid point
            self.pres = eq.pres # pressure evaluated at each psi_N
            self.psi_N = eq.psi_N # Normalized psi
            self.psi_boundary = eq.psi_boundary # poloidal flux at boundary

            # Saarelma-Connor model input parameters
            self.S_plasma = eq.geometry["surfArea"] # m^2, total surface area of plasma
            self.V_plasma = eq.geometry["vol"] # m^3, volume enclosed by the plasma per poloidal flux
            self.a = eq.geometry["a"] # m, minor radius
            self.P = eq.pres # pressure evaluated at each psi_N
            self.R0 = eq.R_mag # location of magnetic axis relative to device rotational line of toroidal symmetry
            self.r_mhd = eq.R_grid # physical radius grid from the file that imports MHD parameters
            self.r_sep =  # physical radius of separatrix as a function of theta

            # EPEDNN input parameters
            self.betan =        # Normalized beta
            self.Bt =           # Toroidal magnetic field (T)
            self.delta =       # Effective triangularity
            self.Ip =           # Plasma current (MA)
            self.kappa =        # Elongation



    def kprof_load(self,kprof_loc='p',T_rat=None):
        """Load kinetic equilibrium parameters using method specified by mhd_eq_loc flag. 
        Parameters that will be loaded include: T_e, n_e
        Calculates: dn_e/dx|x=-inf, T_i
        
        Parameters
        ----------
        kprof_loc : string
            which method to use to load kinetic parameters.
             
        """

        if kprof_loc == 'p':
            from bouquet import read_pfile # need bouquet as dependency, "pip install bouquet"

            # Load a p-file
            pf = read_pfile("p123456.01000")

            # Extract profiles
            self.n_e = pf.get("ne")       # returns (psinorm, values, derivatives) tuple
            self.T_e = pf.get("te")

            # Calculate boundary condition from profiles
            self.dne_dx_ninf = # (particles/m^3) / m, electron density gradient in the core of the plasma

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