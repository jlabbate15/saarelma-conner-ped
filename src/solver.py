import numpy as np

class saarelma_connor:
    """

    Description
    ----------
    Creates instance of a tokamak pedestal configuration that the Saarelma-Connor (S. Saarelma et al 2024 Nucl. Fusion 64 076025) model can be applied to

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
        r = None,
        rmax = None,
        r_res = 30,
        E_FC = 3 * 1.60218e-19, # J,
        M_i = 3.344e-27, # kg, mass of deuterium nuclei
        sigma_i = 1, # m^2, ionization cross-section
        sigma_cx = 1, # m^2, charge-exchange cross-section
    ):
        if r == None:
            r = np.linspace(0,rmax,r_res)

        self.V_FC = np.sqrt(8*E_FC/((np.pi^2) * M_i))


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

    def Dped():
        """Finds diffusion coefficient as outlined by S. Saarelma et al 2024 Nucl. Fusion 64 076025

        Parameters
        ----------
        parm : bool, optional
            info.
             
        """

    def fsa(A):
        """Flux surface average a quantity as defined by ⟨A⟩= int(R^2Adθ)/ int(R^22dθ) in S. Saarelma et al 2023 Nucl. Fusion 63 052002

        Parameters
        ----------
        parm : bool, optional
            info.
             
        """