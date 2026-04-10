from solver import saarelma_connor
test_model = saarelma_connor(
        P_tot_e = 5e6, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
        alpha_crit = 1,
        C_KBM = 0.1,
        De_chie_etg = 0.5,
        mhd_fp = 'g192185.02440', # filepath to MHD paramter file
        kprof_fp = '/Users/nelsonlab/codes/Equilibria/p193754.1850.0', # filepath to kinetic paramter file
)
test_model.solve()