# Loop between EPEDNN and coupled version of Saarelma-Connor model to find self-consistent pedestal

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d

ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(ROOT))

from src.solver_nondim import saarelma_connor_nondim

def profiles_loop_solve(
    MHD_FP = None,
    KPROF_FP = None,
    ne_success_fp = 'compare_nondim',
    success_fp = 'success_PTHmode.txt',
    failure_fp = 'failure_PTHmode.txt',
    error_messages_fp = 'error_messages_PTHmode.txt',
    initial_guess = "tanh",
    ne_inner_bc = "neumann",
    x_res = 40,
    N = 3, # Saarelma-Connor free param scan size
    P_tot_e = 5e6, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
    psi_N_inner = 0.85,
    alpha_crits_minmax = [-1,1],
    C_KBMs_minmax = [-1,1],
    De_chie_etgs_minmax = [-1,1],
    nFC_x0s_minmax = [14,18],
    ncx_x0_ratios_minmax = [0.1,1.25],
    eped_tol_max = 1e-3,
    eped_iter_max = 50,
    verbose = False,
):

    # Free parameters
    alpha_crits = np.logspace(alpha_crits_minmax[0], alpha_crits_minmax[1], N)
    C_KBMs = np.logspace(C_KBMs_minmax[0], C_KBMs_minmax[1], N)
    De_chie_etgs = np.logspace(De_chie_etgs_minmax[0], De_chie_etgs_minmax[1], N)
    nFC_x0s = np.logspace(nFC_x0s_minmax[0], nFC_x0s_minmax[1], N)
    ncx_x0_ratios = np.logspace(ncx_x0_ratios_minmax[0],ncx_x0_ratios_minmax[1],N)

    SOLVE_KW = dict(
        x_res=x_res,
        fe_degree=2,
        initial_guess=initial_guess,
        ne_inner_bc=ne_inner_bc,   # Saarelma A7 default; see dirichlet comparison below
        linear_solver="lu",      # or "gamg" for GMRES + algebraic multigrid on J
        nCX_ic="solve",
        kbm_treatment="inline",
        kbm_gate_eps=0.01, # 1e-3 minimum
        verbose=False,
    )

    # Output data and files
    with open(success_fp, 'w') as f:
            f.write(f"alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, psi_N_inner\n")
    with open(failure_fp, 'w') as f:
            f.write(f"alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, psi_N_inner\n")
    with open(error_messages_fp, 'w') as f:
            f.write(f"alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, psi_N_inner, message\n")

    base_model = saarelma_connor_nondim(
            P_tot_e      = P_tot_e,
            alpha_crit   = round(float(alpha_crits[0]), 3),
            C_KBM        = round(float(C_KBMs[0]), 3),
            De_chie_etg  = round(float(De_chie_etgs[0]), 3),
            nFC_x0       = round(float(nFC_x0s[0]), 3),
            ncx_x0_ratio = round(float(ncx_x0_ratios[0]), 3),
            psi_N_inner_boundary = psi_N_inner,
            mhd_fp       = MHD_FP,
            kprof_fp     = KPROF_FP,
            verbose      = verbose,
            # psi_N_inner_boundary = 0.85, # set to None to use adaptive inner boundary method
    )
    base_model.setup_epednn()
    print("Base model built.")

    for eped_iter in range(eped_iter_max):

        # Clear outputs from any previous scan (including appended failure logs) and setup logging files
        for file in os.listdir(ne_success_fp):
            os.remove(os.path.join(ne_success_fp, file))
        os.remove(success_fp)
        os.remove(failure_fp)
        os.remove(error_messages_fp)
        with open(success_fp, "w") as f:
            f.write("alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, psi_N_inner\n")
        with open(failure_fp, "w") as f:
            f.write("alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, psi_N_inner\n")
        with open(error_messages_fp, "w") as f:
            f.write("alpha_crit, C_KBM, De_chie_etg, nFC_x0, ncx_x0_ratio, psi_N_inner, message\n")

        # Scan free parameters for SC model
        i=0
        for alpha_crit in alpha_crits:
                for C_KBM in C_KBMs:
                        for De_chie_etg in De_chie_etgs:
                                for nFC_x0 in nFC_x0s:
                                        for ncx_x0_ratio in ncx_x0_ratios:
                                                ac = round(float(alpha_crit), 3)
                                                ck = round(float(C_KBM), 3)
                                                de = round(float(De_chie_etg), 3)
                                                nf = round(float(nFC_x0), 3)
                                                nc = round(float(ncx_x0_ratio),3)
                                                try:
                                                        base_model.update_free_params(
                                                                alpha_crit            = ac,
                                                                C_KBM                 = ck,
                                                                De_chie_etg           = de,
                                                                nFC_x0                = nf,
                                                                ncx_x0_ratio          = nc,
                                                        )
                                                        x_sol, ne_sol, nFC_sol, nCX_sol = base_model.solve_coupled_nondim(**SOLVE_KW)
                                                        sol = {'x': x_sol, 'y': ne_sol, 'nFC': nFC_sol, 'nCX': nCX_sol}
                                                except Exception as e: # run fails
                                                        with open(failure_fp, 'a') as f:
                                                                f.write(f"{ac}, {ck}, {de}, {nf}, {nc}, {psi_N_inner:.4f}\n")
                                                        with open(error_messages_fp, 'a') as f:
                                                                f.write(f"{ac}, {ck}, {de}, {nf}, {nc}, {psi_N_inner:.4f}, {e}\n")
                                                else: # run works
                                                        with open(success_fp, 'a') as f:
                                                                f.write(f"{ac}, {ck}, {de}, {nf}, {nc}, {psi_N_inner:.4f}\n")
                                                        np.save(f'{ne_success_fp}/ne_a{ac}_C{ck}_D{de}_n{nf}_nc{nc}_b{psi_N_inner:.4f}', sol, allow_pickle=True)
                # print(f"Completed {i} of {len(alpha_crits)} alpha_crits") # progress logging

        # --- Pick best ne profile -----------------------------------------
        n_combos = (len(alpha_crits) * len(C_KBMs) * len(De_chie_etgs)
                    * len(nFC_x0s) * len(ncx_x0_ratios))

        # psi_N <-> x map (reuse base_model; no extra solver instantiation)
        psi_N_pres = np.asarray(base_model.psi_N_pres, dtype=float)
        x_grid_full = np.asarray(base_model.r_psi, dtype=float) - float(base_model.r_psi[-1])
        psi_to_x = interp1d(psi_N_pres, x_grid_full, kind='linear',
                            bounds_error=False, fill_value='extrapolate')
        psi_ped_grid = np.linspace(psi_N_inner, 1.0, x_res)
        x_ped_grid = psi_to_x(psi_ped_grid)

        if n_combos <= 1:
            # Single-combination run: the only sol from the inner loop IS the best.
            best_x = np.asarray(sol['x'], dtype=float)
            best_ne = np.asarray(sol['y'], dtype=float)
        else:
            if KPROF_FP is None:
                raise ValueError("Must specify pfile for a parameter scan implementation "
                                 "of SC model in EPEDNN loop")

            # Read p-file ne once
            psi_pfile, ne_pfile = [], []
            in_ne = False
            with open(KPROF_FP) as f:
                for line in f:
                    if '3 N Z A' in line:
                        break
                    if line.startswith('201'):
                        in_ne = 'ne(10^20/m^3)' in line
                        continue
                    if in_ne:
                        p, v, _ = line.split()
                        psi_pfile.append(float(p)); ne_pfile.append(float(v))
            psi_pfile = np.asarray(psi_pfile); ne_pfile = np.asarray(ne_pfile) * 1e20
            ne_pfile_on_ped = interp1d(psi_pfile, ne_pfile, kind='linear',
                                       bounds_error=False, fill_value='extrapolate')(psi_ped_grid)
            mean_ne_pfile = float(np.mean(ne_pfile_on_ped))

            # Sweep .npy successes, keep minimum-L2 profile
            best_l2 = np.inf
            best_x = None
            best_ne = None
            for npy_path in Path(ne_success_fp).glob("ne_*.npy"):
                sol_i = np.load(npy_path, allow_pickle=True).item()
                x_i = np.asarray(sol_i['x'], dtype=float)
                ne_i = np.asarray(sol_i['y'], dtype=float)
                ne_on_ped = interp1d(x_i, ne_i, kind='linear',
                                     bounds_error=False, fill_value='extrapolate')(x_ped_grid)
                l2 = float(np.sqrt(np.mean((ne_on_ped - ne_pfile_on_ped) ** 2)) / mean_ne_pfile)
                if l2 < best_l2:
                    best_l2, best_x, best_ne = l2, x_i, ne_i
            if best_ne is None:
                raise RuntimeError(f"No successful runs found in {ne_success_fp}")
            print(f"Best normalized L2 = {best_l2:.4f}")

        # --- Feed best profile into EPEDNN --------------------------------
        if eped_iter == 0:
            pedestal_height_prev = 0.0
            pedestal_width_prev = 0.0
        else:
            pedestal_height_prev, pedestal_width_prev = pedestal_height, pedestal_width
        pedestal_height, pedestal_width = base_model.feed_epednn(ne_ped=best_ne, x_epednn=best_x)
        print(f"Pedestal height: {pedestal_height} MPa, Pedestal width: {pedestal_width} (psi_N)")

        if eped_iter > 0:
            eped_tol = ((pedestal_height - pedestal_height_prev) / pedestal_height_prev
                        + (pedestal_width - pedestal_width_prev) / pedestal_width_prev)
            print(f"Normalized pedestal pressure height and width tolerance: {eped_tol}")
            if eped_tol < eped_tol_max:
                break

        # --- New T_e profile (EPED1 tanh form, Eq. 1b without core H term) ---
        #   T(psi) = T_sep + aT0 * { tanh[2(1 - psi_mid)/Delta]
        #                          - tanh[2(psi - psi_mid)/Delta] }
        # On [psi_ped, 1] the tanh shape peaks at psi_ped; aT0 is fixed so
        # T(psi_ped) = Te_ped derived from EPED pedestal pressure (p_ped =
        # 2 * ne_ped * Te_ped, Ti = Te, Zeff ~ 1).
        Delta = float(pedestal_width)
        psi_mid = 1.0 - 0.5 * Delta
        psi_ped = 1.0 - Delta
        ne_ped_val = float(interp1d(best_x, best_ne, kind='linear',
                                    bounds_error=False, fill_value='extrapolate')(
                                        psi_to_x(psi_ped)))
        T_sep_eV = float(np.asarray(base_model.T_e)[-1] * 1e3)  # keV -> eV
        eV_to_J = 1.602176634e-19
        Te_ped_eV = (float(pedestal_height) * 1.0e6 / (2.0 * ne_ped_val)) / eV_to_J
        tanh_peak = 2.0 * np.tanh(1.0)  # shape-function maximum on [psi_ped, 1]
        aT0 = (Te_ped_eV - T_sep_eV) / tanh_peak

        def _eped1_tanh_Te(psi_N):
            return T_sep_eV + aT0 * (
                np.tanh(2.0 * (1.0 - psi_mid) / Delta)
                - np.tanh(2.0 * (psi_N - psi_mid) / Delta)
            )

        # EPED tanh only on the pedestal strip [psi_ped, 1]; splice onto p-file core.
        psi_tanh = np.linspace(psi_ped, 1.0, x_res)
        Te_tanh_eV = _eped1_tanh_Te(psi_tanh)

        psi_prev = np.asarray(base_model.psi_Te_eval, dtype=float)
        Te_prev_keV = np.asarray(base_model.T_e, dtype=float)

        # Add offset to T_e core profile
        Te_prev_keV_ped = interp1d(psi_prev, Te_prev_keV, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_ped)
        Te_tanh_eV_ped = interp1d(psi_tanh, Te_tanh_eV / 1e3, kind='linear', bounds_error=False, fill_value='extrapolate')(psi_ped)
        T_e_offset = Te_tanh_eV_ped - Te_prev_keV_ped # ped - core at psi_ped

        keep = psi_prev < psi_ped
        psi_N_Te_new = np.concatenate([psi_prev[keep], psi_tanh])
        T_prof_keV = np.concatenate([Te_prev_keV[keep] + T_e_offset, Te_tanh_eV / 1e3])

        psi_N_inner_boundary_new = psi_ped

        if True:
            Te_spliced_eV = T_prof_keV * 1e3
            print(f"  Te_ped = {Te_ped_eV:.1f} eV, T_sep = {T_sep_eV:.1f} eV "
                  f"(ne_ped = {ne_ped_val:.3e} m^-3, psi_ped = {psi_ped:.4f}, "
                  f"Delta = {Delta:.4f})")
            print(f"  psi_N_inner_boundary_new = {psi_N_inner_boundary_new:.4f}")
            og_Te_peak = float(interp1d(psi_prev, Te_prev_keV * 1e3, kind='linear',
                                        bounds_error=False, fill_value='extrapolate')(psi_ped))
            print(f"Percent change from previous T_e at psi_ped = "
                  f"{((Te_ped_eV - og_Te_peak) / og_Te_peak):.4f}")
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(psi_N_Te_new, Te_spliced_eV, lw=2, label='New T_e profile')
            ax.plot(psi_prev, Te_prev_keV * 1e3, lw=2, ls='--', label='Previous T_e profile')
            ax.axvline(psi_ped, color='k', ls='--', lw=0.8, label=fr'$\psi_{{ped}}={psi_ped:.3f}$')
            ax.axhline(Te_ped_eV, color='r', ls=':', lw=0.8, label=fr'$T_{{e,ped}}={Te_ped_eV:.0f}$ eV')
            ax.set_xlabel(r'$\psi_N$'); ax.set_ylabel(r'$T_e$ [eV]')
            ax.set_title(f'EPED1 tanh T_e profile (iter {eped_iter})')
            ax.legend(); ax.grid(alpha=0.3)
            ax.set_xlim(0.8, 1.0)
            fig.tight_layout(); plt.show()

        # MAKE THIS PART FASTER
        base_model = saarelma_connor_nondim( # reset base_model to the new T_e profile and related quantities
            P_tot_e      = P_tot_e,
            alpha_crit   = round(float(alpha_crits[0]), 3),
            C_KBM        = round(float(C_KBMs[0]), 3),
            De_chie_etg  = round(float(De_chie_etgs[0]), 3),
            nFC_x0       = round(float(nFC_x0s[0]), 3),
            ncx_x0_ratio = round(float(ncx_x0_ratios[0]), 3),
            mhd_fp       = MHD_FP,
            kprof_fp     = KPROF_FP,
            verbose      = verbose,
            psi_N_inner_boundary = psi_N_inner_boundary_new,
            T_e_source = 'epednn',
            T_prof = T_prof_keV,
            T_prof_psi_N = psi_N_Te_new,
        )
        base_model.setup_epednn()