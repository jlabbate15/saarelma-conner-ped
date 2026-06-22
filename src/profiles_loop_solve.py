# Loop between EPEDNN and coupled version of Saarelma-Connor model to find self-consistent pedestal

import sys
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import os
import csv
import re
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
    x_res = 20,
    N = 3, # Saarelma-Connor free param scan size
    P_tot_e = 5e6, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
    psi_N_inner = 0.85,
    alpha_crits_minmax = [-1,1],
    C_KBMs_minmax = [-1,1],
    De_chie_etgs_minmax = [-1,1],
    nFC_x0s_minmax = [14,17],
    ncx_x0_ratios_minmax = [0.1,1.25],
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
            mhd_fp       = MHD_FP,
            kprof_fp     = KPROF_FP,
            verbose      = verbose,
            # psi_N_inner_boundary = 0.85, # set to None to use adaptive inner boundary method
    )
    base_model.setup_epednn()
    print("Base model built.")

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
            j=0
            for C_KBM in C_KBMs:
                    k=0
                    for De_chie_etg in De_chie_etgs:
                            l=0
                            for nFC_x0 in nFC_x0s:
                                    for ncx_x0_ratio in ncx_x0_ratios:
                                            ac = round(float(alpha_crit), 3)
                                            ck = round(float(C_KBM), 3)
                                            de = round(float(De_chie_etg), 3)
                                            nf = round(float(nFC_x0), 3)
                                            nc = round(float(ncx_x0_ratio),3)

                                            base_model.update_free_params(
                                                    alpha_crit    = ac,
                                                    C_KBM         = ck,
                                                    De_chie_etg   = de,
                                                    nFC_x0        = nf,
                                                    ncx_x0_ratio  = nc,
                                            )
                                            try:
                                                    base_model.update_free_params(
                                                            alpha_crit            = ac,
                                                            C_KBM                 = ck,
                                                            De_chie_etg           = de,
                                                            nFC_x0                = nf,
                                                            psi_N_inner_boundary  = psi_N_inner,
                                                            ncx_x0_ratio  = nc,
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
            print(f"Completed {i} of {len(alpha_crits)} alpha_crits") # progress logging

    if (len(alpha_crits)+len(De_chie_etgs)+len(C_KBMs)+len(nFC_x0s)+len(ncx_x0_ratios))<=1:
        pass # not implemented for a single free parameter run yet
    else: # requires a pfile input
        if KPROF_FP is None:
            raise ValueError: "Must specify pfile for a parameter scan implementation of SC model in EPEDNN loop"
        # Compute best n_e profile out of free parameter runs
        scan_dir = Path(ne_success_fp)
        pattern = re.compile(r"_([a-zA-Z]+)([\d.eE+-]+)")

        # Read p-file density
        def read_pfile_ne(path):
            psi_arr, ne_arr = [], []
            in_ne_block = False
            with open(path) as f:
                for line in f:
                    if '3 N Z A' in line:
                        break
                    if line.startswith('201'):
                        in_ne_block = 'ne(10^20/m^3)' in line
                        continue
                    if in_ne_block:
                        psi, val, _ = line.split()
                        psi_arr.append(float(psi))
                        ne_arr.append(float(val))
            return np.array(psi_arr), np.array(ne_arr) * 1e20

        psi_pfile, ne_pfile = read_pfile_ne(KPROF_FP)

        # Build psi_N -> x mapping
        ref = saarelma_connor_nondim(
            P_tot_e=P_tot_e,
            alpha_crit=alpha_crits[0],
            C_KBM=C_KBMs[0],
            De_chie_etg=De_chie_etgs[0],
            nFC_x0=nFC_x0s[0],
            mhd_fp=MHD_FP,
            kprof_fp=KPROF_FP,
            ncx_x0_ratio=ncx_x0_ratios[0],
            verbose=False,
        )
        psi_to_x = interp1d(ref.psi_N_pres, ref.r_psi - ref.r_psi[-1],
                            kind='linear', bounds_error=False, fill_value='extrapolate')
        x_to_psi = interp1d(ref.r_psi - ref.r_psi[-1], ref.psi_N_pres,
                            kind='linear', bounds_error=False, fill_value='extrapolate')

        # Compute pedestal grid
        psi_ped_grid = np.linspace(0.85, 1.0, 200)
        x_ped_grid = psi_to_x(psi_ped_grid)
        ne_pfile_on_ped = interp1d(psi_pfile, ne_pfile, kind='linear',
                                bounds_error=False, fill_value='extrapolate')(psi_ped_grid)
        mean_ne_pfile = float(np.mean(ne_pfile_on_ped))

        # Load success records
        success_records = []
        for npy_path in scan_dir.glob("ne_*.npy"):
            pairs = pattern.findall(npy_path.stem)
            params = {letter: float(num) for letter, num in pairs}
            sol = np.load(npy_path, allow_pickle=True).item()
            ne_pred_on_ped = interp1d(sol['x'], sol['y'], kind='linear',
                                    bounds_error=False, fill_value='extrapolate')(x_ped_grid)
            l2 = np.sqrt(np.mean((ne_pred_on_ped - ne_pfile_on_ped) ** 2))
            success_records.append([
                params.get('a'), params.get('C'), params.get('D'),
                params.get('n'), params.get('nc'), params.get('b'), l2 / mean_ne_pfile,
            ])
        success_arr_summary = np.array(success_records) if success_records else np.empty((0, 7))

        # Load failure records
        failure_records = []
        try:
            with open(failure_fp, newline="") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    failure_records.append([float(x.strip()) for x in row])
        except FileNotFoundError:
            pass
        failure_arr_summary = np.array(failure_records) if failure_records else np.empty((0, 6))

        param_values = [
            np.round(alpha_crits.astype(float), 3),
            np.round(C_KBMs.astype(float), 3),
            np.round(De_chie_etgs.astype(float), 3),
            np.round(nFC_x0s.astype(float), 3),
            np.round(ncx_x0_ratios.astype(float), 3),
            np.array(psi_N_inner),
        ]

        def _record_to_bins(record6):
            """Map one run to 6 scan-bin indices, or None if any parameter is off-grid."""
            bins = []
            for d in range(6):
                grid = param_values[d]
                val = float(record6[d])
                i = int(np.argmin(np.abs(grid - val)))
                if d == 5:  # psi_inner
                    on_grid = np.isclose(grid[i], val, atol=5e-4)
                elif d == 4:  # ncx_x0_ratio
                    on_grid = np.isclose(grid[i], val, rtol=1e-2)
                elif d == 3:  # nFC_x0
                    on_grid = np.isclose(grid[i], val, rtol=1e-3)
                else:
                    on_grid = np.isclose(grid[i], val, rtol=0, atol=1e-6)
                if not on_grid:
                    return None
                bins.append(i)
            return tuple(bins)

        # Build run outcomes
        run_outcomes = {}
        for record in success_arr_summary:
            bins6 = _record_to_bins(record[:6])
            if bins6 is None:
                continue
            run_outcomes[bins6] = {"success": True, "l2": float(record[6])}

        for record in failure_arr_summary:
            bins6 = _record_to_bins(record[:6])
            if bins6 is None:
                continue
            run_outcomes.setdefault(bins6, {"success": False})

        print(f"{sum(1 for o in run_outcomes.values() if o['success'])} succeeded")

        profile_runs = []
        for npy_path in sorted(scan_dir.glob("ne_*.npy")):
            pairs = pattern.findall(npy_path.stem)
            params = {letter: float(num) for letter, num in pairs}
            record6 = [params.get("a"), params.get("C"), params.get("D"),
                    params.get("n"), params.get("nc"), params.get("b")]
            bins6 = _record_to_bins(record6)
            if bins6 is None:
                continue

            sol = np.load(npy_path, allow_pickle=True).item()
            x_sol = np.asarray(sol['x'], dtype=float)
            ne_sol = np.asarray(sol['y'], dtype=float)
            ne_pred_on_ped = interp1d(
                x_sol, ne_sol, kind="linear", bounds_error=False, fill_value="extrapolate",
            )(x_ped_grid)
            l2_norm = float(np.sqrt(np.mean((ne_pred_on_ped - ne_pfile_on_ped) ** 2)) / mean_ne_pfile)

            label = (
                rf"$\alpha={params['a']:.3g}$, $C={params['C']:.3g}$, $D={params['D']:.3g}$, "
                rf"$n_{{FC,0}}={params['n']:.2e}$, $n_{{CX}}/n_{{FC}}={params['nc']:.3g}$, "
                rf"$\psi_N={params['b']:.4f}$, L2={l2_norm:.3f}"
            )
            entry = dict(l2=l2_norm, label=label, x=x_sol, ne=ne_sol, bins6=bins6)
            prev = next((r for r in profile_runs if r["bins6"] == bins6), None)
            if prev is None or l2_norm < prev["l2"]:
                if prev is not None:
                    profile_runs.remove(prev)
                profile_runs.append(entry)

        profile_runs.sort(key=lambda r: r["l2"])
        best_profile = profile_runs[-1]

        # Plug best profile into EPEDNN