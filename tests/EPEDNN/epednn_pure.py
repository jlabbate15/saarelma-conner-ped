# Run only EPEDNN

import sys
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(ROOT))
from src.solver_nondim import saarelma_connor_nondim
from src.load_equil import initialize_inputs

out_dir = 'EPEDNN_pure_output/'

gfile_fps = '/mnt/homes_global/jal2351/software/sc_inputs/gHighPerfHMode/'
kprof_fps = '/mnt/homes_global/jal2351/software/sc_inputs/OMFITnc_HighPerfHMode/'

psiN_ped_est = 0.95

epednn_model = 'EPED1'

def OMFITnc_load(filename):
    """
    Reads an OMFITnc file, averages kinetic profiles across the first dimension,
    and interpolates them onto a common, unified psi_N grid.

    Parameters:
        filename (str): Path to the NetCDF file.

    Returns:
        tuple: (psi_N_unified, Te_1d, ne_1d, p_ion_1d) as 1D numpy arrays.
        Te_1d in keV, ne_1d in m^-3, p_ion_1d in Pa.

    Notes:
        IDA OMFITnc files typically lack T_i / n_i. Ion pressure is taken from
        p_ion when present (checked O(1) vs pe = ne*Te*e); otherwise pe is used.

    # Example usage:
    # psi_grid, Te, ne, p_ion = OMFITnc_load('my_plasma_data.cdf')
    """
    from omfit_classes.omfit_nc import OMFITnc
    nc = OMFITnc(filename)
    e_charge = 1.602176634e-19  # C

    def _nc_array(var_name):
        """OMFITnc variables are SortedDicts; numeric data lives under 'data'."""
        var = nc[var_name]
        if hasattr(var, 'keys') and 'data' in var:
            return np.asarray(var['data'], dtype=float)
        return np.asarray(var, dtype=float)

    def _nc_has(var_name):
        return var_name in nc

    def _reduce_time(arr):
        """Average over a leading time axis if present."""
        arr = np.asarray(arr, dtype=float)
        if arr.ndim > 1:
            return np.mean(arr, axis=0)
        return arr

    # 1. Extract and average electron profiles
    Te_raw = _reduce_time(_nc_array('T_e'))
    ne_raw = _reduce_time(_nc_array('n_e'))

    # Convert Te to keV if the file stores eV (IDA OMFITnc files use eV)
    te_unit = ''
    if hasattr(nc['T_e'], 'keys') and 'unit' in nc['T_e']:
        te_unit = str(nc['T_e']['unit']).lower()
    if te_unit in ('ev', 'electron volt', 'electron-volt'):
        Te_raw = Te_raw / 1e3
    elif te_unit in ('kev',):
        pass
    elif np.nanmax(np.abs(Te_raw)) > 100:
        # Heuristic: values >> 100 are almost certainly eV, not keV
        Te_raw = Te_raw / 1e3

    # 2. Ion pressure from file if present; else fall back to pe after interpolation
    has_p_ion = _nc_has('p_ion')
    if has_p_ion:
        p_ion_raw = _reduce_time(_nc_array('p_ion'))  # Pa
    else:
        print(
            f"No p_ion found in {filename}. "
            f"Available keys: {[k for k in nc.keys() if not str(k).startswith('__')]}"
        )
        print('Using pe = ne*Te*e as p_ion')
        p_ion_raw = None

    # 3. Determine psi_N grids (files may use psi_n / psi_N / separate Te,ne grids)
    if _nc_has('psi_N_Te') and _nc_has('psi_N_ne'):
        psi_Te = _reduce_time(_nc_array('psi_N_Te'))
        psi_ne = _reduce_time(_nc_array('psi_N_ne'))
        psi_ion = (
            _reduce_time(_nc_array('psi_N_ion'))
            if _nc_has('psi_N_ion')
            else psi_ne
        )
    else:
        for psi_key in ('psi_n', 'psi_N', 'psiN', 'psin'):
            if _nc_has(psi_key):
                break
        else:
            raise KeyError(
                f"No psi_N grid found in {filename}. "
                f"Available keys: {[k for k in nc.keys() if not str(k).startswith('__')]}"
            )
        psi_shared = _reduce_time(_nc_array(psi_key))
        psi_Te = psi_ne = psi_ion = psi_shared

    # Keep only the closed-flux domain for interpolation onto [0, 1]
    def _clip_profile(psi, prof):
        psi = np.asarray(psi, dtype=float).ravel()
        prof = np.asarray(prof, dtype=float).ravel()
        order = np.argsort(psi)
        psi, prof = psi[order], prof[order]
        mask = (psi >= 0.0) & (psi <= 1.0)
        if np.count_nonzero(mask) < 2:
            mask = np.ones_like(psi, dtype=bool)
        # Drop duplicate psi points that break interp1d
        _, uniq = np.unique(psi[mask], return_index=True)
        uniq = np.sort(uniq)
        return psi[mask][uniq], prof[mask][uniq]

    psi_Te, Te_raw = _clip_profile(psi_Te, Te_raw)
    psi_ne, ne_raw = _clip_profile(psi_ne, ne_raw)
    if has_p_ion:
        psi_ion, p_ion_raw = _clip_profile(psi_ion, p_ion_raw)
        num_points = max(len(psi_Te), len(psi_ne), len(psi_ion))
    else:
        num_points = max(len(psi_Te), len(psi_ne))

    # 4. Unified evaluation grid on [0, 1]
    psi_N_unified = np.linspace(0.0, 1.0, num_points)

    # 5. Interpolate profiles onto the unified grid
    Te_unified = interp1d(
        psi_Te, Te_raw, kind='cubic', bounds_error=False, fill_value='extrapolate'
    )(psi_N_unified)
    ne_unified = interp1d(
        psi_ne, ne_raw, kind='cubic', bounds_error=False, fill_value='extrapolate'
    )(psi_N_unified)

    pe_unified = ne_unified * (Te_unified * 1e3) * e_charge  # Pa
    if has_p_ion:
        p_ion_unified = interp1d(
            psi_ion, p_ion_raw, kind='cubic', bounds_error=False, fill_value='extrapolate'
        )(psi_N_unified)
        # Sanity check: p_ion should be O(1) times electron pressure
        ratio_med = float(np.nanmedian(p_ion_unified / pe_unified))
        if not (0.3 < ratio_med < 5.0):
            raise ValueError(
                f"p_ion does not look like ion pressure in {filename}: "
                f"median(p_ion / (ne*Te*e)) = {ratio_med:.3g} (expected O(1))"
            )
    else:
        p_ion_unified = pe_unified

    return psi_N_unified, Te_unified, ne_unified, p_ion_unified


equil_num = len(list(Path(kprof_fps).glob('*.cdf')))
equilibria = initialize_inputs(equil_num, gfile_fps, kprof_fps, p_filetype='OMFITnc')

i=0
for mhd_fp, kprof_fp in equilibria:

    psi_N_unified, Te_unified, ne_unified, p_ion_unified = OMFITnc_load(kprof_fp)
    ne_ped = np.interp(psiN_ped_est, psi_N_unified, ne_unified)  # m^-3
    ne_ped = ne_ped / 1e19  # m^-3 -> 10^19 m^-3 (EPEDNN input)
    e_charge = 1.602176634e-19  # C
    # Total kinetic pressure profile [Pa]: p_ion + pe, both on psi_N_unified
    # pe = ne[m^-3] * Te[eV] * e; Te_unified is keV so *1e3
    kfile_pres = p_ion_unified + ne_unified * (Te_unified * 1e3) * e_charge

    # Run EPEDNN
    base_model = saarelma_connor_nondim(
        P_tot_e      = 5e6, # W, total heating power given to electrons (can be assumed to be half the total heating power according to S. Saarelma et al 2023 Nucl. Fusion 63 052002), will be read from TokTox
        alpha_crit   = 1,
        C_KBM        = 1,
        De_chie_etg  = 1,
        nFC_x0       = 1e15,
        ncx_x0_ratio = 1,
        psi_N_inner_boundary = 0.85,
        mhd_fp       = mhd_fp,
        kprof_fp     = kprof_fp,
        kprof_loc    = 'OMFITnc',
        verbose      = False,
    )
    base_model.setup_epednn(model=epednn_model)
    pedestal_height, pedestal_width, betan = base_model.feed_epednn(model=epednn_model, ne_ped=ne_ped, EPEDNN_core='pfile')
    equil_tag = Path(mhd_fp).name[1:]
    np.save(Path(out_dir) / Path(equil_tag + '.npy'), {'pedestal_height': pedestal_height, 'pedestal_width': pedestal_width, 'betan': betan, 'kfile_pres': kfile_pres, 'psiN_kfilepres': psi_N_unified})