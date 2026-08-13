# Run only EPEDNN

import sys
import csv
from collections import defaultdict
import numpy as np
from scipy.interpolate import interp1d
from pathlib import Path

ROOT = Path.cwd().parent.parent
sys.path.insert(0, str(ROOT))
from src.solver_nondim import saarelma_connor_nondim
from src.load_equil import initialize_inputs

out_dir = 'EPEDNN_pure_output_v2/'

gfile_fps = '/mnt/homes_global/jal2351/software/sc_inputs/gHighPerfHMode/'
kprof_fps = '/mnt/homes_global/jal2351/software/sc_inputs/OMFITnc_HighPerfHMode/'

# Fallback removed: equilibria without a fit ped-top are skipped (see loop below).
# Pedestal-top psiN from run_fit_pedestals.py (p_tot, gradient method).
# EPEDNN neped / zeffped must be taken here — NOT at a fixed psiN=0.95, and
# NOT with Zeff=1 (IDA ped Zeff is typically ~1.5-1.9).
PED_TOP_CSV = (
    '/mnt/homes_global/jal2351/software/sc_inputs/'
    'pedestal_tops_summary_HighPerfHMode.csv'
)

epednn_model = 'EPED1'

ne_ped_source = 'Oak'


def load_ped_tops(csv_path=PED_TOP_CSV):
    """shot -> list of {time_ms, ped_top_grad, ped_top_1mw, grad_width}."""
    by_shot = defaultdict(list)
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(
            f'Missing ped-top CSV: {csv_path}\n'
            f'Run sc_inputs/run_fit_pedestals.py first.'
        )
    with open(path) as f:
        for row in csv.DictReader(f):
            if row.get('bad') in ('1', 'True', 'true') or not row.get('ped_top_grad'):
                continue
            by_shot[int(row['shot'])].append({
                'time_ms': float(row['time_ms']),
                'ped_top_grad': float(row['ped_top_grad']),
                'ped_top_1mw': float(row['ped_top_1mw']),
                'grad_width': float(row['grad_width']),
            })
    if not by_shot:
        raise RuntimeError(f'No usable ped-top rows in {csv_path}')
    return by_shot


def ped_top_for_tag(equil_tag, ped_tops, prefer='ped_top_grad'):
    """Map g-file tag '125729.03589' -> nearest fit ped-top psiN, or None."""
    shot_s, time_s = equil_tag.split('.', 1)
    shot = int(shot_s)
    time_ms = float(time_s)  # '03589' -> 3589 ms
    rows = ped_tops.get(shot)
    if not rows:
        return None
    return min(rows, key=lambda r: abs(r['time_ms'] - time_ms))[prefer]


def OMFITnc_load(filename):
    """
    Reads an OMFITnc file, averages kinetic profiles across the first dimension,
    and interpolates them onto a common, unified psi_N grid.

    Parameters:
        filename (str): Path to the NetCDF file.

    Returns:
        tuple: (psi_N_unified, Te_1d, ne_1d, p_ion_1d, Zeff_1d) as 1D numpy arrays.
        Te_1d in keV, ne_1d in m^-3, p_ion_1d in Pa, Zeff_1d dimensionless.

    Notes:
        IDA OMFITnc files typically lack T_i / n_i. Ion pressure is taken from
        p_ion when present (checked O(1) vs pe = ne*Te*e); otherwise pe is used.
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

    # Zeff profile (needed for EPEDNN zeffped). Fall back to 1.0 if absent.
    if _nc_has('Zeff'):
        zeff_raw = _reduce_time(_nc_array('Zeff'))
        psi_z, zeff_raw = _clip_profile(psi_ne, zeff_raw)
        Zeff_unified = interp1d(
            psi_z, zeff_raw, kind='cubic', bounds_error=False, fill_value='extrapolate'
        )(psi_N_unified)
    else:
        Zeff_unified = np.ones_like(psi_N_unified)

    return psi_N_unified, Te_unified, ne_unified, p_ion_unified, Zeff_unified


ped_tops = load_ped_tops()
equil_num = len(list(Path(kprof_fps).glob('*.cdf')))
equilibria = initialize_inputs(equil_num, gfile_fps, kprof_fps, p_filetype='OMFITnc')

Path(out_dir).mkdir(parents=True, exist_ok=True)

if ne_ped_source == 'Oak':
    _oak = np.load('Oak_pedestal_fits.npy', allow_pickle=True).item()
    ne_peds = np.asarray(_oak['n_e'], dtype=float) / 1e19  # m^-3 -> 10^19 m^-3
    equil_shotandtime_ne = list(_oak['equil_shotandtime'])

i = 0
n_skip = 0
for mhd_fp, kprof_fp in equilibria:
    equil_tag = Path(mhd_fp).name[1:]
    psiN_ped = ped_top_for_tag(equil_tag, ped_tops, prefer='ped_top_grad')
    if psiN_ped is None or not np.isfinite(psiN_ped):
        print(f'  SKIP {equil_tag}: no fit ped-top in {PED_TOP_CSV}')
        n_skip += 1
        continue

    psi_N_unified, Te_unified, ne_unified, p_ion_unified, Zeff_unified = OMFITnc_load(
        kprof_fp
    )
    if ne_ped_source == 'John':
        ne_ped = np.interp(psiN_ped, psi_N_unified, ne_unified)  # m^-3
        ne_ped = ne_ped / 1e19  # m^-3 -> 10^19 m^-3 (EPEDNN input)
    elif ne_ped_source == 'Oak':
        # Match shot exactly; pick closest time (g-tag ms vs Oak float ms).
        # equil_tag: '125729.03589' ; equil_shotandtime_ne: '125729.3589.5480'
        shot_s, time_s = equil_tag.split('.', 1)
        shot = int(shot_s)
        time_ms = float(time_s)
        best_i, best_dt = None, None
        for i_oak, tag_oak in enumerate(equil_shotandtime_ne):
            s_oak, t_oak = str(tag_oak).split('.', 1)
            if int(s_oak) != shot:
                continue
            dt = abs(float(t_oak) - time_ms)
            if best_dt is None or dt < best_dt:
                best_dt = dt
                best_i = i_oak
        if best_i is None:
            print(f'  SKIP {equil_tag}: no Oak ne_ped for shot {shot}')
            n_skip += 1
            continue
        ne_ped = float(ne_peds[best_i])
        if not np.isfinite(ne_ped):
            print(f'  SKIP {equil_tag}: Oak ne_ped is nan (idx={best_i})')
            n_skip += 1
            continue
    zeff_ped = float(np.interp(psiN_ped, psi_N_unified, Zeff_unified))
    e_charge = 1.602176634e-19  # C
    # Total kinetic pressure profile [Pa]: p_ion + pe, both on psi_N_unified
    # pe = ne[m^-3] * Te[eV] * e; Te_unified is keV so *1e3
    kfile_pres = p_ion_unified + ne_unified * (Te_unified * 1e3) * e_charge

    # Run EPEDNN
    base_model = saarelma_connor_nondim(
        P_tot_e      = 5e6,  # W
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
    # feed_epednn maps self.Z_i -> zeffped. Override AFTER __init__ so ion
    # charge e_i (set from the constructor default Z_i=1 for D) stays correct.
    base_model.Z_i = zeff_ped
    base_model.setup_epednn(model=epednn_model)
    pedestal_height, pedestal_width, betan = base_model.feed_epednn(
        model=epednn_model, ne_ped=ne_ped, EPEDNN_core='pfile'
    )
    np.save(
        Path(out_dir) / Path(equil_tag + '.npy'),
        {
            'pedestal_height': pedestal_height,
            'pedestal_width': pedestal_width,
            'betan': betan,
            'kfile_pres': kfile_pres,
            'psiN_kfilepres': psi_N_unified,
            'psiN_ped_exp': psiN_ped,
            'ne_ped_1e19': ne_ped,
            'zeff_ped': zeff_ped,
        },
    )
    i += 1
    if i % 25 == 0:
        print(f'  {i}/{len(equilibria)}  last={equil_tag}  '
              f'psiN_ped={psiN_ped:.4f}  zeff={zeff_ped:.3f}  '
              f'p_h={float(pedestal_height)*1e3:.2f} kPa')

print(f'Done: saved {i}/{len(equilibria)} equilibria to {out_dir} '
      f'(skipped {n_skip} without ped-top fit)')
