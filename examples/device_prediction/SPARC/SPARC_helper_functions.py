import os
import matplotlib.pyplot as plt
import numpy as np
import sys
from pathlib import Path
import tempfile
import urllib.request # needed for geqdsk import
ROOT = Path.cwd().parent.parent.parent
sys.path.insert(0, str(ROOT))
from src.profiles_loop_solve import profiles_loop_solve

tokamaker_python_path = os.getenv('OFT_ROOTPATH')
if tokamaker_python_path is not None:
    sys.path.append(os.path.join(tokamaker_python_path,'python'))
from OpenFUSIONToolkit import OFT_env
from OpenFUSIONToolkit.TokaMaker import TokaMaker
from OpenFUSIONToolkit.TokaMaker.meshing import load_gs_mesh
from OpenFUSIONToolkit.TokaMaker.util import create_isoflux, read_eqdsk


def read_sparcpublic_profiles(path):
    """
    Parse a SPARCPublic-style profile text file (TRANSP or CGYRO layout).

    Format: blocks headed by ``# name | unit``, each followed by
    ``index  value`` rows.

    Returns
    -------
    dict
        Keys for available quantities, each an ndarray. Always includes
        ``rho``. Electron/ion profiles added when present:
          - ne  [m^-3]
          - Te  [keV]
          - Ti  [keV]   (if present)
          - ni  [m^-3]  (if present)
        Also keeps raw auxiliaries ``polflux``, ``q`` when present, plus
        ``units`` metadata.
    """
    path = Path(path)
    sections = {}
    current = None
    current_unit = ''
    values = []

    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            if line.startswith('#'):
                if current is not None:
                    sections[current] = {
                        'data': np.asarray(values, dtype=float),
                        'unit': current_unit,
                    }
                # "# te | keV"  or  "# rho | -"
                body = line[1:].strip()
                name, _, unit = body.partition('|')
                current = name.strip().lower()
                current_unit = unit.strip()
                values = []
            else:
                parts = line.split()
                if len(parts) < 2:
                    continue
                values.append(float(parts[-1]))

    if current is not None:
        sections[current] = {
            'data': np.asarray(values, dtype=float),
            'unit': current_unit,
        }

    if 'rho' not in sections:
        raise KeyError(f'No rho grid found in {path}. Sections: {list(sections)}')

    rho = sections['rho']['data']
    out = {
        'rho': rho,
        'units': {'rho': sections['rho']['unit']},
    }
    for key in ('polflux', 'q'):
        if key in sections:
            out[key] = sections[key]['data']
            out['units'][key] = sections[key]['unit']

    # Temperatures: file stores keV
    for src, dst in (('te', 'Te'), ('ti', 'Ti')):
        if src in sections:
            if len(sections[src]['data']) != len(rho):
                raise ValueError(
                    f'{src} length {len(sections[src]["data"])} != rho length {len(rho)}'
                )
            out[dst] = sections[src]['data'].copy()
            out['units'][dst] = 'keV'

    # Densities: convert known file units -> m^-3
    def _density_to_si(name, unit, data):
        u = unit.lower().replace(' ', '')
        if '10^20' in u or '10**20' in u or '1e20' in u:
            return data * 1e20
        if '10^19' in u or '10**19' in u or '1e19' in u:
            return data * 1e19
        if u in ('m^-3', 'm-3', '1/m^3', '/m^3'):
            return data.copy()
        raise ValueError(f'Unrecognized {name} unit {unit!r} in {path}')

    for src, dst in (('ne', 'ne'), ('ni', 'ni')):
        if src in sections:
            if len(sections[src]['data']) != len(rho):
                raise ValueError(
                    f'{src} length {len(sections[src]["data"])} != rho length {len(rho)}'
                )
            out[dst] = _density_to_si(src, sections[src]['unit'], sections[src]['data'])
            out['units'][dst] = 'm^-3'

    return out


def psi_n_and_rho_psi(profiles):
    """Return (psi_N, rho_psi=sqrt(psi_N)) from polflux."""
    if 'polflux' not in profiles:
        raise KeyError('Profile file needs polflux to map rho -> psi_N correctly')
    pf = profiles['polflux']
    psi_N = (pf - pf[0]) / (pf[-1] - pf[0])
    rho_psi = np.sqrt(np.clip(psi_N, 0.0, None))
    return psi_N, rho_psi


def build_manual_profs(profiles):
    """manual_profs for kprof_loc='manual rho grid' with psi_N-consistent rho."""
    _, rho_psi = psi_n_and_rho_psi(profiles)
    manual_profs = {
        'Te': profiles['Te'],
        'rho_Te': rho_psi,
        'ne': profiles['ne'] / 1e20,   # m^-3 -> 10^20 m^-3 (solver multiplies by 1e20)
        'rho_ne': rho_psi,
    }
    if 'Ti' in profiles:
        manual_profs['Ti'] = profiles['Ti']
        manual_profs['rho_Ti'] = rho_psi
    if 'ni' in profiles:
        manual_profs['ni'] = profiles['ni'] / 1e20
        manual_profs['rho_ni'] = rho_psi
    return manual_profs


def calc_pressure_profile(profiles):
    """
    Thermal pressure from kinetic profiles on the polflux psi_N grid.

    Defaults to ``p = 2 * ne * Te`` (Te = Ti, ne = ni). If ``Ti`` is present,
    uses ``p = ne * Te + ni * Ti``, with ``ni = ne`` when ``ni`` is missing.

    Parameters
    ----------
    profiles : dict
        Output of ``read_sparcpublic_profiles`` (ne [m^-3], Te/Ti [keV]).

    Returns
    -------
    psi_N : ndarray
    p : ndarray
        Pressure in Pa.
    mode : str
        Formula used.
    """
    eV_to_J = 1.602176634e-19
    psi_N, _ = psi_n_and_rho_psi(profiles)
    ne = np.asarray(profiles['ne'], dtype=float)
    Te_J = np.asarray(profiles['Te'], dtype=float) * 1e3 * eV_to_J  # keV -> J

    if 'Ti' in profiles:
        Ti_J = np.asarray(profiles['Ti'], dtype=float) * 1e3 * eV_to_J
        if 'ni' in profiles:
            ni = np.asarray(profiles['ni'], dtype=float)
            mode = 'ne*Te + ni*Ti'
        else:
            ni = ne
            mode = 'ne*Te + ne*Ti (ni=ne)'
        p = ne * Te_J + ni * Ti_J
    else:
        p = 2.0 * ne * Te_J
        mode = '2*ne*Te'

    return psi_N, p, mode


def _print_profile_summary(label, path, profiles):
    psi_N, rho_psi = psi_n_and_rho_psi(profiles)
    print(f'Loaded {path.name} ({label}): {len(profiles["rho"])} points')
    print(f'  quantities: {[k for k in profiles if k != "units"]}')
    print(f'  psi_N mapping: using polflux -> rho_psi=sqrt(psi_N) '
          f'(max |rho_file^2 - psi_N| = {np.max(np.abs(profiles["rho"]**2 - psi_N)):.3f})')
    print(f'  Te axis  = {profiles["Te"][0]:.2f} keV,  edge = {profiles["Te"][-1]:.2f} keV')
    print(f'  ne axis  = {profiles["ne"][0]/1e20:.2f}e20 m^-3,  '
          f'edge = {profiles["ne"][-1]/1e20:.2f}e20 m^-3')
    if 'Ti' in profiles:
        print(f'  Ti axis  = {profiles["Ti"][0]:.2f} keV,  edge = {profiles["Ti"][-1]:.2f} keV')
    else:
        print('  Ti: not in file')
    if 'ni' in profiles:
        print(f'  ni axis  = {profiles["ni"][0]/1e20:.2f}e20 m^-3')
    else:
        print('  ni: not in file')


def read_popcon(path, keys):
    """
    Read named entries from the SPARCPublic POPCON CSV.

    Returns
    -------
    dict
        key -> {'value': float, 'unit': str}
    """
    import csv

    wanted = {k.lower(): k for k in keys}
    found = {}
    with open(path, newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = (row.get('Variable') or row.get('variable') or '').strip()
            if name.lower() not in wanted:
                continue
            val = float(row.get('Value') or row.get('value'))
            unit = (row.get('Unit') or row.get('unit') or '').strip()
            found[wanted[name.lower()]] = {'value': val, 'unit': unit}

    missing = [k for k in keys if k not in found]
    if missing:
        raise KeyError(f'Missing {missing} in {path}')
    return found


def _to_watts(value, unit):
    unit_to_W = {'': 1.0, 'w': 1.0, 'kw': 1e3, 'mw': 1e6, 'gw': 1e9}
    key = (unit or 'W').strip().lower()
    if key not in unit_to_W:
        raise ValueError(f'Unrecognized power unit {unit!r}')
    return value * unit_to_W[key]