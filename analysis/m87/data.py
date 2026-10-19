"""EHT MWL 2017 SED points (data/obs_data/*.dat) and the instrument -> aperture assignment.

File columns: nu [Hz], nuFnu [erg s^-1 cm^-2], err (err < 0 marks an upper limit at nuFnu).
"""
from dataclasses import dataclass

import numpy as np

from .paths import DATA

# Aperture radius R* [R_g] = half the FWHM of the circular restoring beam with which each peak flux was measured
# (EHT MWL Science WG 2021, Table A1 and the data release m87_2017_radio.csv; 1 mas = 261.9 R_g):
# EAVN 1.0/0.5 mas at 22/43 GHz, HSA 1.5/1.0/1.0 mas at 8.4/15/24 GHz, VERA and KVN 1.0 mas,
# VLBA 1.0/0.5 mas at 24/43 GHz, GMVA 0.2 mas, EVN 5.0 mas. EHT: 7.3 R_g kept (the flux is from a 60x60 muas box,
# half-side 7.9 R_g). ALMA: 3.5e5 R_g = the whole jet. SMA, optical/UV and X-rays: whole jet (None).
APERTURES_RG = [7.3, 26.0, 65.0, 131.0, 195.0, 650.0, 3.5e5]
APERTURE_OF = {"EHT": 7.3, "GMVA": 26.0, "EVN": 650.0, "ALMA": 3.5e5}
APERTURE_BY_NU = {  # instrument -> {nu [Hz]: R* [R_g]}; nu matched within 1 per cent
    "EAVN": {22e9: 131.0, 43e9: 65.0},
    "HSA": {8.4e9: 195.0, 15e9: 131.0, 24e9: 131.0},
    "VERA": {22e9: 131.0},
    "VLBA": {24e9: 131.0, 43e9: 65.0},
    "KVN": {21.7e9: 131.0, 43.4e9: 131.0, 86.8e9: 131.0, 129e9: 131.0},
}

def aperture_of(name, nu):
    """Aperture R* [R_g] of one point; None = whole jet."""
    for nu0, r in APERTURE_BY_NU.get(name, {}).items():
        if abs(nu / nu0 - 1) < 0.01:
            return r
    assert name not in APERTURE_BY_NU, f"{name} at {nu:.3e} Hz has no aperture in APERTURE_BY_NU"
    return APERTURE_OF.get(name)

BANDS = {  # band used in the chi^2 table; gamma rays are not modelled (synchrotron only) and are only shown
    "EHT": "radio", "GMVA": "radio", "EAVN": "radio", "HSA": "radio", "EVN": "radio", "ALMA": "radio",
    "KVN": "radio", "VERA": "radio", "VLBA": "radio", "SMA": "radio",
    "HST": "optuv", "Swift": "mixed",  # Swift: UVOT (optical/UV) points + one XRT upper limit
    "Chandra": "xray", "NuSTAR": "xray",
    "Fermi": "gamma", "HESS": "gamma", "MAGIC": "gamma", "VERITAS": "gamma",
}
GAMMA = [k for k, v in BANDS.items() if v == "gamma"]


@dataclass
class Obs:
    name: str
    nu: np.ndarray
    nufnu: np.ndarray
    err: np.ndarray      # 1-sigma error; nan for upper limits
    uplim: np.ndarray    # bool

    @property
    def apertures_rg(self):
        """Aperture R* [R_g] per point; None = whole jet."""
        return [aperture_of(self.name, nu) for nu in self.nu]

    @property
    def tagged(self):
        """True if every point has an aperture (compared with a cut jet, drawn in the radio panel)."""
        return all(r is not None for r in self.apertures_rg)

    def band(self, i):
        b = BANDS[self.name]
        if b == "mixed":  # Swift: UVOT below 1e16 Hz, XRT above
            return "optuv" if self.nu[i] < 1e16 else "xray"
        return b


def load_obs(names=None):
    """{instrument: Obs}, sorted by name (deterministic colours/markers)."""
    out = {}
    for f in sorted(DATA.glob("*.dat")):
        if names is not None and f.stem not in names:
            continue
        nu, nufnu, err = np.loadtxt(f, skiprows=1, ndmin=2).T
        ul = err < 0
        out[f.stem] = Obs(f.stem, nu, nufnu, np.where(ul, np.nan, err), ul)
    return out
