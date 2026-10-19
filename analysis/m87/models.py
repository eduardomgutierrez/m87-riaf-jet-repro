"""Model spectra: jet + counter-jet per aperture (extended-jet run.x output) and the RIAF (thermal-riaf)."""
from dataclasses import dataclass, field

import numpy as np

from .constants import DIST
from .paths import RUNS


@dataclass
class Model:
    name: str
    nu: np.ndarray                   # [Hz]
    apertures_rg: list
    jet: np.ndarray                  # F_nu (jet + CJ) per aperture, shape (n_ap, n_nu) [erg s^-1 cm^-2 Hz^-1]
    jet_total: np.ndarray            # F_nu with no aperture cut
    riaf: np.ndarray = field(default=None)  # F_nu of the RIAF on the same grid (None = Jet-only)

    @property
    def total(self):
        """F_nu jet+CJ+RIAF per aperture (the full RIAF is added to every aperture)."""
        return self.jet + (0.0 if self.riaf is None else self.riaf)

    def aperture_index(self, r_rg):
        """Index of the aperture column r_rg; the column must exist (within 1 per cent), no snapping to a neighbour."""
        k = int(np.argmin(np.abs(np.log(np.array(self.apertures_rg) / r_rg))))
        assert abs(self.apertures_rg[k] / r_rg - 1) < 0.01, f"{self.name}: no aperture column for R* = {r_rg} R_g"
        return k

    def nufnu_obs(self, o):
        """Total nuFnu at each point of an Obs, each with its own aperture."""
        return np.array([self.nufnu_at(r, [nu])[0] for r, nu in zip(o.apertures_rg, o.nu)])

    def nufnu_at(self, r_rg, nu):
        """Total nuFnu for aperture r_rg (None = whole jet) at nu (log-log interpolation; 0 where the model is 0)."""
        f = self.jet_total + (0.0 if self.riaf is None else self.riaf) if r_rg is None \
            else self.total[self.aperture_index(r_rg)]
        y = self.nu * f
        pos = y > 0
        out = np.exp(np.interp(np.log(nu), np.log(self.nu[pos]), np.log(y[pos]), left=-np.inf, right=-np.inf))
        return out




def load_riaf(nu, run="riaf_m87"):
    """RIAF F_nu interpolated (log-log) onto nu; zero outside the RIAF frequency grid."""
    t = np.loadtxt(RUNS / run / "spectrum" / "lumThermal.dat")
    nuR, fR = t[:, 0], t[:, 8] / (4 * np.pi * DIST**2) / t[:, 0]
    pos = fR > 0
    return np.exp(np.interp(np.log(nu), np.log(nuR[pos]), np.log(fR[pos]), left=-np.inf, right=-np.inf))


def _read_flux(path):
    """(nu, columns, apertures_rg) from a run.x Flux file; the 2nd header line lists the apertures."""
    with open(path) as f:
        f.readline()
        aps = [float(x) for x in f.readline().split(":")[1].split()]
    d = np.loadtxt(path, comments="#").T
    return d, aps


def load_jet_model(run, riaf_run=None):
    """Jet + counter-jet of analysis/runs/<run>; add the RIAF of analysis/runs/<riaf_run> if given."""
    j, aps = _read_flux(RUNS / run / "Flux_e.dat")
    cj, aps_cj = _read_flux(RUNS / run / "FluxCJ_e.dat")
    assert np.array_equal(j[0], cj[0]) and aps == aps_cj
    n = len(aps)
    assert j.shape[0] == n + 2, f"{run}: expected {n} aperture columns + total, got {j.shape[0] - 2}"
    riaf = load_riaf(j[0], riaf_run) if riaf_run else None
    return Model(run, j[0], aps, j[1:n + 1] + cj[1:n + 1], j[n + 1] + cj[n + 1], riaf)
