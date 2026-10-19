"""chi^2 of a model against the aperture-tagged SED points.

sigma^2 = err^2 + (floor * data)^2 (systematic floor, default 10 per cent of the measured nuFnu). Each point is
compared with the model integrated up to its own aperture (jet + CJ + RIAF). Upper limits are not in the
chi^2; they are returned separately so they can be checked. Gamma rays are not modelled (synchrotron only).
"""
from dataclasses import dataclass

import numpy as np

from .data import GAMMA


@dataclass
class Point:
    instrument: str
    band: str
    nu: float
    data: float
    err: float
    model: float
    uplim: bool
    chi2: float  # nan for upper limits
    aperture: float = None  # R* [R_g]; None = whole jet


def evaluate(model, obs, floor=0.1):
    pts = []
    for name, o in obs.items():
        m, aps = model.nufnu_obs(o), o.apertures_rg
        for i in range(len(o.nu)):
            sig2 = o.err[i]**2 + (floor * o.nufnu[i])**2
            chi = np.nan if o.uplim[i] or name in GAMMA else (o.nufnu[i] - m[i])**2 / sig2
            pts.append(Point(name, o.band(i), o.nu[i], o.nufnu[i], o.err[i], m[i], bool(o.uplim[i]), chi, aps[i]))
    return pts


def summary(pts, groups):
    """{group label: (N, chi2)} with groups = {label: predicate(Point)}; upper limits and gamma rays excluded."""
    out = {}
    for label, sel in groups.items():
        c = [p.chi2 for p in pts if sel(p) and np.isfinite(p.chi2)]
        out[label] = (len(c), float(np.sum(c)))
    return out
