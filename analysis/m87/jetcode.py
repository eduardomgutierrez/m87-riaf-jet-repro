"""The extended-jet f2py module (JetCode, built with `make f2py` in codes/extended-jet/Code).

The dynamics functions (Gamma, L_e(<z), L_p, sigma, B, ...) are evaluated with the code's own routines, so the
figures use exactly the physics of run.x. Parameters are set from a namelist (read_namelist), optionally overridden
through the *_rg / *_eV mirrors, followed by derive_parameters().
"""
import sys

import numpy as np

from .paths import JETCODE

_J = None
# namelist keys that are lengths in R_g or energies in eV: set the mirror, derive_parameters() converts them
_MIRRORS = {"z0", "zmax", "zmax_inj", "r_br", "z_br", "z_acc", "rin_0"}


def module():
    global _J
    if _J is None:
        sys.path.insert(0, str(JETCODE))
        import JetCode
        _J = JetCode
    return _J


def setup(nml, fix_norm_taper=None, fix_inj_emin=None, **overrides):
    """Load a namelist into the module, apply overrides (lengths in R_g, e.g. z0=10), derive the cgs values.

    Returns the system_parameters module. fix_* override the &run control flags (None = as in the namelist).
    """
    J = module()
    sp = J.system_parameters
    sp.read_namelist(str(nml))
    for k, v in overrides.items():
        k = k.lower()
        setattr(sp, f"{k}_rg" if k in _MIRRORS else k, v)
    if fix_norm_taper is not None:
        J.controls.fix_norm_taper = fix_norm_taper
    if fix_inj_emin is not None:
        J.controls.fix_inj_emin = fix_inj_emin
    sp.derive_parameters()
    return sp


def dynamics(sp, z_rg):
    """Jet quantities along z [R_g] with the code's functions (all in cgs; powers in erg/s)."""
    z = np.asarray(z_rg, float) * sp.r_g
    f = lambda fun: np.array([fun(x) for x in z])  # noqa: E731
    return {
        "z_rg": np.asarray(z_rg, float), "Gamma": f(sp.lorentzfactor), "beta": f(sp.betajet),
        "Le": f(sp.le_z), "Lp": f(sp.lp), "LB": f(sp.l_b), "sigma": f(sp.sigma_b), "B": f(sp.bff_fun),
        "r": f(sp.r_fun_out), "Lj": float(sp.lj), "Linj": float(sp.lnt),
    }


def load_segments(run, jet=""):
    """Per-segment quantities of analysis/runs/<run> (quant_FF<jet>.dat) and the electron spectra (Ne_cuts.dat).

    Returns dict: z [cm, >0], B [G], R (outer) and R_in [cm], Gamma, delta, dz [cm], sigma, E [erg], N (nIn x Ns).
    The comoving N(E) is the same for jet and counter-jet (as in synrad.SEDsyn).
    """
    from .paths import RUNS
    q = np.loadtxt(RUNS / run / f"quant_FF{jet}.dat")
    ne = np.loadtxt(RUNS / run / "Ne_cuts.dat")
    ns = q.shape[0]
    E = ne[::ns, 0]
    N = ne[:, 1].reshape(len(E), ns)
    assert np.allclose(ne[:ns, 2], np.arange(1, ns + 1)), "Ne_cuts.dat is not ordered (E outer, segment inner)"
    return {"z": np.abs(q[:, 0]), "B": q[:, 1], "R": q[:, 2], "R_in": q[:, 3], "Gamma": q[:, 4], "delta": q[:, 5],
            "dz": q[:, 10], "sigma": q[:, 12], "Le": q[:, 13], "E": E, "N": N}


def emission(seg, nu):
    """Comoving j'_nu' [erg s^-1 cm^-3 Hz^-1 sr^-1] and alpha'_nu' [cm^-1] per segment at the OBSERVED frequency nu,
    i.e. at nu' = nu/delta, with the code's jsync / alpha_sync (sync_approx as loaded by setup()).
    Segments with N = 0 get j = alpha = 0.
    """
    J = module()
    h = float(J.constants.h)
    j = np.zeros(len(seg["z"]))
    a = np.zeros(len(seg["z"]))
    for k in range(len(seg["z"])):
        n = seg["N"][:, k]
        if not np.any(n > 0):
            continue
        eph = h * nu / seg["delta"][k]
        j[k] = J.synrad.jsync(eph, seg["B"][k], n, seg["E"], "e") * h
        a[k] = J.synrad.alpha_sync(eph, seg["B"][k], n, seg["E"], "e")
    return j, a
