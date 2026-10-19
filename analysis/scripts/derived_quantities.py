"""Derived jet quantities for the paper: Jet_profiles.pdf and analysis/runs/derived/derived.json.

For the Jet+RIAF and Jet-only runs (run.x outputs in analysis/runs/<run>):
  - B'(z) (comoving), with Ro et al. (2023) overplotted: B = (0.3-1.0 G) (r/2 mas)^-0.73, r = projected distance,
    over ~900-4500 R_s deprojected (i = 17 deg: 2 mas projected = 895 R_s);
  - sigma(z) and Gamma(z);
  - the tau = 1 height per frequency (on-axis chord, jet side): tau = alpha' 2R / (delta sin i), with the code's
    alpha_sync at nu' = nu/delta (as dFlux_dz in synrad.f90) -> core position and core shift;
  - the 230 GHz jet + counter-jet flux (no RIAF) with projected distance <= y for a few y, including 0.3 mas,
    to compare with the Saurabh et al. (2026) jet-base limit (<~ 60 mJy at ~(320, 60) micro-arcsec).

    python analysis/scripts/derived_quantities.py [--outdir DIR]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m87 import jetcode, provenance, style  # noqa: E402
from m87.constants import DIST, INCL_DEG, R_G  # noqa: E402
from m87.paths import FIGURES, JETCODE, RUNS  # noqa: E402

style.use()
import matplotlib.pyplot as plt  # noqa: E402

RUNS_ = {"jet_riaf": ("Jet+RIAF", "tab:red"), "jet_only": ("Jet-only", "tab:blue")}
SIN_I = np.sin(np.deg2rad(INCL_DEG))
MAS = np.deg2rad(1 / 3.6e6)
RG_MAS = R_G / DIST / MAS                     # 1 R_g in mas
NU_CORE = np.array([2.3, 5, 8.4, 15, 22, 43, 86, 230]) * 1e9
Y_FLUX_MAS = [0.05, 0.1, 0.2, 0.3, 0.5, 1.0]
RO23 = dict(B2mas=(0.3, 1.0), slope=-0.73, rs_range=(900, 4500))   # Ro et al. 2023, A&A 673, A159
# Saurabh et al. (2026) jet-base Gaussian: centre ~0.32 mas from the ring, FWHM ~0.17 mas -> centre +- FWHM/2
SAURABH26_BIN = [0.235, 0.405]


def z_proj_mas(z_cm):
    return z_cm * SIN_I / R_G * RG_MAS


def tau1_height(seg, nu):
    """(z [cm] beyond which the on-axis optical depth drops below 1 (jet side) or None if thin everywhere,
    tau at the first emitting segment)."""
    _, a = jetcode.emission(seg, nu)
    tau = a * 2 * seg["R"] / (seg["delta"] * SIN_I)
    tau_base = float(tau[a > 0][0])
    thick = np.where(tau >= 1)[0]
    if len(thick) == 0:
        return None, tau_base
    k = thick[-1]
    if k == len(tau) - 1:
        return float(seg["z"][k]), tau_base
    # log-interpolate tau = 1 between segments k (thick) and k+1 (thin)
    lt = np.log(tau[k:k + 2])
    lz = np.log(seg["z"][k:k + 2])
    return float(np.exp(lz[0] + (0 - lt[0]) * (lz[1] - lz[0]) / (lt[1] - lt[0]))), tau_base


def flux_within(run, nu, y_mas):
    """Jet + counter-jet F_nu [mJy] from segments with projected distance <= y_mas (code's dFlux_dz)."""
    J = jetcode.module()
    h = float(J.constants.h)
    out = np.zeros(len(y_mas))
    for jet in ("", "CJ"):
        seg = jetcode.load_segments(run, jet)
        for k in range(1, len(seg["z"])):
            dF = J.synrad.dflux_dz(seg["N"][:, k], seg["B"][k], h * nu / seg["delta"][k], seg["E"], seg["delta"][k],
                                   seg["R"][k], seg["R_in"][k], "e") * seg["dz"][k] * h * 1e26
            out += dF * (z_proj_mas(seg["z"][k]) <= np.asarray(y_mas))
    return out


def analyse(run):
    jetcode.setup(RUNS / run / f"{run}.nml")
    seg = jetcode.load_segments(run)
    z_rg = seg["z"] / R_G
    res = {"z_rg": z_rg, "B": seg["B"], "sigma": seg["sigma"], "Gamma": seg["Gamma"]}
    zc, tb = zip(*[tau1_height(seg, nu) for nu in NU_CORE])
    res["core"] = {f"{nu / 1e9:g} GHz": {"tau_base": t, **({"thin_everywhere": True} if z is None else
                   {"z_rg": z / R_G, "proj_mas": z_proj_mas(z)})} for nu, z, t in zip(NU_CORE, zc, tb)}
    ok = [(nu, z) for nu, z in zip(NU_CORE, zc) if z is not None and z > seg["z"][1]]
    if len(ok) >= 3:
        lnu, lz = np.log([o[0] for o in ok]), np.log([o[1] for o in ok])
        res["core_shift_index_k"] = float(-np.polyfit(lnu, lz, 1)[0])   # z_core ∝ nu^-k (not nu^-1/k)
    f = flux_within(run, 230e9, Y_FLUX_MAS + [0.25, 0.35] + SAURABH26_BIN)
    res["F230_within_mJy"] = dict(zip([f"{y:g} mas" for y in Y_FLUX_MAS], map(float, f[:-4])))
    res["F230_0.25-0.35mas_mJy"] = float(f[-3] - f[-4])
    res["F230_0.235-0.405mas_mJy"] = float(f[-1] - f[-2])   # quoted in the paper
    ro_z = np.array(RO23["rs_range"]) * 2          # R_g, deprojected
    res["B_at_Ro23"] = {f"{z:g} R_g ({z_proj_mas(z * R_G):.2f} mas)": float(np.exp(np.interp(np.log(z), np.log(z_rg),
                        np.log(seg["B"])))) for z in (ro_z[0], ro_z[1])}
    res["sigma_at"] = {f"{z:g} R_g": float(np.interp(np.log(z), np.log(z_rg), seg["sigma"]))
                       for z in (z_rg[0], 1e3, 1e4, 1e5, z_rg[-1])}
    res["Gamma_max"] = float(seg["Gamma"].max())
    return res


def plot(results, outpath):
    fig, axs = plt.subplots(2, 2, figsize=(11, 8.5), gridspec_kw={"hspace": 0.28, "wspace": 0.28})
    ax = axs[0, 0]
    for run, r in results.items():
        name, c = RUNS_[run]
        ax.loglog(r["z_rg"], r["B"], c=c, lw=2, label=name)
    z = np.geomspace(*RO23["rs_range"], 50) * 2          # R_g
    rmas = z_proj_mas(z * R_G)
    lo, hi = (b * (rmas / 2.0)**RO23["slope"] for b in RO23["B2mas"])
    ax.fill_between(z, lo, hi, color="grey", alpha=0.4, label="Ro et al. (2023)")
    ax.set_xlabel(r"$z~[R_{\rm g}]$")
    ax.set_ylabel(r"$B'~[{\rm G}]$")
    ax.legend(fontsize=11)
    sec = ax.secondary_xaxis("top", functions=(lambda x: x * SIN_I * RG_MAS, lambda m: m / (SIN_I * RG_MAS)))
    sec.set_xlabel("projected distance [mas]", fontsize=13)
    ax = axs[0, 1]
    for run, r in results.items():
        name, c = RUNS_[run]
        ax.loglog(r["z_rg"], r["sigma"], c=c, lw=2, label=rf"$\sigma'$, {name}")
        ax.loglog(r["z_rg"], r["Gamma"], c=c, lw=2, ls="--", label=rf"$\Gamma_{{\rm j}}$, {name}")
    ax.set_xlabel(r"$z~[R_{\rm g}]$")
    ax.set_ylabel(r"$\sigma',\;\Gamma_{\rm j}$")
    ax.legend(fontsize=10)
    ax = axs[1, 0]
    for run, r in results.items():
        name, c = RUNS_[run]
        pts = [(float(k.split()[0]), v["proj_mas"]) for k, v in r["core"].items() if "proj_mas" in v]
        if pts:
            nu, m = np.array(pts).T
            lab = name + (rf" ($k={r['core_shift_index_k']:.2f}$)" if "core_shift_index_k" in r else "")
            ax.loglog(nu, m, "o-", c=c, lw=2, label=lab)
    ax.set_xlabel(r"$\nu$ [GHz]")
    ax.set_ylabel(r"$\tau_\nu=1$ position [mas, projected]")
    notes = [f"{RUNS_[run][0]}: " + r"$\tau<1$ at $z_0$ for $\nu\geq$" + f"{min(thin):g} GHz"
             for run, r in results.items()
             if (thin := [float(k.split()[0]) for k, v in r["core"].items() if v.get("thin_everywhere")])]
    ax.text(0.03, 0.04, "\n".join(notes), transform=ax.transAxes, fontsize=10)
    ax.legend(fontsize=11)
    ax = axs[1, 1]
    for run, r in results.items():
        name, c = RUNS_[run]
        y = [float(k.split()[0]) for k in r["F230_within_mJy"]]
        ax.loglog(y, list(r["F230_within_mJy"].values()), "o-", c=c, lw=2, label=name)
    ax.axvline(0.3, c="grey", ls=":")
    ax.set_xlabel(r"projected distance $y$ [mas]")
    ax.set_ylabel(r"jet $S_{230}(<y)$ [mJy]")
    ax.legend(fontsize=11)
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=FIGURES)
    a = ap.parse_args()
    results = {run: analyse(run) for run in RUNS_}
    out = RUNS / "derived"
    out.mkdir(exist_ok=True)
    summary = {run: {k: v for k, v in r.items() if k not in ("z_rg", "B", "sigma", "Gamma")} for run, r in results.items()}
    summary["notes"] = {"B": "comoving B' from quant_FF.dat", "core": "on-axis tau = 1, jet side",
                        "F230": "jet + counter-jet, no RIAF, segments with z sin i <= y", "Ro23": RO23}
    with open(out / "derived.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))
    plot(results, a.outdir / "Jet_profiles.pdf")
    provenance.write(out / "provenance.json", code={"extended-jet": provenance.git_state(JETCODE)},
                     runs=list(RUNS_), figure="Jet_profiles.pdf", outdir=str(a.outdir))


if __name__ == "__main__":
    main()
