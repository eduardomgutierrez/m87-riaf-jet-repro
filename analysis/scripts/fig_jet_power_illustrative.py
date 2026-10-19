"""Jet power figure (paper version): Jet_power.pdf.

Illustrative parameters of the caption (sigma'_0 = 10, g = 0.4, z_acc = 220 R_g, z_0 = 10 R_g, eps_e = 0.3, sharp cut at
z_max,e = 1e4 R_g). a_e = 0.2 is used in the enthalpy h' (so sigma' and L_B do not depend on the plotted a_e), and a_e = 0.01 is used
for the curve labelled a_e = 0 (a_e = 0 is 0/0 in L_e(<z)).
All lengths in R_g (only ratios enter).

    python analysis/scripts/fig_jet_power_illustrative.py [--outdir DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m87 import provenance, style  # noqa: E402
from m87.paths import FIGURES, RUNS  # noqa: E402

style.use()
import matplotlib.pyplot as plt  # noqa: E402

# illustrative parameters (lengths in R_g, powers in units of L_j)
Z0, GAMMA0, G, Z_ACC, ZMAX_INJ, EPS_E, SIGMA0 = 10.0, 1.001, 0.4, 220.0, 1e4, 0.3, 10.0
MU = GAMMA0 * (1.0 + SIGMA0)
MDOT = 1.0 / GAMMA0 / (SIGMA0 + 1.0)        # Mdot c^2 / L_j
HP_ZMAX = 1.0 / (1.0 - EPS_E)
A_E = [(-0.5, "-0.5"), (0.01, "0"), (0.5, "0.5")]  # (value used, label)
LS = [":", "-.", "-"]
LW = 2.0


def gamma(z):
    res = (1.0 + z / Z_ACC) ** G
    return np.where(res < MU / HP_ZMAX, res, MU / HP_ZMAX)


def l_p(z):
    return gamma(z) * MDOT


def l_e(z, a=0.2):
    res = EPS_E * (1 - (Z0 / z) ** a) / (1.0 - (Z0 / ZMAX_INJ) ** a)
    return np.where(z < ZMAX_INJ, res, EPS_E)


def hp(z):
    return 1.0 + l_e(z) / l_p(z)


def sigma(z):
    return MU / gamma(z) / hp(z) - 1.0


def l_b(z):
    res = MU - gamma(z) * hp(z)
    return MDOT * np.where(res > 0.0, res, 0.0)


def make(outdir):
    z = np.geomspace(Z0, 5e4, 400)
    fig, (ax_top, ax) = plt.subplots(2, 1, figsize=(4.2, 6.0), sharex=True,
                                     gridspec_kw={"height_ratios": [1, 2], "hspace": 0.03})
    ax_top.loglog(z, sigma(z), label=r"$\sigma'$", c="tab:green", lw=LW)
    ax_top.loglog(z, gamma(z), label=r"$\Gamma_{\rm j}$", c="tab:brown", lw=LW)
    ax_top.set_ylabel(r"$\sigma',\;\Gamma_{\rm j}$")
    ax_top.legend(fontsize=10, frameon=True, fancybox=True, framealpha=0.8, edgecolor="grey", loc="lower left")
    ax_top.set_ylim(1e-1, 11)

    ax.loglog(z, l_b(z), label=r"$L_B$", c="tab:red", lw=LW)
    ax.loglog(z, l_p(z), label=r"$L_{\rm p}$", c="tab:blue", lw=LW)
    ax.loglog(z, l_b(z) + l_e(z) + l_p(z), ls="--", c="k", lw=LW)
    for (a, lab), ls in zip(A_E, LS):
        ax.loglog(z, l_e(z, a), ls=ls, c="orange", lw=LW, label=rf"$L_{{\rm e}}(<z),\;a_{{\rm e}}={lab}$")
    ax.axhline(EPS_E, ls="--", c="grey", lw=LW)
    ax.text(0.05, 0.80, r"$\epsilon_{\rm e}=0.3$", transform=ax.transAxes, fontsize=12, va="top", ha="left")
    ax.legend(fontsize=9.5, frameon=True, fancybox=True, framealpha=0.8, edgecolor="grey",
              loc="lower right", bbox_to_anchor=(0.88, 0.0))
    ax.set_xlabel(r"$z~[R_{\rm g}]$")
    ax.set_ylabel(r"$L/L_{\rm j}$")
    ax.set_ylim(1e-3, 2)
    ax.set_xlim(z[0] * 0.8, z[-1] * 1.25)
    for a in (ax_top, ax):
        a.tick_params(labelsize=12)
        a.yaxis.label.set_size(14)
    ax.xaxis.label.set_size(14)

    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / "Jet_power.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)  # no white margin (as tight_layout(pad=0))
    plt.close(fig)
    provenance.write(out.with_suffix(".provenance.json") if outdir != FIGURES else RUNS / "jet_power.provenance.json",
                     figure=out.name, source="illustrative parameters",
                     parameters=dict(z0=Z0, Gamma0=GAMMA0, g=G, z_acc=Z_ACC, zmax_e=ZMAX_INJ, eps_e=EPS_E,
                                     sigma_0=SIGMA0, a_e=[a for a, _ in A_E]))
    print(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=FIGURES)
    make(ap.parse_args().outdir)


if __name__ == "__main__":
    main()
