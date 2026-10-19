"""SED figure SED_FluxDensity.pdf: Jet-only model (top row) and Jet+RIAF model (bottom row).

In each row, left: broadband nuFnu with residuals; right: radio
flux density with residuals (aperture labels in the top row only). Each model curve is the jet (+ counter-jet) integrated up to the height matching one aperture
R*; each data point is coloured by its own aperture (data.aperture_of: per instrument, or per frequency for KVN, VERA
and VLBA); the other instruments (SMA, optical/UV, X-rays) are compared with the whole jet and take the colour of the
3.5e5 R_g curve, which for the paper runs contains the whole jet (z_max sin i = 3.51e5 R_g). In the radio panels,
points at nearly the same frequency are spread horizontally (dodged_ghz, display only) so that they do not hide
each other; residuals are (obs - model)/model. The 131 R_g curve takes a colour between the 65 and
195 R_g ones, so the six original curves keep their colours.
The ALMA points are compared with the 3.5e5 R_g aperture (the whole jet).
Gamma-ray data are not shown (the model is synchrotron only; they are discussed in the text).

    python analysis/scripts/fig_sed.py [--outdir DIR]
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m87 import style  # noqa: E402
from m87.constants import Jy, eV, h  # noqa: E402
from m87.data import APERTURES_RG, GAMMA, load_obs  # noqa: E402
from m87.models import load_jet_model  # noqa: E402
from m87.paths import FIGURES  # noqa: E402

style.use()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

OUT = "SED_FluxDensity.pdf"
MODELS = [("Jet-only", "jet_only", None), ("Jet+RIAF", "jet_riaf", "riaf_m87")]  # rows: (label, jet run, RIAF run)
XLIM, YLIM = (5e8, 3e20), (5e-15, 1e-11)
XLIM_GHZ, YLIM_JY, RES_LIM, RES_TICKS = (1, 1e3), (1e-2, 1e1), (-1.3, 3), [-1, 0, 1, 2]
DODGE_DEX, DODGE_GAP = 0.05, 0.06  # radio panels: points closer than DODGE_GAP dex are spread DODGE_DEX dex apart
MARKERS = {"EHT": "o", "GMVA": "s", "EAVN": "^", "HSA": "<", "EVN": "v", "ALMA": "X",  # unique among radio arrays
           "KVN": "D", "VERA": "p", "VLBA": "h", "SMA": "*",
           "HST": "o", "Swift": "s", "Chandra": "^", "NuSTAR": "v"}
# aperture label boxes in the right panel (GHz, Jy), as in the draft figure
LABEL_POS = {7.3: (1.2e2, 0.5e-1), 26.0: (0.55e2, 0.9e-1), 65.0: (0.37e2, 2.3), 131.0: (1.95e1, 2.3),
             195.0: (1.0e1, 2), 650.0: (0.17e1, 1.5), 3.5e5: (3e2, 1)}
NEW_APS = [131.0]  # added in 04b; drawn between their neighbours' colours


def aperture_colors(aps):
    """Draft palette for the original six apertures; each added aperture gets the mid colour of its neighbours."""
    old = [r for r in aps if r not in NEW_APS]
    cold = dict(zip(old, style.aperture_colors(len(old))))
    x = dict(zip(old, np.linspace(0.15, 0.85, len(old))))
    out = []
    for k, r in enumerate(aps):
        out.append(cold[r] if r in cold else plt.cm.CMRmap_r(0.5 * (x[aps[k - 1]] + x[aps[k + 1]])))
    return out


def data_style(name, colors, r=None):
    """Point style: colour of aperture r (None = whole jet: colour of the largest aperture, which contains it)."""
    c = colors[APERTURES_RG.index(r) if r is not None else -1]
    return dict(ls="", marker=MARKERS[name], color=c, markeredgecolor="k", ecolor="k", zorder=100)


def dodged_ghz(obs):
    """{instrument: display frequency [GHz] per point} for the radio panels: points within DODGE_GAP dex of each other
    form a cluster, spread symmetrically DODGE_DEX dex apart in frequency order (display only)."""
    pts = sorted((np.log10(o.nu[i] / 1e9), n, i) for n, o in obs.items() if o.tagged for i in range(len(o.nu)))
    out = {n: o.nu / 1e9 for n, o in obs.items()}
    k = 0
    while k < len(pts):
        j = k + 1
        while j < len(pts) and pts[j][0] - pts[j - 1][0] < DODGE_GAP:
            j += 1
        c = np.mean([p[0] for p in pts[k:j]])
        for m, (_, n, i) in enumerate(pts[k:j]):
            out[n] = out[n].copy()
            out[n][i] = 10**(c + (m - (j - k - 1) / 2) * DODGE_DEX)
        k = j
    return out


def plot_points(ax, x, y, dy, o, colors, big=False):
    """errorbar per point (each point may have its own aperture colour)."""
    kw = dict(markersize=8.5, elinewidth=2, capsize=4, capthick=2) if big else \
        dict(markersize=9, elinewidth=2.5, capsize=4, capthick=2.5)
    for i, r in enumerate(o.apertures_rg):
        ax.errorbar(x[i:i + 1], y[i:i + 1], yerr=dy[i:i + 1], uplims=o.uplim[i:i + 1], **kw,
                    **data_style(o.name, colors, r))


def errorbars(o, scale=1.0):
    """y errors; upper limits get a 0.5 x value arrow as in the draft figure."""
    return np.where(o.uplim, 0.5 * o.nufnu, o.err) * scale


def plot_row(fig, gs, title, model, obs, top_row):
    colors = aperture_colors(model.apertures_rg)
    labels = [style.aperture_label(r) for r in model.apertures_rg]
    has_riaf = model.riaf is not None
    nu, ghz = model.nu, model.nu / 1e9

    gs = gs.subgridspec(2, 2, height_ratios=[3, 1], hspace=0.05, wspace=0.2)
    ax1, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax1r, ax2r = fig.add_subplot(gs[1, 0], sharex=ax1), fig.add_subplot(gs[1, 1], sharex=ax2)
    ax1.text(0.03, 0.95, title, transform=ax1.transAxes, fontsize=20, va="top")

    # --- left: broadband nuFnu
    for k in range(len(model.apertures_rg)):
        jet_nonzero = np.any(model.jet[k] > 0)
        if has_riaf:
            if not jet_nonzero:  # total = RIAF alone (7.3 R_g for Jet+RIAF): drawn as the RIAF curve
                continue
            ax1.plot(nu, nu * model.jet[k], lw=1, ls="--", color=colors[k])
        ax1.plot(nu, nu * model.total[k], lw=2, color=colors[k], label=labels[k])
    if has_riaf:
        ax1.plot(nu, nu * model.riaf, lw=1.5, ls="-.", color="k", label="RIAF")

    for name, o in obs.items():
        plot_points(ax1, o.nu, o.nufnu, errorbars(o), o, colors)
        m = model.nufnu_obs(o)
        plot_points(ax1r, o.nu, (o.nufnu - m) / m, errorbars(o) / m, o, colors)

    ax1.set(xscale="log", yscale="log", xlim=XLIM, ylim=YLIM, ylabel=r"${\rm Flux~[erg~cm^{-2}~s^{-1}]}$")
    ax1.tick_params(axis="x", labelbottom=False, top=False, which="both")
    top = ax1.secondary_xaxis("top", functions=(lambda x: x * h / eV, lambda e: e * eV / h))
    top.set_xlabel(r"${\rm Energy~[eV]}$" if top_row else "", labelpad=10)
    top.tick_params(which="both", direction="in", labeltop=top_row)
    ax1r.set(xscale="log", xlim=XLIM, ylim=RES_LIM, xlabel=r"${\rm Frequency~[Hz]}$", ylabel=r"$\Delta F/F$")

    # --- right: radio flux density
    for k in range(len(model.apertures_rg)):
        if has_riaf:
            ax2.plot(ghz, model.jet[k] / Jy, lw=2, ls="--", color=colors[k])
        ax2.plot(ghz, model.total[k] / Jy, lw=4, color=colors[k])
    if has_riaf:
        ax2.plot(ghz, model.riaf / Jy, lw=1.5, ls="-.", color="k")
    xd = dodged_ghz(obs)
    for name, o in obs.items():
        if not o.tagged:
            continue
        s, ds = o.nufnu / o.nu / Jy, errorbars(o, 1 / o.nu / Jy)
        x = xd[name]
        plot_points(ax2, x, s, ds, o, colors, big=True)
        m = model.nufnu_obs(o) / o.nu / Jy
        plot_points(ax2r, x, (s - m) / m, ds / m, o, colors, big=True)
    if top_row:
        for r, lab, c in zip(model.apertures_rg, labels, colors):
            ax2.text(*LABEL_POS[r], lab, backgroundcolor=c, fontsize=16, rotation=60, color="white")
    ax2.set(xscale="log", yscale="log", xlim=XLIM_GHZ, ylim=YLIM_JY, ylabel=r"$S_\nu~[{\rm Jy}]$")
    ax2.tick_params(axis="x", labelbottom=False)
    ax2r.set(xscale="log", xlim=XLIM_GHZ, ylim=RES_LIM, xlabel=r"${\rm Frequency~[GHz]}$", ylabel=r"$\Delta S/S$")
    for ax in (ax1r, ax2r):
        ax.axhline(0, color="k", ls="--", lw=1)
        ax.set_yticks(RES_TICKS)
        ax.set_yticks(np.arange(-1, 3.01, 0.5), minor=True)
    if not top_row:
        return

    # marker key for all instruments, each in the colour(s) its points have in the figure: one marker per aperture
    # (e.g. EAVN: 131 and 65 R_g side by side); instruments compared with the whole jet keep their own colour
    order = ["EHT", "GMVA", "EAVN", "HSA", "EVN", "KVN", "VERA", "VLBA", "ALMA", "SMA", "HST", "Swift", "Chandra", "NuSTAR"]
    handles, names = [], []
    for n in (n for n in order if n in obs):
        aps = list(dict.fromkeys(obs[n].apertures_rg))
        marks = tuple(Line2D([], [], ls="", marker=MARKERS[n], markersize=8, markeredgecolor="k",
                             color=data_style(n, colors, r)["color"]) for r in aps)
        handles.append(marks if len(marks) > 1 else marks[0])
        names.append(n)
    ax1.legend(handles, names, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.1)}, loc="lower center",
               bbox_to_anchor=(0.58, 0.0), ncol=5, fontsize=10, handletextpad=0.2, columnspacing=0.6, labelspacing=0.3)


def plot(rows, obs, out):
    """rows: [(label, Model)], one row of panels each, top to bottom."""
    fig = plt.figure(figsize=(14, 12.4))
    gs = fig.add_gridspec(len(rows), 1, hspace=0.22)
    for k, (title, model) in enumerate(rows):
        plot_row(fig, gs[k], title, model, obs, top_row=(k == 0))
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=FIGURES)
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)
    obs = {k: v for k, v in load_obs().items() if k not in GAMMA}
    rows = []
    for title, run, riaf in MODELS:
        model = load_jet_model(run, riaf)
        assert model.apertures_rg == APERTURES_RG, (run, model.apertures_rg)
        rows.append((title, model))
    plot(rows, obs, args.outdir / OUT)


if __name__ == "__main__":
    main()
