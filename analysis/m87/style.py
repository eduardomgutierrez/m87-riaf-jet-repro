"""Plot style close to smplotlib (used for the draft figures): serif fonts, inward ticks on all sides, minor ticks."""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

RC = {
    "font.family": "serif", "font.serif": ["STIX Two Text", "STIXGeneral", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 16,
    "axes.linewidth": 1.2, "axes.labelsize": 18,
    "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.size": 8, "ytick.major.size": 8, "xtick.minor.size": 4, "ytick.minor.size": 4,
    "xtick.major.width": 1.2, "ytick.major.width": 1.2, "xtick.minor.width": 0.9, "ytick.minor.width": 0.9,
    "xtick.labelsize": 16, "ytick.labelsize": 16,
    "legend.frameon": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
}


def use():
    mpl.use("Agg")
    plt.rcParams.update(RC)


def aperture_colors(n=6):
    """Aperture palette of the draft figures (CMRmap_r between 0.15 and 0.85), smallest aperture first."""
    return [plt.cm.CMRmap_r(x) for x in np.linspace(0.15, 0.85, n)]


def aperture_label(r_rg, sym="r"):
    if r_rg >= 1e4:
        e = int(np.floor(np.log10(r_rg)))
        m = r_rg / 10**e
        return rf"${sym}<{m:g}\times10^{{{e}}}$"
    return rf"${sym}<{r_rg:g}$"
