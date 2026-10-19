"""Synthetic maps, Jet+RIAF model: Map_I_43GHz.pdf (I at 43 GHz) and Map_spix.pdf (spectral index 22-43, 43-86 GHz).

M87 geometry (i = 17 deg, d = 16.8 Mpc).
Jet and counter-jet: I_nu = delta^3 (j'/alpha') [1 - exp(-alpha' l / (delta sin i))], l = chord through the jet
cylinder of radius R(z) at transverse offset x, z = |y| / sin i (as dFlux_dz in synrad.f90). j' and alpha' are
evaluated per segment at nu' = nu/delta with the code's jsync / alpha_sync (f2py) from the run's Ne_cuts.dat and
quant_FF*.dat, then interpolated in log z. RIAF: I_nu(r) from thermal-riaf spectrum/lumSy_radii.dat (22, 43, 86 GHz),
projected as a disc perpendicular to the jet, r = sqrt(x^2 + (y/cos i)^2).
Each map is convolved (FFT) with a Gaussian beam normalised to unit peak, so pixel values are mJy/beam.

The RIAF is included at all three frequencies; each spectral-index pair is convolved to the beam of its lower
frequency (BEAMS).

    python analysis/scripts/fig_maps.py [--outdir DIR] [--run jet_riaf] [--riaf riaf_m87]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m87 import jetcode, provenance, style  # noqa: E402
from m87.constants import DIST, INCL_DEG, R_G  # noqa: E402
from m87.paths import CONFIGS, FIGURES, JETCODE, RIAFCODE, RUNS  # noqa: E402

style.use()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm, Normalize  # noqa: E402
from matplotlib.patches import Ellipse  # noqa: E402

INCL = np.deg2rad(INCL_DEG)
MAS = np.deg2rad(1 / 3.6e6)                 # 1 mas in rad
RG_MAS = R_G / DIST / MAS                    # 1 R_g in mas
FREQS = {22: 22e9, 43: 43e9, 86: 86e9}       # GHz label -> Hz (the RIAF file is tabulated at these)
# beam FWHM (major along the jet, minor across) [mas]; each spix pair uses the beam of its lower frequency
BEAMS = {22: (1.2, 1.2), 43: (0.31, 0.31)}
PIX = 0.004                                  # map pixel [mas] (~1 R_g)
HALF_X = 2.5                                 # map half-width across the jet [mas]
Y_MAX = {22: 8.0, 43: 4.0}                   # map extent along the jet (each side) per beam [mas]
LEVEL0, LEVEL_STEP = 3.0, np.sqrt(2)         # contours: LEVEL0 x (1, 1.4, 2, 2.8, ...) mJy/beam
# I(43 GHz) is drawn on the sky (relative RA/Dec, RA increasing to the left) with the field of EHT MWL Science WG 2021,
# fig. 14 (VLBA 43 GHz, same 0.31 mas beam and contours), so the two can be compared directly
JET_PA_DEG = -72.0                           # position angle of the approaching jet (east of north; Walker+2018)
SKY_LIM = {"I43": ((0.9, -1.7), (-1.0, 1.0)),  # (RA, Dec) limits [mas] per panel; spectral-index maps likewise
           "22-43": ((2.5, -5.5), (-3.0, 3.0)), "43-86": ((0.9, -1.7), (-1.0, 1.0))}


def jet_tables(run, nu):
    """(log z, log j, log alpha, delta, R) per segment for jet and counter-jet at observed nu."""
    out = {}
    for jet in ("", "CJ"):
        seg = jetcode.load_segments(run, jet)
        j, a = jetcode.emission(seg, nu)
        ok = j > 0
        out[jet] = (np.log(seg["z"][ok]), np.log(j[ok]), np.log(a[ok]), seg["delta"][ok], seg["R"][ok])
        assert np.all(seg["R_in"] == 0), "inner radius not supported"
    return out


def jet_intensity(tab, x_mas, y_mas):
    """I_nu [erg s^-1 cm^-2 Hz^-1 sr^-1] on the (y, x) grid; y > 0 approaching jet, y < 0 counter-jet."""
    img = np.zeros((len(y_mas), len(x_mas)))
    x = x_mas * MAS * DIST
    for side, jet in ((1, ""), (-1, "CJ")):
        lz, lj, la, dl, rr = tab[jet]
        rows = np.where(side * y_mas > 0)[0]
        z = np.abs(y_mas[rows]) * MAS * DIST / np.sin(INCL)
        inside = (np.log(z) >= lz[0]) & (np.log(z) <= lz[-1])
        rows, z = rows[inside], z[inside]
        f = lambda t: np.interp(np.log(z), lz, t)  # noqa: E731
        j, a, d, R = np.exp(f(lj)), np.exp(f(la)), f(dl), np.exp(f(np.log(rr)))
        l = 2 * np.sqrt(np.clip(R[:, None]**2 - x[None, :]**2, 0, None))
        tau = a[:, None] * l / (d[:, None] * np.sin(INCL))
        img[rows] = (d**3 * j / a)[:, None] * -np.expm1(-tau)
    return img


def riaf_intensity(riaf, ghz, x_mas, y_mas):
    """RIAF I_nu(r) of lumSy_radii.dat, disc perpendicular to the jet (inclination i); 0 outside the tabulated radii."""
    t = np.loadtxt(RUNS / riaf / "spectrum" / "lumSy_radii.dat")
    col = {22: 1, 43: 2, 86: 3}[ghz]
    r_t, I_t = t[:, 0], t[:, col]
    X, Y = np.meshgrid(x_mas, y_mas)
    r = np.hypot(X, Y / np.cos(INCL)) / RG_MAS
    pos = I_t > 0
    out = np.exp(np.interp(np.log(np.clip(r, r_t[0], None)), np.log(r_t[pos]), np.log(I_t[pos]),
                           left=np.log(I_t[pos][0]), right=-np.inf))
    return np.where(r <= r_t[pos][-1], out, 0.0)


def convolve(img, beam):
    """Intensity map -> mJy/beam, Gaussian beam (FWHM along y, along x) with unit peak."""
    pix_mJy = img * (PIX * MAS)**2 * 1e26
    sy, sx = (b / (2 * np.sqrt(2 * np.log(2))) for b in beam)
    ky = np.arange(-int(4 * sy / PIX), int(4 * sy / PIX) + 1) * PIX
    kx = np.arange(-int(4 * sx / PIX), int(4 * sx / PIX) + 1) * PIX
    k = np.exp(-0.5 * (ky[:, None]**2 / sy**2 + kx[None, :]**2 / sx**2))
    return fftconvolve(pix_mJy, k, mode="same"), pix_mJy.sum()


def make_maps(run, riaf):
    x = np.arange(-HALF_X, HALF_X + PIX / 2, PIX)
    out, info = {}, {"pixel_mas": PIX, "R_g_mas": RG_MAS}
    tabs = {g: jet_tables(run, nu) for g, nu in FREQS.items()}
    for beam_ghz, pair in ((43, (43, 86)), (22, (22, 43))):
        ymax = Y_MAX[beam_ghz]
        y = np.arange(-ymax, ymax + PIX / 2, PIX)
        for g in pair:
            I_jet = jet_intensity(tabs[g], x, y)
            I_riaf = riaf_intensity(riaf, g, x, y)
            m, s_tot = convolve(I_jet + I_riaf, BEAMS[beam_ghz])
            out[(g, beam_ghz)] = (x, y, m)
            info[f"{g}GHz_beam{beam_ghz}"] = {
                "S_total_mJy": float(s_tot), "S_jet_mJy": float(I_jet.sum() * (PIX * MAS)**2 * 1e26),
                "S_riaf_mJy": float(I_riaf.sum() * (PIX * MAS)**2 * 1e26), "peak_mJy_per_beam": float(m.max())}
    return out, info


def flux_check(run, info):
    """Map flux of the jet (no RIAF) within the map vs the code's per-aperture integration, same z range."""
    J = jetcode.module()
    h = float(J.constants.h)
    res = {}
    for g, nu in FREQS.items():
        ymax = Y_MAX[22 if g == 22 else 43]   # extent of the map in which S_jet of this frequency is reported
        tot = 0.0
        for jet in ("", "CJ"):
            seg = jetcode.load_segments(run, jet)
            for k in range(1, len(seg["z"])):
                if seg["z"][k] * np.sin(INCL) / (MAS * DIST) <= ymax:
                    tot += J.synrad.dflux_dz(seg["N"][:, k], seg["B"][k], h * nu / seg["delta"][k], seg["E"],
                                             seg["delta"][k], seg["R"][k], seg["R_in"][k], "e") * seg["dz"][k] * h
        res[f"{g}GHz"] = {"ymax_mas": ymax, "code_mJy": tot * 1e26}
    return res


def levels(m):
    n = int(np.floor(np.log(m.max() / LEVEL0) / np.log(LEVEL_STEP))) + 1
    return LEVEL0 * LEVEL_STEP**np.arange(max(n, 1))


def beam_patch(ax, beam, xy):
    ax.add_patch(Ellipse(xy, beam[0], beam[1], color="grey", zorder=10))


def sky(x, y):
    """(RA, Dec) offsets [mas] of the map grid (y along, x across the jet), jet axis at JET_PA_DEG."""
    pa = np.deg2rad(JET_PA_DEG)
    Y, X = np.meshgrid(y, x, indexing="ij")
    return Y * np.sin(pa) + X * np.cos(pa), Y * np.cos(pa) - X * np.sin(pa)


def _frame(ax, key, beam):
    (ra0, ra1), (dec0, dec1) = SKY_LIM[key]
    beam_patch(ax, beam, (ra0 - 0.6 * beam[0], dec0 + 0.6 * beam[1]))
    ax.set_xlim(ra0, ra1)
    ax.set_ylim(dec0, dec1)
    ax.set_aspect("equal")
    ax.set_xlabel("relative RA [mas]")


def plot_intensity(maps, outpath):
    """I(43 GHz) with the 43 GHz beam, on the sky: the jet axis is rotated to JET_PA_DEG (the beam is circular)."""
    fig, ax = plt.subplots(figsize=(6.5, 5.2))
    x, y, m = maps[(43, 43)]
    lv = levels(m)
    img = np.where(m > lv[0] * 0.1, m, lv[0] * 0.1)
    ra, dec = sky(x, y)
    pc = ax.pcolormesh(ra, dec, img, cmap="afmhot_r", norm=LogNorm(lv[0] * 0.3, m.max()), shading="auto",
                       rasterized=True)
    ax.contour(ra, dec, m, levels=lv, colors="k", linewidths=0.4)
    fig.colorbar(pc, ax=ax, label=r"$S_{43}$ [mJy beam$^{-1}$]", shrink=0.8, pad=0.02)
    _frame(ax, "I43", BEAMS[43])
    ax.set_ylabel("relative Dec [mas]")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def plot_spix(maps, outpath):
    """Spectral index 22-43 and 43-86 GHz, each pair at the beam of its lower frequency; contours of that frequency."""
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"wspace": 0.3})
    for ax, (lo, hi), key in ((axs[0], (22, 43), "22-43"), (axs[1], (43, 86), "43-86")):
        _, _, mlo = maps[(lo, lo)]
        x, y, mhi = maps[(hi, lo)]
        lv = levels(mlo)
        with np.errstate(divide="ignore", invalid="ignore"):
            spix = np.log(mhi / mlo) / np.log(hi / lo)
        spix = np.where((mlo > lv[0]) & (mhi > lv[0]), spix, np.nan)
        ra, dec = sky(x, y)
        pc = ax.pcolormesh(ra, dec, spix, cmap="jet", norm=Normalize(-1.5, 0.5), shading="auto", rasterized=True)
        ax.contour(ra, dec, mlo, levels=lv, colors="w", linewidths=0.6, alpha=0.8)
        fig.colorbar(pc, ax=ax, label=r"spectral index $\alpha$", shrink=0.8, pad=0.02)
        ax.set_title(rf"{lo}–{hi} GHz", fontsize=14)
        _frame(ax, key, BEAMS[lo])
    axs[0].set_ylabel("relative Dec [mas]")
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=FIGURES)
    ap.add_argument("--run", default="jet_riaf")
    ap.add_argument("--riaf", default="riaf_m87")
    a = ap.parse_args()
    jetcode.setup(RUNS / a.run / f"{a.run}.nml")
    maps, info = make_maps(a.run, a.riaf)
    info["flux_check"] = flux_check(a.run, info)
    caption = {"beams_FWHM_mas": {f"{k} GHz": v for k, v in BEAMS.items()},
               "contours": {f"{g} GHz (beam {b})": [float(v) for v in levels(maps[(g, b)][2])]
                            for g, b in ((43, 43), (22, 22))},
               "contour_rule": f"{LEVEL0} mJy/beam x {LEVEL_STEP:.3f}^n", "spix_mask": "both maps > lowest contour"}
    out = [Path(a.outdir) / "Map_I_43GHz.pdf", Path(a.outdir) / "Map_spix.pdf"]
    plot_intensity(maps, out[0])
    plot_spix(maps, out[1])
    summary = {"caption": caption, "maps": info}
    print(json.dumps(summary, indent=1))
    provenance.write(RUNS / "maps.provenance.json" if a.outdir == FIGURES else Path(a.outdir) / "maps.provenance.json",
                     code={"extended-jet": provenance.git_state(JETCODE), "thermal-riaf": provenance.git_state(RIAFCODE)},
                     figures=[o.name for o in out], runs=[a.run, a.riaf], config=str(CONFIGS / f"{a.run}.nml"), summary=summary)


if __name__ == "__main__":
    main()
