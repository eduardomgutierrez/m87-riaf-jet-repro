"""Parameter scans around the Jet+RIAF jet model: Scans_SED.pdf (nu L_nu) and Scans_radio.pdf (S_nu, 1-1000 GHz).

Each panel varies one parameter of analysis/configs/jet_riaf.nml,
all others fixed; the base model is the thick black curve. The spectra are jet + counter-jet, whole jet (no aperture
cut), full synchrotron (sync_approx = 'full' from the namelist), computed with the f2py JetCode.syn_model.
syn_model writes intermediate files to the module's outdir, so each worker redirects it to analysis/runs/scans/tmp_*.
Results are cached in analysis/runs/scans/scans.npz (use --recompute to redo them).

    python analysis/scripts/fig_scans.py [--outdir DIR] [--recompute] [--nproc N]
"""
import argparse
import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m87 import jetcode, provenance, style  # noqa: E402
from m87.constants import DIST, h  # noqa: E402
from m87.paths import CONFIGS, FIGURES, JETCODE, RUNS  # noqa: E402

style.use()
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

NML = CONFIGS / "jet_riaf.nml"
OUT = RUNS / "scans"
NU = np.geomspace(1e8, 1e21, 140)
# module parameter -> (values, label, value format); lengths in R_g
SCANS = {
    "a_inj": ([-0.5, 0.01, 0.5, 0.99], r"$a_{\rm e}$", "{:g}"),
    "lj": ([1e42, 3e42, 1.2e43, 3e43], r"$L_{\rm j}$ [$10^{42}$ erg s$^{-1}$]", "e42"),
    "z0": ([2.0, 10.0, 50.0, 200.0], r"$z_0$ [$R_{\rm g}$]", "{:g}"),
    "zmax_inj": ([1e3, 6840.0, 1e5, 1e6], r"$z_{\rm max,e}$ [$R_{\rm g}$]", "sci"),
    "gammaindex": ([0.1, 0.16, 0.3, 0.5], r"$g$", "{:g}"),
    "geomindex_out": ([0.25, 0.5, 0.56, 0.75, 1.0], r"$w$", "{:g}"),
    "sigma_0": ([2.0, 5.0, 30.0, 100.0], r"$\sigma_0$", "{:g}"),
    "facc": ([1e-5, 1e-4, 1e-3, 1e-2], r"$\eta_{\rm acc}$", "sci"),
    "p": ([2.01, 2.05, 2.2, 2.4], r"$p$", "{:g}"),
}
# panels whose curves cross the default legend: tighter legend under the L_j = 1e42 curve, and the SED
# p legend moved to the lower left, clear of the steep p = 2.4 curve
SED_LEGEND = {
    "lj": dict(bbox_to_anchor=(0.58, 0.0), labelspacing=0.1, borderaxespad=0.05),
    "p": dict(loc="lower left", bbox_to_anchor=(0.12, 0.0)),
}
RADIO_LEGEND = {"lj": dict(ncol=2, labelspacing=0.1, borderaxespad=0.05)}


def fmt_value(v, fmt):
    """Legend value: fmt.format(v), A x 10^n (10^n when A = 1) for fmt == 'sci', v / 1e42 for fmt == 'e42'."""
    if fmt == "e42":
        return f"{v / 1e42:g}"
    if fmt != "sci":
        return fmt.format(v)
    e = int(np.floor(np.log10(abs(v))))
    m = round(v / 10**e, 2)
    return rf"$10^{{{e}}}$" if m == 1 else rf"${m:g}\times10^{{{e}}}$"


def base_params():
    sp = jetcode.setup(NML)
    return {"sigma_0": float(sp.sigma_0), "zmax": float(sp.zmax_rg), "lj": float(sp.lj), "a_inj": float(sp.a_inj),
            "zmax_inj": float(sp.zmax_inj_rg), "gammaindex": float(sp.gammaindex),
            "geomindex_out": float(sp.geomindex_out), "z0": float(sp.z0_rg), "fnt": float(sp.fnt),
            "facc": float(sp.facc), "p": float(sp.p)}


def spectrum(pars):
    """F_nu [erg s^-1 cm^-2 Hz^-1] of jet + counter-jet for the namelist with pars (lengths in R_g) substituted."""
    sp = jetcode.setup(NML)
    tmp = OUT / f"tmp_{os.getpid()}"
    sp.set_outdir(str(tmp))
    rg = float(sp.r_g)
    J = jetcode.module()
    s = J.syn_model(pars["sigma_0"], pars["zmax"] * rg, pars["lj"], pars["a_inj"], pars["zmax_inj"] * rg,
                    pars["gammaindex"], pars["geomindex_out"], pars["z0"] * rg, pars["fnt"], pars["facc"],
                    pars["p"], h * NU)
    return np.asarray(s)[:len(NU)] * h


def compute(nproc):
    base = base_params()
    jobs = [(k, v) for k, (vals, _, _) in SCANS.items() for v in vals]
    pars = [dict(base, **{k: v}) for k, v in jobs]
    with get_context("fork").Pool(nproc) as pool:
        spec = pool.map(spectrum, pars)
    for d in OUT.glob("tmp_*"):
        for f in d.iterdir():
            f.unlink()
        d.rmdir()
    res = {f"{k}={v!r}": s for (k, v), s in zip(jobs, spec)}
    np.savez(OUT / "scans.npz", nu=NU, base=json.dumps(base), **res)
    return base, res


def check_base(res, base):
    """The base-model spectrum of every scan must equal the run.x total of analysis/runs/jet_riaf."""
    f = np.loadtxt(RUNS / "jet_riaf" / "Flux_e.dat", comments="#")
    fc = np.loadtxt(RUNS / "jet_riaf" / "FluxCJ_e.dat", comments="#")
    ref = np.exp(np.interp(np.log(NU), np.log(f[:, 0]), np.log(f[:, -1] + fc[:, -1] + 1e-300)))
    sel = (NU > 1e9) & (NU < 1e20)
    out = {}
    for k in SCANS:
        s = res[f"{k}={base[k]!r}"]
        out[k] = float(np.max(np.abs(s[sel] / ref[sel] - 1)))
    return out


def plot(res, base, kind, outpath):
    fig, axs = plt.subplots(3, 3, figsize=(13, 11), sharex=True, sharey=True,
                            gridspec_kw={"hspace": 0.06, "wspace": 0.06})
    for ax, (k, (vals, label, fmt)) in zip(axs.flat, SCANS.items()):
        cols = plt.cm.viridis(np.linspace(0, 0.9, len(vals)))
        for v, c in zip(vals, cols):
            s = res[f"{k}={v!r}"]
            isbase = np.isclose(v, base[k])
            y = NU * s * 4 * np.pi * DIST**2 if kind == "sed" else s / 1e-23
            x = NU if kind == "sed" else NU / 1e9
            ax.loglog(x, np.where(y > 0, y, np.nan), color="k" if isbase else c, lw=2.6 if isbase else 1.8,
                      ls="--" if isbase else "-", label=fmt_value(v, fmt) + (" (model)" if isbase else ""), zorder=5 if isbase else 3)
        kw = dict(loc="lower center" if kind == "sed" else "lower left", ncol=2 if kind == "sed" else 1,
                  borderaxespad=0.3, columnspacing=0.8)
        kw.update((SED_LEGEND if kind == "sed" else RADIO_LEGEND).get(k, {}))
        ax.legend(title=label, fontsize=13, title_fontsize=14, handlelength=1.6, handletextpad=0.5,
                  frameon=False, **kw)
    if kind == "sed":
        axs[0, 0].set_xlim(1e8, 1e21)
        axs[0, 0].set_ylim(1e38, 3e41)
        axs[0, 0].xaxis.set_minor_locator(FixedLocator(10.0 ** np.arange(8, 22)))  # minor ticks every dex
        axs[0, 0].xaxis.set_minor_formatter(NullFormatter())
        xl, yl = r"$\nu$ [Hz]", r"$\nu L_\nu$ [erg s$^{-1}$]"
    else:
        axs[0, 0].set_xlim(1, 900)  # stops short of 10^3 so its label does not collide with the next panel's 10^0
        axs[0, 0].set_ylim(5e-3, 30)
        xl, yl = r"$\nu$ [GHz]", r"$S_\nu$ [Jy]"
    for ax in axs.flat:
        ax.tick_params(labelsize=13)
    for ax in axs[-1]:
        ax.set_xlabel(xl, fontsize=15)
    for ax in axs[:, 0]:
        ax.set_ylabel(yl, fontsize=15)
    fig.savefig(outpath, dpi=300, pad_inches=0)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=FIGURES)
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--nproc", type=int, default=8)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    base = base_params()
    cache = OUT / "scans.npz"
    stale = not cache.exists() or "base" not in np.load(cache).files or json.loads(str(np.load(cache)["base"])) != base
    if a.recompute or stale:
        _, res = compute(a.nproc)
    else:
        d = np.load(cache)
        res = {k: d[k] for k in d.files if k not in ("nu", "base")}
    chk = check_base(res, base)
    print("base model vs run.x (max |rel. diff|, 1e9-1e20 Hz):", json.dumps(chk))
    plot(res, base, "sed", a.outdir / "Scans_SED.pdf")
    plot(res, base, "radio", a.outdir / "Scans_radio.pdf")
    provenance.write(OUT / "provenance.json", code={"extended-jet": provenance.git_state(JETCODE)},
                     config=NML.name, base=base, scans={k: v[0] for k, v in SCANS.items()}, base_check=chk,
                     figures=["Scans_SED.pdf", "Scans_radio.pdf"], outdir=str(a.outdir))


if __name__ == "__main__":
    main()
