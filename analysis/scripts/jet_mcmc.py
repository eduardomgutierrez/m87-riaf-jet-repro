"""MCMC fit of the jet parameters to the SED of M87 (Appendix B of the paper), with bilby and the emcee sampler.

    python analysis/scripts/jet_mcmc.py FIT [--npool N]          run a chain, or continue a stored one
    python analysis/scripts/jet_mcmc.py FIT --test [--npool N]   quick test: the best fit of the paper and a 3-step chain
    python analysis/scripts/jet_mcmc.py FIT --summary            summarise the stored chain again

FIT = jet_riaf   jet of the Jet+RIAF scenario; the RIAF spectrum (analysis/runs/riaf_m87) is added and kept fixed
      jet_only   Jet-only scenario
Needs `make build`, and `make riaf` for jet_riaf.

Parameters. The eight jet parameters of PARAMS are sampled with uniform priors (uniform in log10 where marked). All
the other parameters and switches come from the namelist of the model, analysis/configs/FIT.nml; the values of the
eight parameters in that file are the best fit of the paper.

Likelihood. Gaussian, ln L = -chi^2/2 - sum_i ln(2 pi sigma_i^2)/2, with the chi^2 of the paper (analysis/m87/chi2.py):
the 45 points that are neither upper limits nor gamma rays, each compared with the model flux within its own effective
aperture, with sigma_i^2 = err_i^2 + (0.1 F_i)^2. A model above an upper limit, or a failed run, gets ln L = -1e30.
Each evaluation is one run of codes/extended-jet/Code/run.x (6-7 s on one core; about 0.8 GB with its Python worker).

Chain. 48 walkers and 3000 steps by default, started in a Gaussian ball around the starting point of the paper's chain
(SEEDS; sigma = 0.1 x half the prior range in the sampled variable). The first 30 per cent of the steps are discarded
as burn-in. emcee evaluates the ensemble in two halves, so nwalkers = 2 x npool x k keeps the pool busy. bilby saves
the sampler only when a call ends, so the chain runs in segments of --segment steps, and the same command continues an
interrupted chain up to --nsteps. A chain of 3000 steps computes about 9e4 models and takes 18-21 hours with
--npool 12. A new chain is statistically equivalent to the paper's, not identical to it.

Outputs, in analysis/runs/mcmc/FIT/:
  emcee_FIT/chain.dat   the chain: walker, the eight sampled variables, ln L and ln prior; one row per walker and step
  evals/*.jsonl         every model computed, with its parameters and chi^2 (one file per worker)
  summary.json          convergence (acceptance fraction, autocorrelation times, effective samples), posterior medians
                        with the 16th and 84th percentiles, and the maximum-likelihood sample of the whole chain with
                        its chi^2 and chi^2/(N - k)
  best_fit.nml          the maximum-likelihood sample as a namelist for run.x
  corner.png            corner plot of the posterior
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import bilby  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from bilby.core.prior import PriorDict, Uniform  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m87 import chi2, provenance  # noqa: E402
from m87.data import GAMMA, load_obs  # noqa: E402
from m87.models import load_jet_model  # noqa: E402
from m87.paths import CONFIGS, JETCODE, ROOT, RUNS  # noqa: E402

FITS = {  # fit -> (RIAF run added to the jet, label in reference/expected.json, k of chi^2/(N - k))
    "jet_riaf": ("riaf_m87", "Jet+RIAF", 10),   # the 8 jet parameters + the 2 RIAF parameters tuned by hand
    "jet_only": (None, "Jet-only", 8),
}
# (namelist key, symbol, lower, upper, sampled in log10?): the prior ranges of Table 1; lengths in R_g, L_j in erg/s
PARAMS = [
    ("lj", r"L_{\rm j}", 1e42, 1e44, True),
    ("fnt", r"\epsilon_{\rm e}", 1e-3, 0.3, True),
    ("p", r"p", 1.9, 2.6, False),
    ("z0_rg", r"z_0/R_{\rm g}", 2.0, 50.0, True),
    ("a_inj", r"a_{\rm e}", -0.5, 2.0, False),
    ("zmax_inj_rg", r"z_{\rm max,e}/R_{\rm g}", 1e3, 1e5, True),
    ("gammaindex", r"g", 0.10, 0.20, False),
    ("facc", r"\eta_{\rm acc}", 1e-4, 0.1, True),
]
NAMES = [f"log10_{key}" if lg else key for key, _, _, _, lg in PARAMS]   # the sampled variables
# Starting points of the paper's chains: the best models of an earlier, wider exploration of the parameters
SEEDS = {
    "jet_riaf": {"lj": 8.020972314708728e+42, "fnt": 0.052379288238360015, "p": 2.1743961477873,
                 "z0_rg": 14.383977560837657, "a_inj": -0.21333769706972866, "zmax_inj_rg": 19880.92063131654,
                 "gammaindex": 0.19732528045142547, "facc": 0.054573388281871194},
    "jet_only": {"lj": 6.848912753915285e+42, "fnt": 0.2242320733755939, "p": 2.1457274068647307,
                 "z0_rg": 2.206693044707988, "a_inj": 0.785445022552266, "zmax_inj_rg": 80791.38681081127,
                 "gammaindex": 0.19736000063007236, "facc": 0.0913598910555617},
}
FLOOR = 0.1          # systematic uncertainty added to every point, as a fraction of the measured flux
BAD = -1e30          # ln L of failed runs and of models above an upper limit
BURN_FRAC = 0.3      # fraction of the steps discarded as burn-in
MCMC = RUNS / "mcmc"


# ---------------------------------------------------------------------------------------------- parameters and priors
def to_physical(sample):
    """{sampled variable: value(s)} -> {namelist key: value(s)} in physical units."""
    return {key: 10 ** np.asarray(sample[name], float) if lg else np.asarray(sample[name], float)
            for name, (key, _, _, _, lg) in zip(NAMES, PARAMS)}


def to_sampled(pars):
    """{namelist key: value} -> {sampled variable: value}."""
    return {name: float(np.log10(pars[key])) if lg else float(pars[key])
            for name, (key, _, _, _, lg) in zip(NAMES, PARAMS)}


def latex_label(name):
    key, sym, _, _, lg = PARAMS[NAMES.index(name)]
    return rf"$\log_{{10}}\,{sym}$" if lg else rf"${sym}$"


def make_priors():
    pri = PriorDict()
    for name, (key, sym, lo, hi, lg) in zip(NAMES, PARAMS):
        lo, hi = (np.log10(lo), np.log10(hi)) if lg else (lo, hi)
        pri[name] = Uniform(lo, hi, name=name, latex_label=latex_label(name))
    return pri


def initial_positions(priors, seed, nwalkers, perturb, rng):
    """Walkers in a Gaussian ball around the seed: sigma = perturb x half the prior range in the sampled variable. A
    coordinate drawn outside its range is redrawn. Walker 0 is the seed itself."""
    s0 = to_sampled(seed)
    rows = [s0]
    while len(rows) < nwalkers:
        row = {}
        for name in NAMES:
            lo, hi = priors[name].minimum, priors[name].maximum
            v = np.inf
            while not lo <= v <= hi:
                v = s0[name] + perturb * 0.5 * (hi - lo) * rng.standard_normal()
            row[name] = v
        rows.append(row)
    return pd.DataFrame(rows)[NAMES]


def write_namelist(groups, path):
    """Write {group: {key: value}} (as read by provenance.read_namelist) as a Fortran namelist."""
    def fmt(v):
        if isinstance(v, bool):
            return ".true." if v else ".false."
        if isinstance(v, str):
            return f"'{v}'"
        if isinstance(v, list):
            return ", ".join(fmt(x) for x in v)
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        return f"{float(v):.10g}"
    lines = ["! written by analysis/scripts/jet_mcmc.py"]
    for group, d in groups.items():
        lines.append(f"&{group}")
        lines += [f"  {k} = {fmt(v)}" for k, v in d.items()]
        lines.append("/")
    Path(path).write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------------------------------------- likelihood
class JetSEDLikelihood(bilby.Likelihood):
    """Gaussian likelihood of the SED for one run.x model (see the module docstring)."""

    def __init__(self, fit, base=None):
        super().__init__()
        self.fit, self.riaf = fit, FITS[fit][0]
        self.base = Path(base) if base else MCMC / fit   # work directories and the log of the evaluations go here
        self.nml = provenance.read_namelist(CONFIGS / f"{fit}.nml")
        self.obs = load_obs()
        sig2 = np.array([o.err[i]**2 + (FLOOR * o.nufnu[i])**2 for name, o in self.obs.items() if name not in GAMMA
                         for i in range(len(o.nu)) if not o.uplim[i]])
        self.n_points, self.norm = len(sig2), float(np.sum(np.log(2 * np.pi * sig2)))   # N and sum ln(2 pi sigma^2)
        self.meta_data = {"fit": fit, "floor": FLOOR}

    def best_fit(self):
        """The eight parameters of the model's namelist: the best fit of the paper."""
        return {key: float(self.nml["model"][key]) for key, *_ in PARAMS}

    def log_likelihood(self, parameters=None):
        if parameters is None:
            parameters = self.parameters
        return self.evaluate({k: float(v) for k, v in to_physical(parameters).items()})["log_l"]

    def namelist(self, pars, outdir):
        """The model's namelist with the eight parameters pars, writing to outdir (inside the repository)."""
        nml = {group: dict(d) for group, d in self.nml.items()}
        nml["model"].update(pars)
        nml["run"]["outdir"] = str(Path(outdir).relative_to(ROOT))
        return nml

    def evaluate(self, pars, outdir=None):
        """Run run.x with the eight parameters pars and score the model. With outdir, the outputs stay there and the
        evaluation is not logged; otherwise the directory of the worker is reused and the evaluation is logged."""
        pars = dict(pars)
        if abs(pars["a_inj"]) < 1e-3:   # the normalisation of the injection profile is 0/0 at a_e = 0
            pars["a_inj"] = 1e-3 if pars["a_inj"] >= 0 else -1e-3
        out = Path(outdir) if outdir else self.base / "work" / str(os.getpid())
        out.mkdir(parents=True, exist_ok=True)
        for f in ("Flux_e.dat", "FluxCJ_e.dat"):
            (out / f).unlink(missing_ok=True)
        write_namelist(self.namelist(pars, out), out / "trial.nml")
        rec = {"params": pars}
        t0 = time.time()
        with open(out / "run.log", "w") as log:
            proc = subprocess.run([str(JETCODE / "run.x"), str((out / "trial.nml").relative_to(ROOT))], cwd=ROOT,
                                  env={**os.environ, "OMP_NUM_THREADS": "1"}, stdout=log, stderr=subprocess.STDOUT)
        if proc.returncode != 0 or not (out / "Flux_e.dat").exists():
            rec.update(log_l=BAD, chi2=None, rejected=f"run.x exit {proc.returncode}")
        else:
            try:
                pts = chi2.evaluate(load_jet_model(str(out.relative_to(RUNS)), self.riaf), self.obs, FLOOR)
                rec["chi2"] = float(sum(q.chi2 for q in pts if np.isfinite(q.chi2)))
                if any(q.uplim and q.band != "gamma" and q.model > q.data for q in pts):
                    rec.update(log_l=BAD, rejected="upper limit")
                else:
                    rec["log_l"] = -0.5 * rec["chi2"] - 0.5 * self.norm
            except Exception as e:   # one failed evaluation must not stop the chain
                rec.update(log_l=BAD, chi2=None, rejected=f"score: {type(e).__name__}: {e}")
        rec["runtime_s"] = round(time.time() - t0, 2)
        if not outdir:
            (self.base / "evals").mkdir(parents=True, exist_ok=True)
            with open(self.base / "evals" / f"{os.getpid()}.jsonl", "a") as f:
                f.write(json.dumps(rec) + "\n")
        return rec


# ------------------------------------------------------------------------------------------------------------- chain
def steps_done(outdir, label):
    """Steps stored in bilby's emcee checkpoint (0 if there is none)."""
    f = Path(outdir) / f"emcee_{label}" / "sampler.pickle"
    if not f.exists() or f.stat().st_size == 0:
        return 0
    import dill
    with open(f, "rb") as fh:
        return len(dill.load(fh).blobs)


def trim_chain(outdir, label, n_steps, nwalkers):
    """Cut chain.dat back to the checkpoint (header + n_steps x nwalkers rows): after an interruption bilby resumes
    from the checkpoint but appends to chain.dat. Blank lines between rows are dropped."""
    f = Path(outdir) / f"emcee_{label}" / "chain.dat"
    if not f.exists() or n_steps == 0:
        return
    head, *rows = [line for line in f.read_text().splitlines(keepends=True) if line.strip()]
    keep = n_steps * nwalkers
    if len(rows) > keep:
        f.write_text(head + "".join(rows[:keep]))
        print(f"chain.dat trimmed from {len(rows)} to {keep} rows (checkpoint at step {n_steps})", flush=True)


def run_chain(fit, like, priors, outdir, label, nwalkers, nsteps, segment, npool, pos0):
    """Run (or continue) the emcee chain up to nsteps, in segments of `segment` steps."""
    done = steps_done(outdir, label)
    targets = list(range(done + segment, nsteps, segment)) + [nsteps] if done < nsteps else [nsteps]
    for target in targets:
        done = steps_done(outdir, label)
        trim_chain(outdir, label, done, nwalkers)
        (Path(outdir) / f"{label}_result.json").unlink(missing_ok=True)   # else bilby returns the stored result
        t0 = time.time()
        print(f"{fit}: steps {done} -> {target} ({nwalkers} walkers, npool {npool})", flush=True)
        bilby.run_sampler(likelihood=like, priors=priors, sampler="emcee", nwalkers=nwalkers, nsteps=target,
                          pos0=pos0, nburn=int(BURN_FRAC * target), npool=npool, outdir=str(outdir), label=label,
                          resume=True, soft_init=True, save="json", plot=False)
        print(f"{fit}: step {target} done in {time.time() - t0:.0f} s", flush=True)


def read_chain(outdir, label):
    """(chain (nsteps, nwalkers, 8), ln L (nsteps, nwalkers)) from chain.dat; complete steps only."""
    f = Path(outdir) / f"emcee_{label}" / "chain.dat"
    keys = f.open().readline().split()[1:-2]
    assert keys == NAMES, f"{f}: columns {keys}, expected {NAMES}"
    d = np.loadtxt(f, skiprows=1, ndmin=2)
    nw = int(d[:, 0].max()) + 1
    ns = len(d) // nw
    d = d[:ns * nw].reshape(ns, nw, -1)
    assert np.all(d[:, :, 0] == np.arange(nw)), f"{f}: rows are not ordered by step and walker"
    return d[:, :, 1:-2], d[:, :, -2]


# ----------------------------------------------------------------------------------------------------------- summary
def summarise(fit, like, outdir, label, plot=True):
    """summary.json, best_fit.nml and corner.png of the chain stored in outdir; returns the summary."""
    import emcee
    chain, logl = read_chain(outdir, label)
    ns, nw, nd = chain.shape
    nburn = int(BURN_FRAC * ns)
    post = chain[nburn:].reshape(-1, nd)
    acc = np.any(np.diff(chain, axis=0) != 0, axis=2).mean(axis=0) if ns > 1 else np.zeros(nw)
    tau = np.full(nd, np.nan)                         # integrated autocorrelation time of each sampled variable
    if ns - nburn >= 50:                              # meaningless for shorter chains (e.g. the test)
        tau = emcee.autocorr.integrated_time(chain[nburn:], tol=0)
    tau_max = float(np.max(tau)) if np.all(np.isfinite(tau)) else None
    quant = {}
    for i, (name, (key, _, _, _, lg)) in enumerate(zip(NAMES, PARAMS)):
        lo, med, hi = (float(10 ** v) if lg else float(v) for v in np.percentile(post[:, i], [16, 50, 84]))
        quant[key] = {"p16": lo, "median": med, "p84": hi}
    imax = int(np.argmax(logl))                       # over the whole chain, burn-in included
    best = {k: float(v) for k, v in to_physical(dict(zip(NAMES, chain.reshape(-1, nd)[imax]))).items()}
    chi2_best = float(-2 * logl.ravel()[imax] - like.norm)
    k = FITS[fit][2]
    write_namelist(like.namelist(best, Path(outdir) / "best_fit"), Path(outdir) / "best_fit.nml")
    out = provenance.write(
        Path(outdir) / "summary.json", code={"extended-jet": provenance.git_state(JETCODE.parent)},
        script="analysis/scripts/jet_mcmc.py", fit=fit,
        priors={key: {"min": lo, "max": hi, "uniform_in": "log10" if lg else "linear"} for key, _, lo, hi, lg in PARAMS},
        seed_params=SEEDS[fit], floor=FLOOR,
        convergence={"nsteps": ns, "nwalkers": nw, "nburn": nburn, "acceptance_mean": float(acc.mean()),
                     "tau": dict(zip(NAMES, (float(t) for t in tau))) if tau_max else None, "tau_max": tau_max,
                     "n_eff": float(nw * (ns - nburn) / tau_max) if tau_max else None},
        posterior_p16_median_p84=quant,
        max_likelihood={"params": best, "log_l": float(logl.ravel()[imax]), "chi2": chi2_best,
                        "N": like.n_points, "k": k, "chi2_red": chi2_best / (like.n_points - k)})
    c, ml = out["convergence"], out["max_likelihood"]
    print(f"\n{fit}: {ns} steps x {nw} walkers, burn-in {nburn} steps, acceptance {c['acceptance_mean']:.2f}"
          + (f", tau_max {tau_max:.0f} steps, {c['n_eff']:.0f} independent samples" if tau_max else ""))
    print(f"  {'parameter':<13}{'median':>12}{'16th':>12}{'84th':>12}{'max. like.':>13}")
    for key, q in quant.items():
        print(f"  {key:<13}{q['median']:>12.4g}{q['p16']:>12.4g}{q['p84']:>12.4g}{best[key]:>13.4g}")
    print(f"  maximum likelihood: chi2 = {ml['chi2']:.2f}, chi2/(N - k) = {ml['chi2_red']:.2f} (N = {ml['N']}, k = {k})")
    if plot and len(post) > nd:
        import corner
        fig = corner.corner(post, labels=[latex_label(n) for n in NAMES], quantiles=[0.16, 0.5, 0.84],
                            show_titles=True, title_fmt=".2f")
        fig.savefig(Path(outdir) / "corner.png", dpi=110)
    return out


# -------------------------------------------------------------------------------------------------------------- test
def test(fit, like, priors, args):
    """The likelihood at the best fit of the paper, against the chi^2 of reference/expected.json; then a chain of 3
    steps (2 steps, continued to 3) with 16 walkers. Exits with status 1 if a check fails."""
    outdir = like.base
    ref = json.load(open(ROOT / "reference" / "expected.json"))["sections"]["chi2"]["values"][FITS[fit][1]]["Total"]
    rec = like.evaluate(like.best_fit(), outdir=outdir / "best_fit")
    ok_fit = rec["chi2"] is not None and abs(rec["chi2"] / ref["chi2"] - 1) < 1e-3 and like.n_points == ref["N"]
    print(f"{fit}: best fit of the paper: chi2 = {rec['chi2']:.4f} for {like.n_points} points (paper: "
          f"{ref['chi2']:.4f} for {ref['N']}), ln L = {rec['log_l']:.4f}, {rec['runtime_s']:.1f} s -> "
          f"{'PASS' if ok_fit else 'FAIL'}", flush=True)
    nw = 2 * len(NAMES)
    pos0 = initial_positions(priors, SEEDS[fit], nw, args.perturb, np.random.default_rng(args.seed))
    for f in (outdir / "emcee_test").glob("*"):
        f.unlink()
    run_chain(fit, like, priors, outdir, "test", nw, 2, 2, args.npool, pos0)
    run_chain(fit, like, priors, outdir, "test", nw, 3, 1, args.npool, pos0)
    chain, logl = read_chain(outdir, "test")
    ok_chain = steps_done(outdir, "test") == 3 and chain.shape == (3, nw, len(NAMES)) and bool(np.all(logl > BAD / 2))
    print(f"{fit}: test chain: {steps_done(outdir, 'test')} steps stored, chain {chain.shape}, "
          f"ln L from {logl.min():.1f} to {logl.max():.1f} -> {'PASS' if ok_chain else 'FAIL'}")
    summarise(fit, like, outdir, "test", plot=False)
    sys.exit(0 if ok_fit and ok_chain else 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("fit", choices=FITS)
    ap.add_argument("--npool", type=int, default=12, help="number of parallel run.x processes")
    ap.add_argument("--nwalkers", type=int, default=48)
    ap.add_argument("--nsteps", type=int, default=3000, help="total chain length (the run continues up to it)")
    ap.add_argument("--segment", type=int, default=100, help="steps between checkpoints")
    ap.add_argument("--seed", type=int, default=1, help="random seed of the initial walker positions")
    ap.add_argument("--perturb", type=float, default=0.1, help="Gaussian-ball width: fraction of half the prior range")
    ap.add_argument("--test", action="store_true", help="best-fit likelihood and a 3-step chain, then stop")
    ap.add_argument("--summary", action="store_true", help="only summarise the stored chain")
    args = ap.parse_args()
    if not (JETCODE / "run.x").exists():
        sys.exit("run.x not built: run `make build` first")
    riaf = FITS[args.fit][0]
    if riaf and not (RUNS / riaf / "spectrum" / "lumThermal.dat").exists():
        sys.exit(f"RIAF spectrum analysis/runs/{riaf} not found: run `make riaf` first")
    bilby.core.utils.random.seed(args.seed)
    outdir = MCMC / args.fit
    if args.test:   # everything the test writes stays in its own directory
        test(args.fit, JetSEDLikelihood(args.fit, base=outdir / "test"), make_priors(), args)
    like, priors = JetSEDLikelihood(args.fit), make_priors()
    if not args.summary:
        pos0 = initial_positions(priors, SEEDS[args.fit], args.nwalkers, args.perturb, np.random.default_rng(args.seed))
        run_chain(args.fit, like, priors, outdir, args.fit, args.nwalkers, args.nsteps, args.segment, args.npool, pos0)
    elif not (outdir / f"emcee_{args.fit}" / "chain.dat").exists():
        sys.exit(f"no chain in {outdir.relative_to(ROOT)}: run the fit first")
    summarise(args.fit, like, outdir, args.fit)
    print(f"{args.fit}: summary in {(outdir / 'summary.json').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
