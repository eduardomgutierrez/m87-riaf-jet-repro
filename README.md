# The accretion flow and jet of M87 — reproduction package

This repository reproduces the model results, figures and numbers of

> E. M. Gutiérrez, S. del Palacio & T. T. Mazzei, *The accretion flow and jet of M87: broadband modelling across
> spatial scales*, accepted for publication in MNRAS ([arXiv:2610.26434](https://arxiv.org/abs/2610.26434)).

It contains the two model codes (as git submodules pinned to the exact commits used in the paper), the observational
data, the model configurations, and one script per figure or table.

The model has a thermal radiatively inefficient accretion flow (RIAF) with winds and a multizone, initially magnetically
dominated jet with continuous electron injection, cooling and synchrotron self-absorption. For each instrument, the jet
emission that projects within its effective aperture is compared with the flux measured at that angular scale (EHT
multi-wavelength campaign 2017). Two scenarios are modelled: *Jet+RIAF* (the RIAF dominates the EHT 230 GHz core) and
*Jet-only*.

This version corresponds to the accepted paper, in which the jet parameters of both scenarios are fitted to the data.
`make all` reproduces the two best-fitting (maximum-likelihood) models and everything derived from them. The fit itself,
a Markov chain Monte Carlo (MCMC) sampling of eight jet parameters, takes 18–21 hours per scenario on 12 cores and is run
separately (see [Fitting the jet parameters](#fitting-the-jet-parameters-mcmc)); it gives the posterior distributions
of Appendix B and the medians and ranges of the fitted parameters in Table 1. The quantities computed from posterior
samples (the shaded bands of Fig. 3 and the posterior ranges of the fluxes, the magnetic field and the jet efficiency)
are not produced here.
Commit `a97b73e` of this repository reproduces the first submitted version, whose models were tuned by hand.

## Contents

```
codes/extended-jet   ExtendedJet (jet code; Fortran 90 + OpenMP, f2py wrapper), tag paper-v2.3 = 1a8abf7
codes/thermal-riaf   thermal-riaf (RIAF code; Python hydrodynamics + C++ radiative transfer), tag paper-v1 = 5e0cd49
data/eht_mwl2017/    EHT MWL 2017 data release (EHT MWL Science WG 2021, ApJL 911, L11; DOI 10.25739/mhh2-cw46)
data/obs_data/       SED points per instrument: nu [Hz], nuF_nu [erg s^-1 cm^-2], error (negative = upper limit)
analysis/configs/    model parameters (see below)
analysis/m87/        shared Python code: data I/O, effective apertures, chi^2, plot style, provenance
analysis/scripts/    one script per figure or table, the model driver, the MCMC fit and the result check
reference/           the paper values that `make check` compares with
```

| Configuration | Model |
|---|---|
| `analysis/configs/riaf_m87.toml` | RIAF of the Jet+RIAF scenario (M = 6.5×10⁹ M☉, Ṁ(R) ∝ R^0.4, β = 0.9, α = 0.1, δ = 0.3) |
| `analysis/configs/jet_riaf.nml` | jet of the Jet+RIAF scenario (best fit) |
| `analysis/configs/jet_only.nml` | jet of the Jet-only scenario (best fit) |

Both jet models use d = 16.8 Mpc, i = 17°, effective aperture radii 7.3, 26, 65, 131, 195, 650 and 3.5×10⁵ R_g
(R_g = GM/c²), and the grid Ns = 700 segments × 1100 energy bins. The gravitational redshift of the emission
(`grav_redshift`) and the weighting of each jet segment by the fraction of its projected cross-section inside the
aperture (`aperture_area`) are switched on. The eight fitted parameters (L_j, ε_e, p, z_0, a_e, z_max,e, g and η_acc) are
`Lj`, `fnT`, `p`, `z0_rg`, `a_inj`, `zmax_inj_rg`, `gammaIndex` and `facc` in the namelists.

## Requirements

- Linux or macOS with `gfortran` (with OpenMP), a C++17 compiler with OpenMP, CMake ≥ 3.16, Boost and GSL
  (Debian/Ubuntu: `sudo apt install build-essential gfortran cmake libboost-dev libgsl-dev`)
- conda or mamba, for the Python environment in `environment.yml` (Python 3.12, numpy, scipy, astropy, matplotlib,
  meson/ninja for the f2py build, and bilby with emcee for the fits)

## Quick start

```bash
git clone --recurse-submodules https://github.com/eduardomgutierrez/m87-riaf-jet-repro.git
cd m87-riaf-jet-repro
conda env create -f environment.yml
conda activate m87
make all
```

`make all` builds both codes, runs every model, makes all figures and numbers, and finishes with `make check`.
It runs the following steps in order:

| Step | What it does | Output |
|---|---|---|
| `make build` | compiles `run.x` and the f2py module of the jet code, and the RIAF radiative code | inside `codes/` |
| `make data` | rebuilds the SED points from the public release and compares them with `data/obs_data/` | report on screen |
| `make riaf` | solves the RIAF hydrodynamics and computes its spectrum and derived numbers | `analysis/runs/riaf_m87/` |
| `make models` | runs the two jet models | `analysis/runs/{jet_riaf,jet_only}/` |
| `make figures` | figures, χ² table and derived numbers | `output/figures/`, `output/tables/`, `analysis/runs/` |
| `make check` | compares every number with the paper values in `reference/` | PASS/FAIL report |

The full chain takes about 3 minutes on a laptop: build 20 s, RIAF 70 s, jet models 10 s, figures 70 s.
Every run writes a `provenance.json` with the code commits and the parameters used.

## Paper results and where they come from

| Paper item | Script | Output |
|---|---|---|
| Fig. 2, jet power and magnetisation profiles (illustrative parameters) | `fig_jet_power_illustrative.py` | `output/figures/Jet_power.pdf` |
| Fig. 3, SEDs of the Jet-only and Jet+RIAF models per effective aperture (best-fitting models, without the posterior bands) | `fig_sed.py` | `output/figures/SED_FluxDensity.pdf` |
| χ² per band and in total (the χ²/(N − k) of the paper has k = 8 for the Jet-only and k = 10 for the Jet+RIAF model) | `chi2_table.py` | `output/tables/chi2.tex`, `analysis/runs/chi2.json` |
| Fig. 4, synthetic 43 GHz map | `fig_maps.py` | `output/figures/Map_I_43GHz.pdf`, `analysis/runs/maps.provenance.json` |
| Fig. 5, spectral-index maps 22–43 and 43–86 GHz | `fig_maps.py` | `output/figures/Map_spix.pdf` |
| Figs. A1–A2, parameter scans around the Jet+RIAF jet | `fig_scans.py` | `output/figures/Scans_SED.pdf`, `Scans_radio.pdf` |
| RIAF numbers (Ṁ, L_acc, jet efficiency, compactness of the RIAF emission) | `riaf_derived.py` (run by `make riaf`) | `analysis/runs/riaf_m87/provenance.json` |
| Jet numbers (τ = 1 heights and core shift, B′ and σ′ along the jet, 230 GHz flux within a radius) | `derived_quantities.py` | `analysis/runs/derived/derived.json` |
| Inverse-Compton estimate | `ic_estimate.py` | `analysis/runs/derived/ic_estimate.json` |
| Fitted parameters (posterior medians and ranges of Table 1, posterior distributions of Appendix B) | `jet_mcmc.py` (not run by `make all`; see below) | `analysis/runs/mcmc/<fit>/summary.json`, `corner.png` |

Figure 1 of the paper is a schematic and is not produced here; neither are the quantities computed from posterior
samples (see above). Each script can also be run on its own from the
repository root, e.g. `python analysis/scripts/fig_sed.py --outdir somewhere/`; `--help` lists the options.

## Checking the reproduction

`make check` (`analysis/scripts/check_results.py`) compares every number of the outputs above with the values of the
paper runs in `reference/expected.json`, and every spectrum of the parameter scans with `reference/scans.npz`, to a
relative tolerance of 10⁻³. It also reports whether the model output files are bit-identical to the paper runs. With
gfortran 15 on x86-64 they are. Other compilers or CPUs can change the last digits (the jet code is built with
`-O3 -march=native -ffast-math`), so a hash difference is only a note; the numerical comparison is the test.

## Fitting the jet parameters (MCMC)

`analysis/scripts/jet_mcmc.py` runs the fits of the paper (Appendix B). It samples the posterior distribution of eight
jet parameters with the ensemble sampler `emcee`, through `bilby`. It is not part of `make all`.

```bash
make build riaf                                           # once: the codes and the RIAF spectrum
make mcmc-test                                            # quick test, about 6 minutes on 4 cores
python analysis/scripts/jet_mcmc.py jet_riaf --npool 12   # the chain of the Jet+RIAF scenario
python analysis/scripts/jet_mcmc.py jet_only --npool 12   # the chain of the Jet-only scenario
```

`make mcmc NPOOL=12` runs the two chains one after the other.

| Parameter | Namelist key | Prior range | Uniform in |
|---|---|---|---|
| Jet power, L_j [erg s⁻¹] | `Lj` | 10⁴²–10⁴⁴ | log |
| Fraction of L_j injected into electrons, ε_e | `fnT` | 10⁻³–0.3 | log |
| Electron power-law index, p | `p` | 1.9–2.6 | linear |
| Jet base height, z_0 [R_g] | `z0_rg` | 2–50 | log |
| Injection profile index, a_e | `a_inj` | −0.5–2 | linear |
| Injection scale height, z_max,e [R_g] | `zmax_inj_rg` | 10³–10⁵ | log |
| Bulk acceleration parameter, g | `gammaIndex` | 0.10–0.20 | linear |
| Acceleration efficiency, η_acc | `facc` | 10⁻⁴–0.1 | log |

- **Model.** Only these eight parameters are sampled. Everything else stays at the values of
  `analysis/configs/<fit>.nml`, and the RIAF of the Jet+RIAF scenario is fixed.
- **Likelihood.** Gaussian, with the χ² of the paper: 45 points, each compared with the model flux within its effective
  aperture, with a systematic uncertainty of 10 per cent added in quadrature. A model above an upper limit is rejected.
- **Chain.** 48 walkers and 3000 steps by default (`--nwalkers`, `--nsteps`), started from the walker positions of the
  paper's chains. The first 30 per cent of the steps are discarded as burn-in.
- **Cost.** Each model takes 6–7 s on one core, and each worker needs about 0.8 GB of memory. A chain computes about
  9×10⁴ models and takes 18–21 hours with `--npool 12`. The chain is saved every 100 steps (`--segment`), and the same command
  continues an interrupted chain.
- **Outputs**, in `analysis/runs/mcmc/<fit>/`: `emcee_<fit>/chain.dat` (the chain), `evals/*.jsonl` (every model
  computed, with its parameters and χ²), `summary.json` (acceptance fraction, autocorrelation times, posterior medians
  with the 16th and 84th percentiles, and the maximum-likelihood sample with its χ²), `best_fit.nml` (that sample as a
  namelist) and `corner.png`. `--summary` rebuilds the summary from a stored chain. `make clean` keeps the chains.

`make mcmc-test` checks, for both scenarios, that the likelihood at the published best fit gives the χ² of the paper,
and runs a 3-step chain with a restart. A new chain will not be identical to the paper's, because the random numbers
of the sampler are not fixed: expect its medians to agree with Table 1 within the sampling error, and its maximum-likelihood sample to
differ slightly from the published best fit. The paper chains were run with bilby 2.8.2 and emcee 3.1.6.

## Codes

- **ExtendedJet** (S. del Palacio et al.; https://github.com/santimda/ExtendedJet, GPL-3.0), partly based on Molina,
  del Palacio & Bosch-Ramon (2019, A&A 629, A129). After `make build`, the jet code's own
  tests run with `make -C codes/extended-jet/Code test` and its physics validation with
  `make -C codes/extended-jet/Code validate`.
- **thermal-riaf** (E. M. Gutiérrez; https://github.com/eduardomgutierrez/thermal-riaf), based on Gutiérrez, Nemmen &
  Cafardo (2020, ApJL 891, L36; https://doi.org/10.3847/2041-8213/ab7998) and Gutiérrez, Vieyro & Romero (2021, A&A 649,
  A87; https://doi.org/10.1051/0004-6361/202039671).

## Data

`data/eht_mwl2017/` is the public data release of the EHT 2017 multi-wavelength campaign (EHT MWL Science WG 2021,
ApJL 911, L11; https://doi.org/10.25739/mhh2-cw46). `data/obs_data/` holds the SED points used in the paper, one file
per instrument. `analysis/scripts/make_obs_data.py` rebuilds them from the release, and lists, with its reason, every
value that differs from a plain conversion.

## License

The scripts in this repository are distributed under the GNU GPL v3 (`LICENSE`); the codes keep their own licenses.

## Citation

If you use this material, please cite the paper (accepted for publication in MNRAS;
[arXiv:2610.26434](https://arxiv.org/abs/2610.26434)):

```bibtex
@ARTICLE{2026arXiv261026434G,
       author = {{Guti{\'e}rrez}, E.~M. and {del Palacio}, S. and {Mazzei}, T.~T.},
        title = "{The accretion flow and jet of M87: broadband modelling across spatial scales}",
      journal = {arXiv e-prints},
         year = 2026,
        month = oct,
          eid = {arXiv:2610.26434},
archivePrefix = {arXiv},
       eprint = {2610.26434},
 primaryClass = {astro-ph.HE},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2026arXiv261026434G}
}
```
