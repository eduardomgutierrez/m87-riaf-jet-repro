# The accretion flow and jet of M87 — reproduction package

This repository reproduces the model results, figures and numbers of

> E. M. Gutiérrez, S. del Palacio & T. T. Mazzei, *The accretion flow and jet of M87: broadband modelling across
> spatial scales*, submitted.

It contains the two model codes (as git submodules pinned to the exact commits used in the paper), the observational
data, the model configurations, and one script per figure or table.

The model has a thermal radiatively inefficient accretion flow (RIAF) with winds and a multizone, initially magnetically
dominated jet with continuous electron injection, cooling and synchrotron self-absorption. The jet emission is integrated
only up to the height matching each instrument's effective aperture and compared with the flux measured at that
angular scale (EHT multi-wavelength campaign 2017). Two scenarios are modelled: *Jet+RIAF* (the RIAF dominates the EHT
230 GHz core) and *Jet-only*.

## Contents

```
codes/extended-jet   ExtendedJet (jet code; Fortran 90 + OpenMP, f2py wrapper), tag paper-v2 = 947d7b0
codes/thermal-riaf   thermal-riaf (RIAF code; Python hydrodynamics + C++ radiative transfer), tag paper-v1 = 5e0cd49
data/eht_mwl2017/    EHT MWL 2017 data release (EHT MWL Science WG 2021, ApJL 911, L11; DOI 10.25739/mhh2-cw46)
data/obs_data/       SED points per instrument: nu [Hz], nuF_nu [erg s^-1 cm^-2], error (negative = upper limit)
analysis/configs/    model parameters (see below)
analysis/m87/        shared Python code: data I/O, effective apertures, chi^2, plot style, provenance
analysis/scripts/    one script per figure or table, the model driver and the result check
reference/           the paper values that `make check` compares with
```

| Configuration | Model |
|---|---|
| `analysis/configs/riaf_m87.toml` | RIAF of the Jet+RIAF scenario (M = 6.5×10⁹ M☉, Ṁ(R) ∝ R^0.4, β = 0.9, α = 0.1, δ = 0.3) |
| `analysis/configs/jet_riaf.nml` | jet of the Jet+RIAF scenario |
| `analysis/configs/jet_only.nml` | representative Jet-only model ("d1") |
| `analysis/configs/ro_B.nml`, `ro_C.nml` | Jet+RIAF variants B and C, with a larger jet power (Ro et al. 2023 magnetic field) |

All jet models use d = 16.8 Mpc, i = 17°, effective aperture radii 7.3, 26, 65, 131, 195, 650 and 3.5×10⁵ R_g
(R_g = GM/c²), and the grid Ns = 700 segments × 1100 energy bins.

## Requirements

- Linux or macOS with `gfortran` (with OpenMP), a C++17 compiler with OpenMP, CMake ≥ 3.16, Boost and GSL
  (Debian/Ubuntu: `sudo apt install build-essential gfortran cmake libboost-dev libgsl-dev`)
- conda or mamba, for the Python environment in `environment.yml` (Python 3.12, numpy, scipy, astropy, matplotlib,
  meson/ninja for the f2py build)

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
| `make models` | runs the four jet models | `analysis/runs/{jet_riaf,jet_only,ro_B,ro_C}/` |
| `make figures` | figures, χ² table and derived numbers | `output/figures/`, `output/tables/`, `analysis/runs/` |
| `make check` | compares every number with the paper values in `reference/` | PASS/FAIL report |

The full chain takes about 3 minutes on a laptop: build 20 s, RIAF 70 s, jet models 10 s, figures 70 s.
Every run writes a `provenance.json` with the code commits and the parameters used.

## Paper results and where they come from

| Paper item | Script | Output |
|---|---|---|
| Fig. 2, jet power and magnetisation profiles (illustrative parameters) | `fig_jet_power_illustrative.py` | `output/figures/Jet_power.pdf` |
| Fig. 3, SEDs of the Jet-only and Jet+RIAF models per effective aperture | `fig_sed.py` | `output/figures/SED_FluxDensity.pdf` |
| χ² per band and in total | `chi2_table.py` | `output/tables/chi2.tex`, `analysis/runs/chi2.json` |
| Fig. 4, synthetic 43 GHz map | `fig_maps.py` | `output/figures/Map_I_43GHz.pdf`, `analysis/runs/maps.provenance.json` |
| Fig. 5, spectral-index maps 22–43 and 43–86 GHz | `fig_maps.py` | `output/figures/Map_spix.pdf` |
| Figs. A1–A2, parameter scans around the Jet+RIAF jet | `fig_scans.py` | `output/figures/Scans_SED.pdf`, `Scans_radio.pdf` |
| RIAF numbers (Ṁ, L_acc, jet efficiency, compactness of the RIAF emission) | `riaf_validate.py` (run by `make riaf`) | `analysis/runs/riaf_m87/provenance.json` |
| Jet numbers (τ = 1 heights and core shift, B′ and σ′ along the jet, 230 GHz flux within a radius) | `derived_quantities.py` | `analysis/runs/derived/derived.json` |
| Inverse-Compton estimate | `ic_estimate.py` | `analysis/runs/derived/ic_estimate.json` |
| Ro et al. (2023)-compatible variants (B′, jet efficiency, χ²) | `ro_variants.py` | `analysis/runs/ro_variants.json` |

Figure 1 of the paper is a schematic and is not produced here. Each script can also be run on its own from the
repository root, e.g. `python analysis/scripts/fig_sed.py --outdir somewhere/`; `--help` lists the options.

## Checking the reproduction

`make check` (`analysis/scripts/check_results.py`) compares every number of the outputs above with the values of the
paper runs in `reference/expected.json`, and every spectrum of the parameter scans with `reference/scans.npz`, to a
relative tolerance of 10⁻³. It also reports whether the model output files are bit-identical to the paper runs. With
gfortran 15 on x86-64 they are. Other compilers or CPUs can change the last digits (the jet code is built with
`-O3 -march=native -ffast-math`), so a hash difference is only a note; the numerical comparison is the test.

## Codes

- **ExtendedJet** (S. del Palacio et al.; https://github.com/santimda/ExtendedJet, GPL-3.0), partly based on Molina,
  del Palacio & Bosch-Ramon (2019, A&A 629, A129). The jet code's own tests run with
  `make -C codes/extended-jet/Code test` after `make build`.
- **thermal-riaf** (E. M. Gutiérrez; https://github.com/eduardomgutierrez/thermal-riaf), based on Gutiérrez, Nemmen &
  Cafardo (2020, ApJL 891, L36; https://doi.org/10.3847/2041-8213/ab7998) and Gutiérrez, Vieyro & Romero (2021, A&A 649,
  A87; https://doi.org/10.1051/0004-6361/202039671).

## Data

`data/eht_mwl2017/` is the public data release of the EHT 2017 multi-wavelength campaign (EHT MWL Science WG 2021,
ApJL 911, L11; https://doi.org/10.25739/mhh2-cw46). `data/obs_data/` holds the SED points used in the paper, one file
per instrument. `analysis/scripts/make_obs_data.py` rebuilds them from the release, and lists, with its reason, every
value that differs from a plain conversion.

## License and citation

The scripts in this repository are distributed under the GNU GPL v3 (`LICENSE`); the codes keep their own licenses.
If you use this material, please cite the paper above.
