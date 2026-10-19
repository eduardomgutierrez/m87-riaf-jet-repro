"""Derived numbers of the RIAF run (aperture flux fractions, Mdot, L_acc) -> analysis/runs/<run>/provenance.json."""
import numpy as np, json
from pathlib import Path
import sys
R = Path(__file__).resolve().parents[2]; run = R/"analysis/runs"/(sys.argv[1] if len(sys.argv) > 1 else "riaf_m87"); S = run/"spectrum"
d = 16.8*3.0857e24; Msun=1.989e33; yr=3.156e7; c=2.998e10; eV=1.602e-12; h=6.626e-27
tot = np.loadtxt(S/"lumThermal.dat"); rad = np.loadtxt(S/"lumThermal_radii.dat")
nu = tot[:,0]; nuL = tot[:,8]
print("max |lumThermal col8 - radii col 3.5e5| / col8:", np.max(abs(rad[:,6]-nuL)/np.maximum(nuL,1e-99)))
f = lambda y, x: np.exp(np.interp(np.log(x), np.log(nu), np.log(np.maximum(y,1e-99))))
out = {}
for name, x in [("22 GHz",22e9),("43 GHz",43e9),("86 GHz",86e9),("230 GHz",230e9),("optical 5e14 Hz",5e14)]:
    out[name] = f(rad[:,1],x)/f(rad[:,6],x); print(f"{name:18s} frac(r<7.3/cos i) = {out[name]:.3f}   nuLnu = {f(rad[:,6],x):.3e} erg/s")
lo, hi = 2e3*eV/h, 10e3*eV/h
g = np.logspace(np.log10(lo), np.log10(hi), 400)
for k,lab in [(1,"7.3"),(6,"3.5e5")]:
    print("X-ray 2-10 keV nuLnu integral, aperture", lab, np.trapezoid(f(rad[:,k],g)/g, g))
xr = np.trapezoid(f(rad[:,1],g)/g,g)/np.trapezoid(f(rad[:,6],g)/g,g); print(f"X-ray 2-10 keV frac = {xr:.3f}")
# fraction of the RIAF nuLnu inside each deprojected aperture R*/cos i (columns 1-5 of lumThermal_radii.dat; 6 = total)
ap_rg = [7.3, 26.0, 65.0, 195.0, 650.0]
frac_ap = {name: {f"{r:g}": round(f(rad[:,k],x)/f(rad[:,6],x), 3) for k, r in enumerate(ap_rg, start=1)}
           for name, x in [("22 GHz",22e9),("43 GHz",43e9),("86 GHz",86e9),("230 GHz",230e9)]}
print("frac(r < R*/cos i) per aperture R* [R_g]:", frac_ap)
Edd = 1.39e18*6.5e9; Rs = 1e4
Mout = json.load(open(run/"hydro/hydro-input.json")).get("accRateNorm", 4.2e-4)*Edd
R_IN = 1.1   # inner boundary of the hydro solution [R_S] (thermal-riaf RIAF/solve.py); L_acc below is Mdot(R_IN) c^2
mdot = {r: Mout*(r/Rs)**0.4 for r in (21, R_IN)}
for r, m in mdot.items():
    print(f"Mdot(r={r} Rs) = {m/Msun*yr:.3e} Msun/yr = {m:.3e} g/s")
dg = json.load(open(S/"diagnostics.json")); print("L_acc(inner) =", dg["inner_accretion_power_erg_s"], " L_rad =", dg["total_luminosity_erg_s"])
import subprocess, datetime
git = lambda *a: subprocess.run(["git", *a], cwd=R, capture_output=True, text=True).stdout.strip()
prov = {"run": run.name, "date": datetime.date.today().isoformat(),
        "config": "analysis/configs/riaf_m87.toml",
        "code": {"thermal-riaf": git("-C", "codes/thermal-riaf", "rev-parse", "HEAD"),
                 "m87-riaf-jet-repro": git("rev-parse", "HEAD")},
        "script": "analysis/scripts/riaf_derived.py",
        "frac_nuLnu_r_lt_7.63Rg": {**{k: round(v, 3) for k, v in out.items()}, "X-ray 2-10 keV": round(xr, 3)},
        "frac_nuLnu_r_lt_Rstar_over_cosi": frac_ap,
        "Mdot_21RS_Msun_yr": mdot[21]/Msun*yr, "Mdot_BH_1.1RS_g_s": mdot[R_IN], "Mdot_BH_1.1RS_Msun_yr": mdot[R_IN]/Msun*yr,
        "L_acc_erg_s": dg["inner_accretion_power_erg_s"], "L_rad_erg_s": dg["total_luminosity_erg_s"],
        "eta_j_per_jet_for_Lj_1.2e43": 1.2e43/dg["inner_accretion_power_erg_s"]}
json.dump(prov, open(run/"provenance.json", "w"), indent=2); print("wrote", run/"provenance.json")
