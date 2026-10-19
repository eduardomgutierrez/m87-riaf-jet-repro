"""Ro+2023-compatible variants of the Jet+RIAF model (Results, 'Magnetic field'; Discussion, jet power).

The variants ro_B and ro_C (analysis/configs/ro_B.nml, ro_C.nml) differ from jet_riaf only in L_j, epsilon_e (fnT) and p:
a larger jet power raises B' (B' ~ L_j^1/2 for sigma' >> 1) and a lower epsilon_e keeps the SED. For each model this
script computes B' at the distances of Ro et al. (2023), the jet efficiency per jet, and the chi^2 over the same points,
error floor and effective apertures as Table chi2 (m87.chi2), and writes analysis/runs/ro_variants.json.

Run the models first:  python analysis/scripts/run_models.py ro_B ro_C   (jet_riaf and jet_only from `make models`)
Usage:                 python analysis/scripts/ro_variants.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from m87 import chi2, provenance  # noqa: E402
from m87.data import load_obs  # noqa: E402
from m87.models import load_jet_model  # noqa: E402
from m87.paths import RUNS  # noqa: E402
import derived_quantities as dq  # noqa: E402

RUNS_JET = {"A (paper, Jet+RIAF)": "jet_riaf", "B": "ro_B", "C": "ro_C"}
FLOOR = 0.1


def main():
    obs = load_obs()
    L_acc = json.load(open(RUNS / "riaf_m87" / "provenance.json"))["L_acc_erg_s"]
    total = {"Total": lambda p: p.band in ("radio", "optuv", "xray")}
    jo = chi2.summary(chi2.evaluate(load_jet_model("jet_only", None), obs, FLOOR), total)["Total"]
    out = {"floor": FLOOR, "chi2_jet_only": {"N": jo[0], "chi2": jo[1]}, "models": {}}
    for label, run in RUNS_JET.items():
        pars = provenance.read_namelist(RUNS / run / f"{run}.nml")["model"]
        n, c = chi2.summary(chi2.evaluate(load_jet_model(run, "riaf_m87"), obs, FLOOR), total)["Total"]
        B = dq.analyse(run)["B_at_Ro23"]
        out["models"][label] = {"run": run, "Lj": pars["lj"], "fnT": pars["fnt"], "p": pars["p"],
                                "eta_j_per_jet": pars["lj"] / L_acc, "N": n, "chi2": c, "B_at_Ro23_G": B}
        print(f"{label:22s} Lj={pars['lj']:.2g}  eps_e={pars['fnt']:.4g}  p={pars['p']}  eta_j={pars['lj'] / L_acc:.2f}/jet  "
              f"chi2={c:.1f} (N={n})  B'=" + ", ".join(f"{v:.3f} G @ {k}" for k, v in B.items()))
    print(f"{'Jet-only':22s} chi2={jo[1]:.1f} (N={jo[0]})")
    path = RUNS / "ro_variants.json"
    provenance.write(path, script="analysis/scripts/ro_variants.py", **out)
    print("wrote", path)


if __name__ == "__main__":
    main()
