"""Build the per-instrument SED files data/obs_data/<INSTR>.dat from the public EHT MWL 2017 release.

Source: data/eht_mwl2017/m87_2017_mwl_spectrum.csv (EHT MWL Science WG 2021, ApJL 911, L11; DOI 10.25739/mhh2-cw46).
Each row is converted to nu [Hz], nuFnu [erg s^-1 cm^-2] and its error; rows with zero error are upper limits and get
err = -1. The instrument is taken from the dataset-index column (the VLBA rows sit under the KVN heading in the CSV).
Chandra and NuSTAR share one dataset and are split at 1.2e17 Hz; Swift/UVOT and Swift/XRT go to Swift.dat.

CORRECTIONS lists every value in data/obs_data that differs from the plain conversion, with its reason. Entries marked
"to check" differ from the release without a recorded reason; they are kept as used in the paper.

Usage:  python analysis/scripts/make_obs_data.py          # compare the existing files with the rebuilt ones (default)
        python analysis/scripts/make_obs_data.py --write  # overwrite data/obs_data/*.dat with the rebuilt files
"""
import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RELEASE = ROOT / "data/eht_mwl2017/m87_2017_mwl_spectrum.csv"
OUT = ROOT / "data/obs_data"
JY = 1e-23
INDEX = {1: "EVN", 2: "HSA", 3: "VERA", 4: "EAVN", 5: "KVN", 6: "VLBA", 7: "GMVA", 8: "SMA", 9: "ALMA", 10: "EHT",
         11: "HST", 12: "Swift", 13: "Swift", 14: "Chandra+NuSTAR", 15: "Fermi", 16: "HESS", 17: "MAGIC", 18: "VERITAS"}

# (instrument, approximate nu [Hz]) -> {"nu", "nufnu", "err"} overrides, with the reason.
CORRECTIONS = {
    ("KVN", 2.17e10): dict(nu=21.7e9, reason="nu from EHT MWL 2021 Table A8"),
    ("KVN", 4.34e10): dict(nu=43.4e9, reason="nu from Table A8"),
    ("KVN", 8.68e10): dict(nu=86.8e9, reason="nu from Table A8"),
    ("VLBA", 2.38e10): dict(nu=24e9, reason="nominal frequency; flux 1.12 Jy from Table A1 and the release"),
    ("VLBA", 4.31e10): dict(nu=43e9, reason="nominal frequency; flux 1.09 Jy from Table A1 and the release"),
    ("GMVA", 8.6e10): dict(err_frac=0.30, reason="30 per cent error budget (EHT MWL 2021 description and Table A8)"),
    ("NuSTAR", 6.13e17): dict(nufnu=9.92e-13, reason="to check: release gives 9.19e-13"),
    ("NuSTAR", 1.18e19): dict(err=1.18e-13, reason="to check: release error 1.18e-10 is 150x the flux (release typo?)"),
    ("MAGIC", 3.82e25): dict(nufnu=1.4e-13, err=8.2e-14, reason="to check: release gives 1.41e-12 (x10); gamma rays not fitted"),
    ("Fermi", 7.65e25): dict(nufnu=6.26e-11, reason="to check: release gives 5.26e-11 (upper limit); gamma rays not fitted"),
}


def convert():
    """{instrument: [[nu, nuFnu, err], ...]} from the release, before corrections."""
    out = {}
    for line in open(RELEASE):
        if not line[:1].isdigit():
            continue
        _, nu, _, f, e, idx = (float(x) for x in line.strip().split(","))
        name = INDEX[int(idx)]
        if name == "Chandra+NuSTAR":
            name = "Chandra" if nu < 1.2e17 else "NuSTAR"
        out.setdefault(name, []).append([nu, nu * f * JY, nu * e * JY if e > 0 else -1.0])
    return out


def corrected():
    data = convert()
    for (name, nu0), c in CORRECTIONS.items():
        rows = data[name]
        k = int(np.argmin([abs(np.log(r[0] / nu0)) for r in rows]))
        r = rows[k]
        if "err_frac" in c:
            r[2] = c["err_frac"] * r[1]
        for key, col in (("nu", 0), ("nufnu", 1), ("err", 2)):
            if key in c:
                r[col] = c[key]
    return data


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="overwrite data/obs_data/*.dat")
    args = ap.parse_args()
    data = corrected()
    bad = 0
    for name, rows in sorted(data.items()):
        rows = np.array(sorted(rows))
        path = OUT / f"{name}.dat"
        if args.write:
            with open(path, "w") as f:
                f.write("freq        nufnu       err\n")
                for nu, y, e in rows:
                    f.write(f"{nu:<11.4g} {y:<11.4g} {e:.3g}\n")
            continue
        old = np.loadtxt(path, skiprows=1, ndmin=2)
        old = old[np.argsort(old[:, 0])]
        same = old.shape == rows.shape and np.allclose(old, rows, rtol=0.03)
        print(f"{name:8s} {len(rows):2d} pts  {'OK' if same else 'DIFFERS'}")
        bad += not same
    if not args.write:
        todo = [f"  {n} {nu:.3g} Hz: {c['reason']}" for (n, nu), c in CORRECTIONS.items() if c["reason"].startswith("to check")]
        print("\nValues that differ from the release without a recorded reason:\n" + "\n".join(todo))
        sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
