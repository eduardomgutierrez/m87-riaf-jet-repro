"""Run the paper models and write provenance.json in each run directory.

    python analysis/scripts/run_models.py                 # both jet models (jet_riaf, jet_only), in parallel
    python analysis/scripts/run_models.py jet_riaf        # one model
    python analysis/scripts/run_models.py --riaf          # also re-run the RIAF (thermal-riaf; slow)
    python analysis/scripts/run_models.py --riaf-only     # only the RIAF

Jet models: codes/extended-jet/Code/run.x <analysis/configs/NAME.nml>, run from the repo root (outdir in the
namelist is relative to it). The RIAF output analysis/runs/riaf_m87 is reused unless --riaf is given.
"""
import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m87 import provenance  # noqa: E402
from m87.paths import CONFIGS, JETCODE, RIAFCODE, ROOT, RUNS  # noqa: E402

JET_MODELS = ["jet_riaf", "jet_only"]


def run_jet(name):
    nml = CONFIGS / f"{name}.nml"
    params = provenance.read_namelist(nml)
    outdir = ROOT / params["run"]["outdir"]
    outdir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "OMP_NUM_THREADS": "1"}
    t0 = time.time()
    with open(outdir / "run.log", "w") as log:
        subprocess.run([str(JETCODE / "run.x"), str(nml.relative_to(ROOT))], cwd=ROOT, env=env,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    shutil.copy(nml, outdir / nml.name)
    provenance.write(outdir / "provenance.json", run=name,
                     code={"extended-jet": provenance.git_state(JETCODE.parent)},
                     config=str(nml.relative_to(ROOT)), config_sha256=provenance.sha256(nml),
                     command=f"OMP_NUM_THREADS=1 codes/extended-jet/Code/run.x {nml.relative_to(ROOT)}",
                     script="analysis/scripts/run_models.py", runtime_s=round(time.time() - t0, 1),
                     parameters=params)
    return name, time.time() - t0


def run_riaf():
    toml = CONFIGS / "riaf_m87.toml"
    subprocess.run([sys.executable, "riaf_pipeline.py", os.path.relpath(toml, RIAFCODE)], cwd=RIAFCODE, check=True)
    # riaf_validate.py computes the derived numbers and writes provenance.json for the RIAF run
    subprocess.run([sys.executable, str(ROOT / "analysis/scripts/riaf_validate.py"), "riaf_m87"], cwd=ROOT, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", default=JET_MODELS, help="jet namelists in analysis/configs (no .nml)")
    ap.add_argument("--riaf", action="store_true", help="re-run the RIAF model too")
    ap.add_argument("--riaf-only", action="store_true", help="run only the RIAF model")
    args = ap.parse_args()
    if args.riaf_only:
        run_riaf()
        return
    if not (JETCODE / "run.x").exists():
        sys.exit("run.x not built: run `make build` first")
    if args.riaf:
        run_riaf()
    with ThreadPoolExecutor(len(args.models)) as pool:
        for name, dt in pool.map(run_jet, args.models):
            print(f"{name}: done in {dt:.0f} s -> {RUNS / name}")


if __name__ == "__main__":
    main()
