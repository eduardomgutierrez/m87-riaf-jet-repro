"""Compare the regenerated results with the values of the paper (reference/expected.json).

Every number in each section of expected.json (chi^2 per band, RIAF derived numbers, jet derived quantities, Ro-compatible
variants, synthetic-map summary, inverse-Compton estimate) and every spectrum of the parameter scans
(reference/scans.npz) is compared with the same key of the corresponding file in
analysis/runs/, with a relative tolerance (default 1e-3). The sha256 of the model flux files is also compared: an
identical hash means a bit-for-bit reproduction, but a different compiler or CPU can change the last digits, so a hash
mismatch is reported as a note only.

    python analysis/scripts/check_results.py [--rtol 1e-3]
Exit status 1 if any number is outside the tolerance or any file is missing.
"""
import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from m87.paths import ROOT, RUNS  # noqa: E402

EXPECTED = ROOT / "reference" / "expected.json"


def leaves(x, prefix=""):
    """Yield (path, value) for every number in a nested dict/list."""
    if isinstance(x, dict):
        for k, v in x.items():
            yield from leaves(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(x, list):
        for i, v in enumerate(x):
            yield from leaves(v, f"{prefix}[{i}]")
    elif isinstance(x, (int, float)) and not isinstance(x, bool):
        yield prefix, float(x)


def compare(expected, actual, rtol):
    got = dict(leaves(actual))
    bad = []
    n = 0
    for path, ref in leaves(expected):
        n += 1
        new = got.get(path)
        if new is None:
            bad.append(f"{path}: missing (expected {ref:.6g})")
        elif not math.isclose(new, ref, rel_tol=rtol, abs_tol=1e-12 * max(1.0, abs(ref))):
            bad.append(f"{path}: {new:.6g} (expected {ref:.6g}, {new / ref - 1:+.2e})" if ref
                       else f"{path}: {new:.6g} (expected 0)")
    return n, bad


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rtol", type=float, default=1e-3)
    rtol = ap.parse_args().rtol
    exp = json.load(open(EXPECTED))
    failed = False

    for name, sec in exp["sections"].items():
        path = RUNS / sec["file"]
        if not path.exists():
            print(f"FAIL  {name}: {path.relative_to(ROOT)} not found")
            failed = True
            continue
        actual = json.load(open(path))
        if sec["key"]:
            actual = actual[sec["key"]]
        n, bad = compare(sec["values"], actual, rtol)
        print(f"{'PASS' if not bad else 'FAIL'}  {name}: {n - len(bad)}/{n} values within rtol = {rtol:g}"
              f"  ({path.relative_to(ROOT)})")
        for b in bad[:20]:
            print("        " + b)
        failed |= bool(bad)

    scans = RUNS / "scans" / "scans.npz"
    if not scans.exists():
        print(f"FAIL  scans: {scans.relative_to(ROOT)} not found")
        failed = True
    else:
        ref, new = np.load(ROOT / "reference" / "scans.npz"), np.load(scans)
        worst, bad = 0.0, []
        for k in ref.files:
            if k == "base":
                continue
            a, b = ref[k], new[k] if k in new.files else None
            if b is None or a.shape != b.shape:
                bad.append(k)
                continue
            m = np.abs(a) > 1e-6 * np.abs(a).max()  # skip numerically empty tails
            worst = max(worst, float(np.max(np.abs(b[m] / a[m] - 1))) if m.any() else 0.0)
        ok = not bad and worst <= rtol
        print(f"{'PASS' if ok else 'FAIL'}  scans: {len(ref.files) - 1} arrays, max relative difference {worst:.2e}"
              + (f"; missing or reshaped: {bad}" if bad else "") + f"  ({scans.relative_to(ROOT)})")
        failed |= not ok

    same = 0
    for rel, ref in exp["sha256"].items():
        path = RUNS / rel
        if not path.exists():
            print(f"FAIL  {rel} not found")
            failed = True
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() == ref:
            same += 1
        else:
            print(f"note  {rel}: not bit-identical to the paper run (expected with another compiler or CPU)")
    print(f"{'info'}  {same}/{len(exp['sha256'])} model output files bit-identical to the paper runs")

    print("\nAll checks passed." if not failed else "\nSome checks FAILED.")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
