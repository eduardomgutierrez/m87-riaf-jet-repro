"""Git hashes, namelist parsing and provenance.json writing."""
import datetime
import hashlib
import json
import re
import subprocess

from .paths import ROOT


def git(*args, cwd=ROOT):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True).stdout.strip()


def git_state(path=ROOT):
    """Commit hash and whether the work tree has uncommitted changes to tracked files."""
    return {"commit": git("rev-parse", "HEAD", cwd=path),
            "dirty": bool(git("status", "--porcelain", "--untracked-files=no", cwd=path))}


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _value(s):
    s = s.strip()
    if s.lower() in (".true.", "t"):
        return True
    if s.lower() in (".false.", "f"):
        return False
    if s[:1] in "'\"":
        return s[1:-1]
    if re.fullmatch(r"[+-]?\d+", s):
        return int(s)
    try:
        return float(s.lower().replace("d", "e"))
    except ValueError:
        return s


def read_namelist(path):
    """Minimal Fortran namelist reader: {group: {key: value}}; arrays become lists."""
    groups, cur = {}, None
    text = re.sub(r"!.*", "", open(path).read())
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("&"):
            cur = groups.setdefault(line[1:].strip().lower(), {})
            continue
        if line == "/" or not line or cur is None:
            continue
        # split "a = 1, b = 2, 3" into key/value pairs
        for key, val in re.findall(r"(\w+)\s*=\s*(.*?)(?=,\s*\w+\s*=|$)", line):
            vals = [_value(v) for v in val.rstrip(",").split(",") if v.strip()]
            cur[key.lower()] = vals[0] if len(vals) == 1 else vals
    return groups


def write(path, code=None, **fields):
    """Write provenance.json; `code` adds {name: git_state(...)} entries next to the state of this repository."""
    prov = {"date": datetime.datetime.now().isoformat(timespec="seconds"),
            "code": {"m87-riaf-jet-repro": git_state(), **(code or {})}, **fields}
    with open(path, "w") as f:
        json.dump(prov, f, indent=2)
    return prov
