"""Repository paths (all derived from this file's location; no absolute paths)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = ROOT / "analysis" / "configs"
RUNS = ROOT / "analysis" / "runs"
DATA = ROOT / "data" / "obs_data"
FIGURES = ROOT / "output" / "figures"
TABLES = ROOT / "output" / "tables"
JETCODE = ROOT / "codes" / "extended-jet" / "Code"
RIAFCODE = ROOT / "codes" / "thermal-riaf"
