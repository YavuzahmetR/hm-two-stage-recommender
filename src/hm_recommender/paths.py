"""Repository paths and Unicode-safe LightGBM model I/O."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data" / "raw" / "hm"


def load_booster(path):
    from lightgbm import Booster
    return Booster(model_str=Path(path).read_text(encoding="utf-8"))


def save_booster(booster, path, **kwargs):
    Path(path).write_text(booster.model_to_string(**kwargs), encoding="utf-8")
