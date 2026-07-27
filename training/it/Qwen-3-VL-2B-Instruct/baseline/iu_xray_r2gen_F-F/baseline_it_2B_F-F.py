#!/usr/bin/env python3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from shield.training import main_baseline

OVERRIDES: dict = {}

if __name__ == "__main__":
    raise SystemExit(main_baseline(__file__, OVERRIDES))
