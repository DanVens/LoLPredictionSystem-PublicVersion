#!/usr/bin/env python3
"""
Compatibility entrypoint for the no-prefix logistic-regression trainer.
"""

from __future__ import annotations

from pathlib import Path
import sys


EXPORT_ROOT = Path(__file__).resolve().parents[1]
SHARED_DIR = EXPORT_ROOT / "shared"
if str(SHARED_DIR) not in sys.path:
    sys.path.insert(0, str(SHARED_DIR))

from train_logistic_regression_no_prefix import main


if __name__ == "__main__":
    raise SystemExit(main())
