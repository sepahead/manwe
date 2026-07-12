#!/usr/bin/env python3
"""Run Manwe's dependency-free current-worktree credential scan."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "src"))

from manwe.common.secret_scan import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(root=ROOT))
