#!/usr/bin/env python3
"""Execute a target Python script under Isaac Sim's python runtime."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path


def _resolve_script(user_arg: str) -> Path:
    candidate = Path(user_arg)
    if candidate.is_absolute():
        return candidate
    return (Path(__file__).resolve().parents[1] / candidate).resolve()


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python yopo_drone/run.py <script.py> [args...]")
        return 1

    script_path = _resolve_script(sys.argv[1])
    if not script_path.exists():
        print(f"Error: script not found: {script_path}")
        return 2

    sys.argv = [str(script_path), *sys.argv[2:]]
    runpy.run_path(str(script_path), run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
