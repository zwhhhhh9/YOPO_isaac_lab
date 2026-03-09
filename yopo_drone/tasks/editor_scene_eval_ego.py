#!/usr/bin/env python3
"""Launch eval_ego in a stage initialized with drone_env_editor scene helpers.

This entry point keeps the planning/control stack inside a scene that first gets
prepared with the same world primitives as `drone_env_editor.py`.
"""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from yopo_drone.utils import eval_ego

    forwarded_args = list(sys.argv[1:])
    if "--disable_env_editor_scene_init" not in forwarded_args:
        forwarded_args = ["--env_editor_world_path", "/World", *forwarded_args]

    sys.argv = [sys.argv[0], *forwarded_args]
    eval_ego.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
