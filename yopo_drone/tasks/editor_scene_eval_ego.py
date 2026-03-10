#!/usr/bin/env python3
"""Launch eval_ego in a stage initialized with drone_env_editor scene helpers.

This entry point keeps the planning/control stack inside a scene that first gets
prepared with the same world primitives as `drone_env_editor.py`.
Use `--hover` for the standard hover/evaluation task or `--target_goal` for a
simple autonomous point-to-point mission.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _parse_editor_scene_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch editor_scene_eval_ego tasks.",
        add_help=False,
    )
    parser.add_argument(
        "--hover",
        action="store_true",
        default=False,
        help="Launch the hover/evaluation task.",
    )
    parser.add_argument(
        "--target_goal",
        action="store_true",
        default=False,
        help="Hover at the startup position, then fly to a target goal and hold there.",
    )
    parser.add_argument(
        "--target_goal_pos",
        type=float,
        nargs=3,
        default=(10.0, 0.0, 1.0),
        metavar=("X", "Y", "Z"),
        help="World-frame target position used by --target_goal.",
    )
    parser.add_argument(
        "--target_goal_hover_s",
        type=float,
        default=5.0,
        help="Seconds to hover at the startup position before starting --target_goal.",
    )
    parser.add_argument(
        "--target_goal_max_speed",
        type=float,
        default=1.0,
        help="Maximum flight speed in m/s used by --target_goal.",
    )
    parser.add_argument(
        "--target_goal_startup_settle_steps",
        type=int,
        default=200,
        help="Closed-loop startup hover settle steps run before --target_goal telemetry and timing begin.",
    )
    parser.add_argument("-h", "--help", action="store_true", default=False, help="Show this help message and exit.")
    return parser.parse_known_args(argv)


def _has_forwarded_option(forwarded_args: list[str], *option_names: str) -> bool:
    option_set = set(option_names)
    return any(arg in option_set for arg in forwarded_args)


def _default_target_goal_telemetry_filename() -> str:
    return f"target_goal_{time.strftime('%Y%m%d_%H%M%S')}.csv"


def main() -> int:
    task_args, forwarded_args = _parse_editor_scene_args(list(sys.argv[1:]))
    if task_args.help:
        print("Usage: editor_scene_eval_ego.py [--hover | --target_goal] [eval_ego args...]")
        print("")
        print("Task suffixes:")
        print("  --hover    Launch the hover/evaluation task.")
        print("  --target_goal")
        print("             Hover at startup, fly to a target goal, then hold there.")
        print("")
        print("Target-goal options:")
        print("  --target_goal_pos X Y Z")
        print("  --target_goal_hover_s SECONDS")
        print("  --target_goal_max_speed MPS")
        print("  --target_goal_startup_settle_steps STEPS")
        return 0
    if task_args.hover and task_args.target_goal:
        raise SystemExit("Only one task suffix can be active: choose either --hover or --target_goal.")
    if task_args.target_goal_hover_s < 0.0:
        raise SystemExit("--target_goal_hover_s must be >= 0.")
    if task_args.target_goal_max_speed <= 0.0:
        raise SystemExit("--target_goal_max_speed must be > 0.")
    if task_args.target_goal_startup_settle_steps < 0:
        raise SystemExit("--target_goal_startup_settle_steps must be >= 0.")
    if not task_args.hover and not task_args.target_goal:
        # Keep the historical no-suffix invocation working for now.
        task_args.hover = True

    repo_root = Path(__file__).resolve().parents[2]
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    from yopo_drone.utils import eval_ego

    if task_args.target_goal:
        forwarded_args = [
            "--startup_hover_settle_steps",
            str(task_args.target_goal_startup_settle_steps),
            "--auto_target_goal",
            *(str(value) for value in task_args.target_goal_pos),
            "--auto_target_goal_initial_hover_s",
            str(task_args.target_goal_hover_s),
            "--auto_target_goal_max_speed",
            str(task_args.target_goal_max_speed),
            *forwarded_args,
        ]
        if not _has_forwarded_option(forwarded_args, "--telemetry_log_path", "--telemetry-log-path"):
            forwarded_args = [
                "--telemetry_log_path",
                _default_target_goal_telemetry_filename(),
                *forwarded_args,
            ]

    if "--disable_env_editor_scene_init" not in forwarded_args:
        forwarded_args = ["--env_editor_world_path", "/World", *forwarded_args]

    sys.argv = [sys.argv[0], *forwarded_args]
    eval_ego.main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
