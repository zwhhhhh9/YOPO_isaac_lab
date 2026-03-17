#!/usr/bin/env python3
"""Launch GUI evaluation for a trained YOPO policy in the random-forest map."""

from __future__ import annotations

import argparse
import importlib.util
import sys
import time
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


# Last verified checkpoint that reaches the sampled mission goal with the
# current docking / hover-hold runtime and full-yaw training distribution.
DEFAULT_CHECKPOINT = (
    _repo_root()
    / "yopo_drone"
    / "network"
    / "checkpoint"
    / "checkpoint_20260317_192343"
    / "epoch50_full_yaw_best.pth"
)
DEFAULT_DATASET_DIR = _repo_root() / "yopo_drone" / "network" / "data_train" / "20260317_022500_recollect_goalfix_1map"


def _load_editor_scene_eval_ego():
    module_path = _repo_root() / "yopo_drone" / "tasks" / "editor_scene_eval_ego.py"
    spec = importlib.util.spec_from_file_location("yopo_editor_scene_eval_ego", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load editor_scene_eval_ego from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _parse_args(argv: list[str]) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Launch YOPO policy GUI evaluation with a trained checkpoint.",
        add_help=False,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="YOPO checkpoint to load.",
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default=str(DEFAULT_DATASET_DIR),
        help="Dataset directory used for YOPO config recovery and random goal sampling.",
    )
    parser.add_argument(
        "--goal_seed",
        type=int,
        default=-1,
        help="Random seed for goal sampling. Use a negative value to sample a fresh seed from wall-clock time.",
    )
    parser.add_argument(
        "--min_goal_distance",
        type=float,
        default=6.0,
        help="Minimum distance in meters between the startup position and the sampled mission goal.",
    )
    parser.add_argument(
        "--goal_xyz",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Optional fixed mission goal in world coordinates. When omitted, a random goal is sampled from the dataset poses.",
    )
    parser.add_argument(
        "--initial_hover_s",
        type=float,
        default=3.0,
        help="Seconds to hover at the startup point before the YOPO policy begins replanning.",
    )
    parser.add_argument(
        "--startup_hover_settle_steps",
        type=int,
        default=200,
        help="Closed-loop startup settle steps run before the mission timer starts.",
    )
    parser.add_argument(
        "--velocity",
        type=float,
        default=2.0,
        help="Original-YOPO testing velocity passed to the runtime config. Default keeps evaluation capped at 2.0 m/s.",
    )
    parser.add_argument(
        "--compact_backbone",
        action="store_true",
        default=False,
        help="Load the compact YOPO backbone instead of the default network.",
    )
    parser.add_argument("-h", "--help", action="store_true", default=False, help="Show this help message and exit.")
    return parser.parse_known_args(argv)


def _has_flag(args: list[str], flag: str) -> bool:
    return any(arg == flag or arg.startswith(f"{flag}=") for arg in args)


def main() -> int:
    args, forwarded_args = _parse_args(list(sys.argv[1:]))
    if args.help:
        print("Usage: yopo_policy_gui.py [launcher options] [eval_ego args...]")
        print("")
        print("Launcher options:")
        print("  --checkpoint PATH")
        print("  --dataset_dir PATH")
        print("  --goal_seed INT")
        print("  --min_goal_distance METERS")
        print("  --goal_xyz X Y Z")
        print("  --initial_hover_s SECONDS")
        print("  --startup_hover_settle_steps STEPS")
        print("  --velocity MPS")
        print("  --compact_backbone")
        print("")
        print("Any remaining arguments are forwarded to editor_scene_eval_ego.py / eval_ego.py.")
        return 0

    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not checkpoint_path.exists():
        raise SystemExit(f"YOPO checkpoint not found: {checkpoint_path}")
    if not dataset_dir.exists():
        raise SystemExit(f"YOPO dataset_dir not found: {dataset_dir}")
    if args.min_goal_distance < 0.0:
        raise SystemExit("--min_goal_distance must be >= 0.")
    if args.initial_hover_s < 0.0:
        raise SystemExit("--initial_hover_s must be >= 0.")
    if args.startup_hover_settle_steps < 0:
        raise SystemExit("--startup_hover_settle_steps must be >= 0.")
    if args.velocity <= 0.0:
        raise SystemExit("--velocity must be > 0.")

    repo_root = _repo_root()
    repo_root_str = str(repo_root)
    if repo_root_str not in sys.path:
        sys.path.insert(0, repo_root_str)

    editor_scene_eval_ego = _load_editor_scene_eval_ego()

    goal_seed = int(time.time()) if int(args.goal_seed) < 0 else int(args.goal_seed)
    launch_args = [
        "--yopo_policy",
        "--yopo_policy_startup_settle_steps",
        str(int(args.startup_hover_settle_steps)),
        "--yopo_policy_checkpoint",
        str(checkpoint_path),
        "--yopo_policy_dataset_dir",
        str(dataset_dir),
        "--yopo_policy_initial_hover_s",
        str(float(args.initial_hover_s)),
        "--yopo_policy_velocity",
        str(float(args.velocity)),
        "--yopo_policy_min_goal_distance",
        str(float(args.min_goal_distance)),
        "--yopo_policy_random_goal_seed",
        str(goal_seed),
    ]
    if args.goal_xyz is not None:
        launch_args.extend(["--yopo_policy_goal_xyz", *(str(float(value)) for value in args.goal_xyz)])
    # Match the original YOPO forest scale during policy evaluation unless the user overrides it.
    if not _has_flag(forwarded_args, "--random-forest-size-x"):
        launch_args.extend(["--random-forest-size-x", "60.0"])
    if not _has_flag(forwarded_args, "--random-forest-size-y"):
        launch_args.extend(["--random-forest-size-y", "60.0"])
    if not _has_flag(forwarded_args, "--random-forest-tile-radius"):
        launch_args.extend(["--random-forest-tile-radius", "0"])
    if not _has_flag(forwarded_args, "--random-forest-clearance-radius"):
        launch_args.extend(["--random-forest-clearance-radius", "0.0"])
    if args.compact_backbone:
        launch_args.append("--yopo_policy_compact_backbone")
    sys.argv = [sys.argv[0], *launch_args, *forwarded_args]
    return editor_scene_eval_ego.main()


if __name__ == "__main__":
    raise SystemExit(main())
