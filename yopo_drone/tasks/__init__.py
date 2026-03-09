"""Task registrations for YOPO drone experiments."""

from __future__ import annotations

import gymnasium as gym

_TASK_ID = "point_ctrl_single_ego"


def _register_task_alias() -> None:
    if _TASK_ID in gym.registry:
        return
    gym.register(
        id=_TASK_ID,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": "yopo_drone.tasks.no_obstacle_env_cfg:NoObstacleEnvCfg",
            "rsl_rl_cfg_entry_point": (
                "isaaclab_tasks.manager_based.drone_arl.track_position_state_based."
                "config.arl_robot_1.agents.rsl_rl_ppo_cfg:TrackPositionNoObstaclesEnvPPORunnerCfg"
            ),
        },
    )


_register_task_alias()

__all__ = ["_TASK_ID"]
