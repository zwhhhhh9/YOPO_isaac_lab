from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab_contrib.actuators import ThrusterCfg
from isaaclab_contrib.assets import MultirotorCfg
from isaaclab_tasks.manager_based.drone_arl.track_position_state_based.config.arl_robot_1.track_position_state_based_env_cfg import (
    TrackPositionNoObstaclesEnvCfg,
)

from yopo_drone.utils.robot_model import DEFAULT_ROBOT_URDF, load_px4_robot_model, resolve_project_path


_ROBOT_MODEL = load_px4_robot_model()
_THRUSTER_NAMES = list(_ROBOT_MODEL["thruster_names"])
_ROTOR_DIRECTIONS = list(_ROBOT_MODEL["rotor_directions"])
_ALLOCATION_MATRIX = [list(row) for row in _ROBOT_MODEL["allocation_matrix"]]
_MOTOR_THRUST_MAX = float(_ROBOT_MODEL["motor_thrust_max"])
_HOVER_RPS = 200.0
_HOVER_THRUST_PER_MOTOR = float(_ROBOT_MODEL["mass"]) * 9.81 / max(len(_THRUSTER_NAMES), 1)
_HOVER_THRUST_CONST = _HOVER_THRUST_PER_MOTOR / (_HOVER_RPS * _HOVER_RPS)


YOPO_ROBOT_THRUSTER_CFG = ThrusterCfg(
    dt=0.01,
    thrust_range=(0.0, _MOTOR_THRUST_MAX),
    thrust_const_range=(_HOVER_THRUST_CONST, _HOVER_THRUST_CONST),
    tau_inc_range=(0.05, 0.08),
    tau_dec_range=(0.005, 0.005),
    torque_to_thrust_ratio=float(_ROBOT_MODEL["kappa"]),
    thruster_names_expr=_THRUSTER_NAMES,
)


YOPO_ROBOT_CFG = MultirotorCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(resolve_project_path(DEFAULT_ROBOT_URDF)),
        fix_base=False,
        merge_fixed_joints=False,
        self_collision=False,
        collision_from_visuals=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
            target_type="none",
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
            rest_offset=0.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True,
            enabled_self_collisions=False,
            fix_root_link=False,
        ),
        activate_contact_sensors=True,
    ),
    init_state=MultirotorCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        lin_vel=(0.0, 0.0, 0.0),
        ang_vel=(0.0, 0.0, 0.0),
        rot=(1.0, 0.0, 0.0, 0.0),
        rps={name: _HOVER_RPS for name in _THRUSTER_NAMES},
    ),
    actuators={"thrusters": YOPO_ROBOT_THRUSTER_CFG},
    rotor_directions=_ROTOR_DIRECTIONS,
    allocation_matrix=_ALLOCATION_MATRIX,
)


@configclass
class NoObstacleEnvCfg(TrackPositionNoObstaclesEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = YOPO_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["thrusters"].dt = self.sim.dt
        self.actions.thrust_command.clip = {".*": (0.0, _MOTOR_THRUST_MAX)}
        self.actions.thrust_command.preserve_order = True


@configclass
class NoObstacleEnvCfg_PLAY(NoObstacleEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
