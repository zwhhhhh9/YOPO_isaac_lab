from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

DEFAULT_ROBOT_URDF = "assets/robot./robot.urdf"
DEFAULT_PX4_PARAMS = "assets/robot./px4_params.json"


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (project_root() / path).resolve()


def load_px4_robot_model(
    urdf_path: str | Path | None = None,
    params_path: str | Path | None = None,
) -> dict[str, Any]:
    resolved_urdf = resolve_project_path(str(urdf_path or DEFAULT_ROBOT_URDF))
    resolved_params = resolve_project_path(str(params_path or DEFAULT_PX4_PARAMS))

    if not resolved_urdf.is_file():
        raise FileNotFoundError(f"URDF model not found: {resolved_urdf}")

    urdf_data = _parse_urdf_model(resolved_urdf)
    params_data = _load_params_file(resolved_params) if resolved_params.is_file() else {}

    kappa = float(params_data.get("kappa", 0.015))
    sources = [
        "mass<-urdf.total_mass",
        "inertia<-urdf.base_link.inertia",
        "arm_length<-urdf.motor_joint_radius_avg",
        "rotor_directions<-urdf.motor_axis_z",
        "thruster_names<-urdf.motor_child_links",
        "allocation_matrix<-urdf.motor_layout+kappa",
    ]
    if "motor_thrust_max" in params_data:
        sources.append("motor_thrust_max<-px4_params")
    if "kappa" in params_data:
        sources.append("kappa<-px4_params")

    model = {
        "urdf_path": str(resolved_urdf),
        "params_path": str(resolved_params) if resolved_params.is_file() else None,
        "mass": urdf_data["mass"],
        "inertia": urdf_data["inertia"],
        "base_com_offset": urdf_data.get("base_com_offset"),
        "arm_length": urdf_data["arm_length"],
        "motors": [dict(motor) for motor in urdf_data["motors"]],
        "thruster_names": [motor["child_link"] for motor in urdf_data["motors"]],
        "rotor_directions": [int(motor["rotor_direction"]) for motor in urdf_data["motors"]],
        "allocation_matrix": _build_allocation_matrix(urdf_data["motors"], kappa),
        "motor_thrust_max": float(params_data.get("motor_thrust_max", 4.5)),
        "kappa": kappa,
        "sources": sources,
    }

    passthrough_keys = (
        "ctrl_thrust_ratio",
        "pos_kp",
        "vel_kp",
        "attitude_fb_kp",
        "max_bodyrate_fb",
        "max_angle_deg",
        "min_collective_acc",
        "att_p_gain",
        "att_rate_limit",
        "att_yaw_weight",
        "rate_p_gain",
        "rate_i_gain",
        "rate_d_gain",
        "rate_k_gain",
        "rate_int_limit",
        "accel_filter_coef",
    )
    for key in passthrough_keys:
        if key in params_data:
            model[key] = params_data[key]

    return model


def _load_params_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"PX4 params file must contain a JSON object: {path}")
    return data


def _parse_urdf_model(path: Path) -> dict[str, Any]:
    root = ET.parse(path).getroot()

    links: dict[str, dict[str, Any]] = {}
    total_mass = 0.0
    for link in root.findall("link"):
        name = link.attrib.get("name", "")
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass_node = inertial.find("mass")
        inertia_node = inertial.find("inertia")
        origin_node = inertial.find("origin")
        if mass_node is None or inertia_node is None:
            continue
        mass_value = float(mass_node.attrib.get("value", "0"))
        total_mass += mass_value
        links[name] = {
            "mass": mass_value,
            "com_offset": list(_parse_xyz(origin_node.attrib.get("xyz", "0 0 0"))) if origin_node is not None else [0.0, 0.0, 0.0],
            "inertia": [
                float(inertia_node.attrib.get("ixx", "0")),
                float(inertia_node.attrib.get("iyy", "0")),
                float(inertia_node.attrib.get("izz", "0")),
            ],
        }

    if total_mass <= 0.0:
        raise ValueError(f"URDF does not contain valid inertial masses: {path}")

    base_link_name = "base_link" if "base_link" in links else next(iter(links))
    base_inertia = links[base_link_name]["inertia"]
    base_com_offset = [float(v) for v in links[base_link_name]["com_offset"]]

    motor_specs: list[dict[str, Any]] = []
    for joint in root.findall("joint"):
        parent = joint.find("parent")
        if parent is None or parent.attrib.get("link") != base_link_name:
            continue
        child = joint.find("child")
        origin = joint.find("origin")
        axis = joint.find("axis")
        if child is None or origin is None or axis is None:
            continue
        xyz = _parse_xyz(origin.attrib.get("xyz", "0 0 0"))
        axis_xyz = _parse_xyz(axis.attrib.get("xyz", "0 0 1"))
        if abs(axis_xyz[2]) <= 1e-6:
            raise ValueError(f"URDF motor joint '{joint.attrib.get('name', '')}' must define a +/-Z axis: {path}")
        motor_specs.append(
            {
                "joint_name": joint.attrib.get("name", ""),
                "child_link": child.attrib.get("link", ""),
                "position": [float(v) for v in xyz],
                "position_com": [
                    float(xyz[0] - base_com_offset[0]),
                    float(xyz[1] - base_com_offset[1]),
                    float(xyz[2] - base_com_offset[2]),
                ],
                "rotor_direction": 1 if axis_xyz[2] > 0.0 else -1,
            }
        )

    if not motor_specs:
        raise ValueError(f"URDF does not expose motor joint offsets under '{base_link_name}': {path}")

    ordered_motors = sorted(motor_specs, key=_controller_motor_order_key)
    _validate_quadrotor_rotor_directions(ordered_motors, path)
    motor_radii = [math.hypot(motor["position_com"][0], motor["position_com"][1]) for motor in ordered_motors]
    arm_length = float(sum(motor_radii) / len(motor_radii))

    return {
        "mass": float(total_mass),
        "inertia": [float(v) for v in base_inertia],
        "base_com_offset": base_com_offset,
        "arm_length": arm_length,
        "motors": ordered_motors,
    }


def _parse_xyz(value: str) -> tuple[float, float, float]:
    parts = value.split()
    padded = (parts + ["0", "0", "0"])[:3]
    return (float(padded[0]), float(padded[1]), float(padded[2]))


def _controller_motor_order_key(motor: dict[str, Any]) -> tuple[int, float]:
    position = motor.get("position_com", motor["position"])
    x_pos, y_pos = float(position[0]), float(position[1])
    if x_pos >= 0.0 and y_pos >= 0.0:
        quadrant = 0
    elif x_pos >= 0.0 and y_pos < 0.0:
        quadrant = 1
    elif x_pos < 0.0 and y_pos < 0.0:
        quadrant = 2
    else:
        quadrant = 3
    return quadrant, math.hypot(x_pos, y_pos)


def _build_allocation_matrix(motors: list[dict[str, Any]], kappa: float) -> list[list[float]]:
    positions = [motor.get("position_com", motor["position"]) for motor in motors]
    return [
        [0.0 for _ in motors],
        [0.0 for _ in motors],
        [1.0 for _ in motors],
        [float(position[1]) for position in positions],
        [-float(position[0]) for position in positions],
        [float(kappa) * int(motor["rotor_direction"]) for motor in motors],
    ]


def _validate_quadrotor_rotor_directions(motors: list[dict[str, Any]], path: Path) -> None:
    if len(motors) != 4:
        return
    directions = [int(motor["rotor_direction"]) for motor in motors]
    diagonal_match = directions[0] == directions[2] and directions[1] == directions[3]
    adjacent_opposite = directions[0] == -directions[1]
    if diagonal_match and adjacent_opposite:
        return
    raise ValueError(
        "Quadrotor URDF motor directions must follow a diagonal pairing pattern "
        f"(got {directions} from {path})."
    )
