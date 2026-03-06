#!/usr/bin/env python3
"""Edit Isaac Lab drone simulation scenes from the command line.

Example:
    # Preview directly in Isaac Lab GUI (default keeps window open):
    python yopo_drone/run.py yopo_drone/env/drone_env_editor.py \
        --ground-size 4000 4000 \
        --sun-intensity 120000

    # Headless smoke test (no GUI):
    python yopo_drone/run.py yopo_drone/env/drone_env_editor.py \
        --headless --close-after-build
"""

from __future__ import annotations

import argparse
import contextlib
import math
import sys
import time
from pathlib import Path
from typing import Any, Iterable

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim  # noqa: F401

try:
    from isaaclab.app import AppLauncher
except ImportError:
    AppLauncher = None

try:
    from isaacsim import SimulationApp
except ImportError as exc:
    raise SystemExit(
        "This script requires Isaac Sim Python runtime (SimulationApp not found). "
        "Run with: /workspace/isaaclab/_isaac_sim/python.sh yopo_drone/run.py "
        "yopo_drone/env/drone_env_editor.py ..."
    ) from exc

Gf = None
Usd = None
UsdGeom = None
UsdLux = None
UsdPhysics = None


def _ensure_isaaclab_pythonpath() -> None:
    """Make Isaac Lab source packages importable when not installed into site-packages."""
    source_root = _project_root() / "source"
    candidates = (
        source_root / "isaaclab",
        source_root / "isaaclab_assets",
        source_root / "isaaclab_mimic",
        source_root / "isaaclab_rl",
        source_root / "isaaclab_tasks",
    )
    for package_root in candidates:
        package_root_str = str(package_root)
        if package_root.is_dir() and package_root_str not in sys.path:
            sys.path.insert(0, package_root_str)


def _ensure_pxr_imported() -> None:
    global Gf, Usd, UsdGeom, UsdLux, UsdPhysics
    if all(module is not None for module in (Gf, Usd, UsdGeom, UsdLux, UsdPhysics)):
        return
    try:
        from pxr import Gf as _Gf, Usd as _Usd, UsdGeom as _UsdGeom, UsdLux as _UsdLux, UsdPhysics as _UsdPhysics
    except ImportError as exc:
        raise SystemExit(
            "This script requires Isaac Sim / Isaac Lab Python runtime (pxr module not found). "
            "Rebuild the image with usd-core installed."
        ) from exc
    Gf, Usd, UsdGeom, UsdLux, UsdPhysics = _Gf, _Usd, _UsdGeom, _UsdLux, _UsdPhysics


def _get_or_create_op(xformable: UsdGeom.Xformable, op_type: UsdGeom.XformOp.Type) -> UsdGeom.XformOp:
    for op in xformable.GetOrderedXformOps():
        if op.GetOpType() == op_type and not op.IsInverseOp():
            return op
    if op_type == UsdGeom.XformOp.TypeTranslate:
        return xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    if op_type == UsdGeom.XformOp.TypeRotateXYZ:
        return xformable.AddRotateXYZOp(UsdGeom.XformOp.PrecisionDouble)
    if op_type == UsdGeom.XformOp.TypeScale:
        return xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    raise ValueError(f"Unsupported xform op type: {op_type}")


def _set_transform(
    xformable: UsdGeom.Xformable,
    *,
    translate: Iterable[float] | None = None,
    rotate_xyz: Iterable[float] | None = None,
    scale: Iterable[float] | None = None,
) -> None:
    if translate is not None:
        _get_or_create_op(xformable, UsdGeom.XformOp.TypeTranslate).Set(Gf.Vec3d(*translate))
    if rotate_xyz is not None:
        _get_or_create_op(xformable, UsdGeom.XformOp.TypeRotateXYZ).Set(Gf.Vec3d(*rotate_xyz))
    if scale is not None:
        _get_or_create_op(xformable, UsdGeom.XformOp.TypeScale).Set(Gf.Vec3d(*scale))


def _define_world(stage: Usd.Stage, world_path: str) -> UsdGeom.Xform:
    world = UsdGeom.Xform.Define(stage, world_path)
    stage.SetDefaultPrim(world.GetPrim())
    return world


def _configure_stage(stage: Usd.Stage) -> None:
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)


def _clear_path(stage: Usd.Stage, path: str) -> None:
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        stage.RemovePrim(path)


def _set_color(gprim: UsdGeom.Gprim, rgb: tuple[float, float, float]) -> None:
    color = Gf.Vec3f(float(rgb[0]), float(rgb[1]), float(rgb[2]))
    gprim.CreateDisplayColorAttr().Set([color])


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _resolve_project_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return (_project_root() / path).resolve()


def _import_isaaclab_modules() -> tuple[Any, Any, Any]:
    _ensure_isaaclab_pythonpath()
    try:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors import TiledCamera, TiledCameraCfg
    except ImportError as exc:
        raise SystemExit(
            "This script requires Isaac Lab Python modules (isaaclab package not found). "
            "Run it from the Isaac Lab environment."
        ) from exc
    return sim_utils, TiledCamera, TiledCameraCfg


def _add_ground(
    stage: Usd.Stage,
    world_path: str,
    *,
    size_x: float,
    size_y: float,
    thickness: float,
    top_z: float,
    color: tuple[float, float, float],
) -> None:
    ground_path = f"{world_path}/Ground"
    _clear_path(stage, ground_path)
    cube = UsdGeom.Cube.Define(stage, ground_path)
    cube.CreateSizeAttr(1.0)
    _set_color(cube, color)

    center_z = top_z - 0.5 * thickness
    _set_transform(
        UsdGeom.Xformable(cube.GetPrim()),
        translate=(0.0, 0.0, center_z),
        scale=(size_x, size_y, thickness),
    )
    # Make ground participate in physics as a static collider.
    ground_prim = cube.GetPrim()
    if not ground_prim.HasAPI(UsdPhysics.CollisionAPI):
        UsdPhysics.CollisionAPI.Apply(ground_prim)
    mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(ground_prim)
    mesh_collision.CreateApproximationAttr().Set("boundingCube")


def _add_ground_grid(
    stage: Usd.Stage,
    world_path: str,
    *,
    size_x: float,
    size_y: float,
    top_z: float,
    spacing: float,
    line_width: float,
    color: tuple[float, float, float],
) -> None:
    grid_path = f"{world_path}/GroundGrid"
    _clear_path(stage, grid_path)
    curves = UsdGeom.BasisCurves.Define(stage, grid_path)

    half_x = max(float(size_x), 1.0)
    half_y = max(float(size_y), 1.0)
    step = max(float(spacing), 0.01)
    z = float(top_z) + 0.002
    eps = 1e-6

    points: list[Any] = []
    counts: list[int] = []

    y = -half_y
    while y <= half_y + eps:
        points.extend((Gf.Vec3f(-half_x, y, z), Gf.Vec3f(half_x, y, z)))
        counts.append(2)
        y += step

    x = -half_x
    while x <= half_x + eps:
        points.extend((Gf.Vec3f(x, -half_y, z), Gf.Vec3f(x, half_y, z)))
        counts.append(2)
        x += step

    curves.CreateTypeAttr(UsdGeom.Tokens.linear)
    curves.CreateBasisAttr(UsdGeom.Tokens.bezier)
    curves.CreateWrapAttr(UsdGeom.Tokens.nonperiodic)
    curves.CreateCurveVertexCountsAttr(counts)
    curves.CreatePointsAttr(points)
    curves.CreateWidthsAttr([max(float(line_width), 0.0001)])
    _set_color(curves, color)


def _add_lights(
    stage: Usd.Stage,
    world_path: str,
    args: argparse.Namespace,
) -> None:
    lights_path = f"{world_path}/Lights"
    if args.clear_lights:
        _clear_path(stage, lights_path)
    UsdGeom.Xform.Define(stage, lights_path)

    if not args.disable_dome_light:
        dome = UsdLux.DomeLight.Define(stage, f"{lights_path}/DomeLight")
        dome.CreateIntensityAttr(float(args.dome_intensity))
        dome.CreateExposureAttr(float(args.dome_exposure))
        dome.CreateColorAttr(Gf.Vec3f(*args.dome_color))

    if not args.disable_sun_light:
        sun = UsdLux.DistantLight.Define(stage, f"{lights_path}/SunLight")
        sun.CreateIntensityAttr(float(args.sun_intensity))
        sun.CreateColorAttr(Gf.Vec3f(*args.sun_color))
        sun.CreateAngleAttr(float(args.sun_angle))
        _set_transform(
            UsdGeom.Xformable(sun.GetPrim()),
            translate=tuple(args.sun_translate),
            rotate_xyz=tuple(args.sun_rotate_xyz),
        )

    if not args.disable_distant_light:
        distant = UsdLux.DistantLight.Define(stage, f"{lights_path}/DistantLight")
        distant.CreateIntensityAttr(float(args.distant_intensity))
        distant.CreateColorAttr(Gf.Vec3f(*args.distant_color))
        distant.CreateAngleAttr(float(args.distant_angle))
        _set_transform(
            UsdGeom.Xformable(distant.GetPrim()),
            translate=tuple(args.distant_translate),
            rotate_xyz=tuple(args.distant_rotate_xyz),
        )


def _add_robot_from_urdf(args: argparse.Namespace, robot_urdf_path: Path, *, sim_utils: Any) -> None:
    robot_cfg = sim_utils.UrdfFileCfg(
        asset_path=str(robot_urdf_path),
        fix_base=bool(args.fix_robot_base),
        merge_fixed_joints=False,
        self_collision=bool(args.robot_self_collision),
        collision_from_visuals=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=None, damping=None),
            target_type="none",
        ),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=False,
            disable_gravity=False,
            linear_damping=float(args.robot_linear_damping),
            angular_damping=float(args.robot_angular_damping),
            max_linear_velocity=float(args.robot_max_linear_velocity),
            max_angular_velocity=float(args.robot_max_angular_velocity),
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=float(args.robot_contact_offset),
            rest_offset=float(args.robot_rest_offset),
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            articulation_enabled=True,
            enabled_self_collisions=bool(args.robot_self_collision),
            fix_root_link=bool(args.fix_robot_base),
        ),
        activate_contact_sensors=True,
    )
    robot_cfg.func(
        args.robot_prim_path,
        robot_cfg,
        translation=tuple(float(v) for v in args.robot_init_pos),
        orientation=tuple(float(v) for v in args.robot_init_rot),
    )
    print(
        "Robot loaded from URDF:"
        f" urdf={robot_urdf_path},"
        f" prim={args.robot_prim_path},"
        f" init_pos={tuple(args.robot_init_pos)},"
        " physics=on, collision=on"
    , flush=True)


def _add_tiled_camera(
    args: argparse.Namespace,
    *,
    sim_utils: Any,
    TiledCamera: Any,
    TiledCameraCfg: Any,
) -> Any:
    tiled_camera_cfg = TiledCameraCfg(
        prim_path=args.tiled_cam_prim_path,
        update_period=float(args.tiled_cam_update_period),
        width=int(args.tiled_cam_width),
        height=int(args.tiled_cam_height),
        data_types=["rgb", "depth"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(args.tiled_cam_focal_length),
            focus_distance=float(args.tiled_cam_focus_distance),
            horizontal_aperture=float(args.tiled_cam_horizontal_aperture),
            clipping_range=(float(args.tiled_cam_clip_near), float(args.tiled_cam_clip_far)),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(float(v) for v in args.tiled_cam_offset_pos),
            rot=tuple(float(v) for v in args.tiled_cam_offset_rot),
            convention=str(args.tiled_cam_offset_convention),
        ),
    )
    tiled_camera = TiledCamera(tiled_camera_cfg)
    print(
        "Tiled camera created:"
        f" prim={args.tiled_cam_prim_path},"
        " data_types=['rgb','depth'],"
        f" resolution=({args.tiled_cam_width}x{args.tiled_cam_height})"
    , flush=True)
    return tiled_camera


def _warm_up_tiled_camera(tiled_camera: Any, sim: Any, *, sim_dt: float, warmup_steps: int) -> None:
    for _ in range(max(int(warmup_steps), 1)):
        sim.step()
    tiled_camera.update(sim_dt)

    rgb = tiled_camera.data.output.get("rgb")
    depth = tiled_camera.data.output.get("depth")
    if depth is None:
        depth = tiled_camera.data.output.get("distance_to_image_plane")
    rgb_shape = tuple(rgb.shape) if rgb is not None else None
    depth_shape = tuple(depth.shape) if depth is not None else None
    print(f"Tiled camera data ready: rgb_shape={rgb_shape}, depth_shape={depth_shape}", flush=True)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Edit an Isaac Lab drone environment scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--world-path", type=str, default="/World", help="Root prim path for the world.")
    parser.add_argument(
        "--close-after-build",
        action="store_true",
        help="Exit immediately after editing the stage. By default GUI stays open for inspection.",
    )
    parser.add_argument("--headless", action="store_true", help="Run without GUI window.")
    parser.add_argument("--sim-dt", type=float, default=1.0 / 60.0, help="Physics simulation timestep.")
    parser.add_argument("--sim-device", type=str, default="cuda:0", help='Simulation device, e.g. "cuda:0" or "cpu".')

    parser.add_argument("--ground-size", type=float, nargs=2, metavar=("X", "Y"), default=(4000.0, 4000.0))
    parser.add_argument("--ground-thickness", type=float, default=0.25)
    parser.add_argument("--ground-top-z", type=float, default=0.0)
    parser.add_argument("--ground-color", type=float, nargs=3, default=(0.35, 0.35, 0.35))
    parser.add_argument("--disable-ground-grid", action="store_true")
    parser.add_argument("--ground-grid-spacing", type=float, default=5.0)
    parser.add_argument("--ground-grid-line-width", type=float, default=0.03)
    parser.add_argument("--ground-grid-color", type=float, nargs=3, default=(0.62, 0.62, 0.62))

    parser.add_argument("--clear-lights", action="store_true", help="Clear /World/Lights before adding new lights.")
    parser.add_argument("--disable-dome-light", action="store_true")
    parser.add_argument("--dome-intensity", type=float, default=1000.0)
    parser.add_argument("--dome-exposure", type=float, default=0.0)
    parser.add_argument("--dome-color", type=float, nargs=3, default=(1.0, 1.0, 1.0))
    parser.add_argument("--disable-sun-light", action="store_true")
    parser.add_argument(
        "--sun-intensity",
        type=float,
        default=120000.0,
        help="Sunlight intensity. Higher values brighten very large environments.",
    )
    parser.add_argument("--sun-angle", type=float, default=0.53)
    parser.add_argument("--sun-color", type=float, nargs=3, default=(1.0, 0.98, 0.95))
    parser.add_argument("--sun-translate", type=float, nargs=3, default=(0.0, 0.0, 3000.0))
    parser.add_argument("--sun-rotate-xyz", type=float, nargs=3, default=(-55.0, 30.0, 0.0))
    parser.add_argument("--disable-distant-light", action="store_true")
    parser.add_argument("--distant-intensity", type=float, default=5000.0)
    parser.add_argument("--distant-angle", type=float, default=0.53)
    parser.add_argument("--distant-color", type=float, nargs=3, default=(1.0, 0.98, 0.95))
    parser.add_argument("--distant-translate", type=float, nargs=3, default=(0.0, 0.0, 25.0))
    parser.add_argument("--distant-rotate-xyz", type=float, nargs=3, default=(-50.0, 35.0, 0.0))

    parser.add_argument("--robot-urdf", type=str, default="assets/robot./robot.urdf")
    parser.add_argument("--robot-prim-path", type=str, default="/World/Robot")
    parser.add_argument("--robot-init-pos", type=float, nargs=3, default=(0.0, 0.0, 1.0))
    parser.add_argument("--robot-init-rot", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument(
        "--robot-visual-height",
        type=float,
        default=0.5,
        help="Approximate drone visual height in meters for overview camera framing.",
    )
    parser.add_argument("--fix-robot-base", action="store_true", help="Fix robot root to world.")
    parser.add_argument("--robot-self-collision", action="store_true", help="Enable robot self-collision.")
    parser.add_argument("--robot-contact-offset", type=float, default=0.01)
    parser.add_argument("--robot-rest-offset", type=float, default=0.0)
    parser.add_argument("--robot-linear-damping", type=float, default=0.0)
    parser.add_argument("--robot-angular-damping", type=float, default=0.0)
    parser.add_argument("--robot-max-linear-velocity", type=float, default=1000.0)
    parser.add_argument("--robot-max-angular-velocity", type=float, default=1000.0)

    parser.add_argument("--disable-tiled-camera", action="store_true")
    parser.add_argument("--tiled-cam-prim-path", type=str, default="/World/Robot/TiledCamera")
    parser.add_argument(
        "--tiled-cam-width",
        type=int,
        default=8,
        help="Tiled camera width (YOPO depth observation style default is 8).",
    )
    parser.add_argument(
        "--tiled-cam-height",
        type=int,
        default=8,
        help="Tiled camera height (YOPO depth observation style default is 8).",
    )
    parser.add_argument("--tiled-cam-update-period", type=float, default=0.0)
    parser.add_argument("--tiled-cam-offset-pos", type=float, nargs=3, default=(0.35, 0.0, 0.12))
    parser.add_argument("--tiled-cam-offset-rot", type=float, nargs=4, default=(1.0, 0.0, 0.0, 0.0))
    parser.add_argument(
        "--tiled-cam-offset-convention",
        type=str,
        choices=("world", "ros", "opengl"),
        default="world",
    )
    parser.add_argument("--tiled-cam-focal-length", type=float, default=24.0)
    parser.add_argument("--tiled-cam-focus-distance", type=float, default=400.0)
    parser.add_argument("--tiled-cam-horizontal-aperture", type=float, default=20.955)
    parser.add_argument("--tiled-cam-clip-near", type=float, default=0.05)
    parser.add_argument("--tiled-cam-clip-far", type=float, default=200.0)
    parser.add_argument("--tiled-cam-warmup-steps", type=int, default=6)
    parser.add_argument(
        "--overview-drone-screen-ratio",
        type=float,
        default=0.05,
        help="Target drone height ratio in default GUI view.",
    )
    parser.add_argument("--overview-cam-fov-deg", type=float, default=60.0)
    parser.add_argument("--overview-cam-yaw-deg", type=float, default=-30.0)
    parser.add_argument("--overview-cam-pitch-deg", type=float, default=16.0)

    return parser


def _create_new_stage() -> Usd.Stage:
    import omni.usd

    if not omni.usd.get_context().new_stage():
        raise RuntimeError("Cannot create a new USD stage.")

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Cannot obtain current USD stage from Isaac Sim.")
    return stage


def _is_gui_enabled() -> bool:
    import carb

    carb_settings = carb.settings.get_settings()
    local_gui = carb_settings.get("/app/window/enabled")
    livestream_gui = carb_settings.get("/app/livestream/enabled")
    return bool(local_gui or livestream_gui)


def _get_robot_overview_target_and_height(stage: Usd.Stage, args: argparse.Namespace) -> tuple[list[float], float]:
    """Estimate camera target/height from robot bounds, fallback to init pose."""
    default_height = max(float(args.robot_visual_height), 0.05)
    default_target = [
        float(args.robot_init_pos[0]),
        float(args.robot_init_pos[1]),
        float(args.robot_init_pos[2]) + 0.25 * default_height,
    ]
    robot_prim = stage.GetPrimAtPath(args.robot_prim_path)
    if not robot_prim or not robot_prim.IsValid():
        return default_target, default_height

    camera_prefix = str(args.tiled_cam_prim_path).rstrip("/")

    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
    )
    child_ranges: list[Any] = []
    for child in robot_prim.GetChildren():
        child_path = child.GetPath().pathString
        if child_path == camera_prefix or child_path.startswith(f"{camera_prefix}/"):
            continue
        child_bounds = bbox_cache.ComputeWorldBound(child).GetRange()
        if not child_bounds.IsEmpty():
            child_ranges.append(child_bounds)

    if child_ranges:
        min_x = min(float(rng.GetMin()[0]) for rng in child_ranges)
        min_y = min(float(rng.GetMin()[1]) for rng in child_ranges)
        min_z = min(float(rng.GetMin()[2]) for rng in child_ranges)
        max_x = max(float(rng.GetMax()[0]) for rng in child_ranges)
        max_y = max(float(rng.GetMax()[1]) for rng in child_ranges)
        max_z = max(float(rng.GetMax()[2]) for rng in child_ranges)
        center = [(min_x + max_x) * 0.5, (min_y + max_y) * 0.5, (min_z + max_z) * 0.5]
        size = [max_x - min_x, max_y - min_y, max_z - min_z]
    else:
        bounds = bbox_cache.ComputeWorldBound(robot_prim)
        bounds_range = bounds.GetRange()
        if bounds_range.IsEmpty():
            return default_target, default_height
        center = list(bounds_range.GetMidpoint())
        size = list(bounds_range.GetSize())

    # Prefer world-space Z height; fallback to max extent for flat/invalid bounds.
    approx_height = max(float(size[2]), float(max(size)), 0.05)
    target = [float(center[0]), float(center[1]), float(center[2])]

    # Guard against outlier bounds (typically camera/frustum pollution).
    max_reasonable_height = max(default_height * 8.0, 5.0)
    center_error = math.dist(target, default_target)
    if approx_height > max_reasonable_height or center_error > max(10.0, default_height * 20.0):
        return default_target, default_height

    return target, approx_height


def _set_overview_camera(stage: Usd.Stage, args: argparse.Namespace) -> None:
    """Set a drone-centric overview camera with controlled screen occupancy."""
    if not _is_gui_enabled():
        return

    try:
        from isaacsim.core.utils.viewports import set_camera_view
    except ImportError:
        return

    target_ratio = min(max(float(args.overview_drone_screen_ratio), 0.005), 0.5)
    vfov_deg = min(max(float(args.overview_cam_fov_deg), 10.0), 140.0)
    target, approx_height = _get_robot_overview_target_and_height(stage, args)

    # Small-angle model: projected height ~= 2*atan(h/(2d)) / vfov.
    angular_size = math.radians(vfov_deg) * target_ratio
    angular_size = max(angular_size, math.radians(0.2))
    eye_dist = approx_height / (2.0 * math.tan(0.5 * angular_size))
    eye_dist = min(max(eye_dist, 2.5), 120.0)

    yaw_rad = math.radians(float(args.overview_cam_yaw_deg))
    pitch_rad = math.radians(min(max(float(args.overview_cam_pitch_deg), 2.0), 85.0))
    planar = eye_dist * math.cos(pitch_rad)
    eye = [
        target[0] + planar * math.sin(yaw_rad),
        target[1] - planar * math.cos(yaw_rad),
        target[2] + eye_dist * math.sin(pitch_rad),
    ]
    set_camera_view(eye=eye, target=target, camera_prim_path="/OmniverseKit_Persp")
    print(
        "Camera adjusted for drone overview:"
        f" eye={eye}, target={target}, target_ratio={target_ratio:.3f},"
        f" approx_drone_height={approx_height}"
    , flush=True)


def _create_simulation_app(*, headless: bool) -> tuple[SimulationApp, Any]:
    global AppLauncher
    if AppLauncher is None:
        _ensure_isaaclab_pythonpath()
        with contextlib.suppress(ImportError):
            from isaaclab.app import AppLauncher as _AppLauncher

            AppLauncher = _AppLauncher
    if AppLauncher is not None:
        launcher = AppLauncher(headless=headless, enable_cameras=True)
        return launcher.app, launcher
    return SimulationApp({"headless": bool(headless)}), None


def main() -> int:
    parser = _build_argparser()
    args = parser.parse_args()

    if not args.world_path.startswith("/"):
        parser.error("--world-path must be an absolute USD path like /World")
    if not args.robot_prim_path.startswith("/"):
        parser.error("--robot-prim-path must be an absolute USD path like /World/Robot")
    if not args.tiled_cam_prim_path.startswith("/"):
        parser.error("--tiled-cam-prim-path must be an absolute USD path like /World/Robot/TiledCamera")
    if args.tiled_cam_width <= 0 or args.tiled_cam_height <= 0:
        parser.error("--tiled-cam-width/--tiled-cam-height must be positive integers")
    if args.robot_contact_offset < args.robot_rest_offset:
        parser.error("--robot-contact-offset must be >= --robot-rest-offset")

    robot_urdf_path = _resolve_project_path(args.robot_urdf)
    if not robot_urdf_path.is_file():
        parser.error(f"--robot-urdf file not found: {robot_urdf_path}")

    simulation_app, _app_owner = _create_simulation_app(headless=bool(args.headless))
    sim = None

    try:
        _ensure_pxr_imported()
        sim_utils, TiledCamera, TiledCameraCfg = _import_isaaclab_modules()

        stage = _create_new_stage()
        _configure_stage(stage)
        sim_cfg = sim_utils.SimulationCfg(dt=float(args.sim_dt), device=str(args.sim_device))
        sim = sim_utils.SimulationContext(sim_cfg)

        _define_world(stage, args.world_path)

        _add_ground(
            stage,
            args.world_path,
            size_x=float(args.ground_size[0]),
            size_y=float(args.ground_size[1]),
            thickness=float(args.ground_thickness),
            top_z=float(args.ground_top_z),
            color=tuple(args.ground_color),
        )
        if not args.disable_ground_grid:
            _add_ground_grid(
                stage,
                args.world_path,
                size_x=float(args.ground_size[0]) * 0.5,
                size_y=float(args.ground_size[1]) * 0.5,
                top_z=float(args.ground_top_z),
                spacing=float(args.ground_grid_spacing),
                line_width=float(args.ground_grid_line_width),
                color=tuple(args.ground_grid_color),
            )
        _add_lights(stage, args.world_path, args)
        _add_robot_from_urdf(args, robot_urdf_path, sim_utils=sim_utils)

        tiled_camera = None
        if not args.disable_tiled_camera:
            tiled_camera = _add_tiled_camera(
                args,
                sim_utils=sim_utils,
                TiledCamera=TiledCamera,
                TiledCameraCfg=TiledCameraCfg,
            )

        sim_utils.update_stage()
        sim.reset()

        if tiled_camera is not None:
            _warm_up_tiled_camera(
                tiled_camera,
                sim,
                sim_dt=float(args.sim_dt),
                warmup_steps=int(args.tiled_cam_warmup_steps),
            )

        _set_overview_camera(stage, args)
        print("Scene updated in current Isaac Lab stage.", flush=True)

        print(
            "Summary:"
            f" ground=({args.ground_size[0]} x {args.ground_size[1]}),"
            f" ground_grid={'off' if args.disable_ground_grid else f'on(spacing={args.ground_grid_spacing})'},"
            f" robot_urdf={robot_urdf_path},"
            f" robot_init_pos={tuple(args.robot_init_pos)},"
            f" tiled_camera={'off' if args.disable_tiled_camera else f'on({args.tiled_cam_width}x{args.tiled_cam_height})'},"
            f" dome_light={'off' if args.disable_dome_light else 'on'},"
            f" sun_light={'off' if args.disable_sun_light else 'on'},"
            f" distant_light={'off' if args.disable_distant_light else 'on'}"
        , flush=True)

        if not args.headless and not args.close_after_build:
            if _is_gui_enabled():
                print("Scene is ready in GUI. Close the Isaac Lab window to exit.", flush=True)
            else:
                print(
                    "GUI keep-alive is enabled. If no window appears, check DISPLAY/xhost. "
                    "Use Ctrl+C to stop.",
                    flush=True,
                )
            print("Entering foreground keep-alive loop. Press Ctrl+C to exit.", flush=True)
            with contextlib.suppress(KeyboardInterrupt):
                while True:
                    try:
                        if simulation_app.is_running():
                            simulation_app.update()
                        else:
                            time.sleep(0.1)
                    except Exception as exc:
                        print(f"Keep-alive loop warning: {exc!r}", flush=True)
                        time.sleep(0.2)
    finally:
        if sim is not None:
            with contextlib.suppress(Exception):
                sim.stop()
                sim.clear_all_callbacks()
                sim.clear_instance()

        if args.headless or args.close_after_build:
            simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
        else:
            simulation_app.close(wait_for_replicator=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
