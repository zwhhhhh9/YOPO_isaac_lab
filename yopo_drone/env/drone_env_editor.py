#!/usr/bin/env python3
"""Edit Isaac Lab drone simulation scenes from the command line.

Example:
    # Preview directly in Isaac Lab GUI (default keeps window open):
    python yopo_drone/run.py yopo_drone/env/drone_env_editor.py \
        --sun-intensity 120000

    # Headless smoke test (no GUI):
    python yopo_drone/run.py yopo_drone/env/drone_env_editor.py \
        --headless --close-after-build
"""

from __future__ import annotations

import argparse
import contextlib
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
    global Gf, Usd, UsdGeom
    if all(module is not None for module in (Gf, Usd, UsdGeom)):
        return
    try:
        from pxr import (
            Gf as _Gf,
            Usd as _Usd,
            UsdGeom as _UsdGeom,
        )
    except ImportError as exc:
        raise SystemExit(
            "This script requires Isaac Sim / Isaac Lab Python runtime (pxr module not found). "
            "Rebuild the image with usd-core installed."
        ) from exc
    Gf, Usd, UsdGeom = _Gf, _Usd, _UsdGeom


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


def _add_lights(world_path: str, *, sim_utils: Any) -> None:
    """Add Isaac Lab default dome light (intensity=2000, color=(0.75, 0.75, 0.75))."""
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func(f"{world_path}/Lights/DomeLight", light_cfg)
    print("Dome light added (Isaac Lab default: intensity=2000, color=(0.75, 0.75, 0.75)).", flush=True)


def _enable_viewport_grid() -> None:
    try:
        import carb
        s = carb.settings.get_settings()
        # Try known keys across Isaac Sim versions
        for key in (
            "/app/viewport/grid/enabled",
            "/persistent/app/viewport/grid/enabled",
            "/app/viewport/show/grid",
        ):
            with contextlib.suppress(Exception):
                s.set(key, True)
    except Exception:
        pass


def _add_ground(world_path: str, *, sim_utils: Any) -> None:
    # Prefer Isaac Lab's built-in ground plane spawner for compatibility/stability.
    ground_cfg = sim_utils.GroundPlaneCfg(size=(500.0, 500.0), color=(0.0, 0.0, 0.0))
    ground_cfg.func(f"{world_path}/Ground", ground_cfg, translation=(0.0, 0.0, 0.0))
    print(f"Ground plane added at {world_path}/Ground (500x500).", flush=True)


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


def _attach_tiled_camera_inset(args: argparse.Namespace) -> Any:
    """Create a small secondary viewport that follows the tiled camera."""
    if args.headless or args.disable_tiled_camera or args.disable_tiled_camera_inset:
        return None
    if not _is_gui_enabled():
        return None
    try:
        from pxr import Sdf
        import omni.ui as ui
        from omni.kit.viewport.window import ViewportWindow

        inset_window = ViewportWindow(
            name="Tiled Camera",
            width=int(args.tiled_cam_inset_width),
            height=int(args.tiled_cam_inset_height),
        )
        # ViewportWindow visibility is method-based in Kit API.
        with contextlib.suppress(Exception):
            inset_window.visible(True)
        with contextlib.suppress(Exception):
            inset_window.visible = True
        inset_window.viewport_api.camera_path = Sdf.Path(str(args.tiled_cam_prim_path))

        def _to_int(value: Any, default: int) -> int:
            with contextlib.suppress(Exception):
                return int(value)
            with contextlib.suppress(Exception):
                return int(float(value))
            return default

        def _place_inset(window_obj: Any) -> None:
            """Place inset above Render Settings window; fallback to default coords."""
            render_settings_window = ui.Workspace.get_window("Render Settings")
            if render_settings_window is not None:
                inset_h = int(args.tiled_cam_inset_height)
                margin = 12

                rs_x = _to_int(getattr(render_settings_window, "position_x", 0), 0)
                rs_y = _to_int(getattr(render_settings_window, "position_y", 0), 0)
                target_x = rs_x
                target_y = max(0, rs_y - inset_h - margin)

                window_obj.position_x = target_x
                window_obj.position_y = target_y
                return

            window_obj.position_x = int(args.tiled_cam_inset_pos_x)
            window_obj.position_y = int(args.tiled_cam_inset_pos_y)

        # Initial placement and continuous lock (prevents manual dragging).
        _place_inset(inset_window)

        print(
            "Tiled camera inset attached:"
            " window='Tiled Camera',"
            f" size=({args.tiled_cam_inset_width}x{args.tiled_cam_inset_height}),"
            f" camera={args.tiled_cam_prim_path},"
            " anchor='above Render Settings'"
        , flush=True)
        return inset_window
    except Exception as exc:
        print(f"Tiled camera inset warning: {exc!r}", flush=True)
        return None


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

    parser.add_argument("--clear-lights", action="store_true", help="Clear /World/Lights before adding new lights.")
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
        "--disable-tiled-camera-inset",
        action="store_true",
        help="Disable the small GUI inset viewport that displays the tiled camera feed.",
    )
    parser.add_argument("--tiled-cam-inset-window-name", type=str, default="Tiled Camera")
    parser.add_argument("--tiled-cam-inset-width", type=int, default=360)
    parser.add_argument("--tiled-cam-inset-height", type=int, default=240)
    parser.add_argument("--tiled-cam-inset-pos-x", type=int, default=60)
    parser.add_argument("--tiled-cam-inset-pos-y", type=int, default=120)

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




def _set_overview_camera(sim: Any) -> None:
    """Set Isaac Lab default viewer camera (eye=(7.5,7.5,7.5), lookat=(0,0,0))."""
    if not _is_gui_enabled():
        return
    try:
        from isaaclab.envs.ui.viewport_camera_controller import ViewportCameraController
        from isaaclab.utils import configclass

        @configclass
        class _ViewerCfg:
            eye: tuple = (7.5, 7.5, 7.5)
            lookat: tuple = (0.0, 0.0, 0.0)
            cam_prim_path: str = "/OmniverseKit_Persp"
            resolution: tuple = (1280, 720)
            origin_type: str = "world"
            env_index: int = 0
            asset_name: str | None = None
            body_name: str | None = None

        ViewportCameraController(sim, _ViewerCfg())
        print("Viewer camera set to Isaac Lab default: eye=(7.5,7.5,7.5), lookat=(0,0,0).", flush=True)
    except Exception:
        # Fallback: use set_camera_view directly
        try:
            from isaacsim.core.utils.viewports import set_camera_view
            set_camera_view(eye=(7.5, 7.5, 7.5), target=(0.0, 0.0, 0.0), camera_prim_path="/OmniverseKit_Persp")
            print("Viewer camera set to Isaac Lab default: eye=(7.5,7.5,7.5), lookat=(0,0,0).", flush=True)
        except Exception:
            pass


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
    inset_window = None

    try:
        _ensure_pxr_imported()
        sim_utils, TiledCamera, TiledCameraCfg = _import_isaaclab_modules()

        stage = _create_new_stage()
        _configure_stage(stage)
        sim_cfg = sim_utils.SimulationCfg(dt=float(args.sim_dt), device=str(args.sim_device))
        sim = sim_utils.SimulationContext(sim_cfg)
        # Disable auto ground plane so our custom ground is not overwritten on reset
        sim.cfg.add_ground_plane = False

        _define_world(stage, args.world_path)
        _add_lights(args.world_path, sim_utils=sim_utils)
        _add_ground(args.world_path, sim_utils=sim_utils)
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
            inset_window = _attach_tiled_camera_inset(args)

        _set_overview_camera(sim)
        _enable_viewport_grid()
        print("Scene updated in current Isaac Lab stage.", flush=True)

        print(
            "Summary:"
            f" robot_urdf={robot_urdf_path},"
            f" robot_init_pos={tuple(args.robot_init_pos)},"
            f" tiled_camera={'off' if args.disable_tiled_camera else f'on({args.tiled_cam_width}x{args.tiled_cam_height})'}"
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
                while simulation_app.is_running():
                    try:
                        sim.step()
                    except Exception as exc:
                        print(f"Keep-alive loop warning: {exc!r}", flush=True)
                        time.sleep(0.2)
    finally:
        if sim is not None:
            with contextlib.suppress(Exception):
                sim.stop()
                sim.clear_all_callbacks()
                sim.clear_instance()

        if inset_window is not None:
            with contextlib.suppress(Exception):
                inset_window.destroy()

        if args.headless or args.close_after_build:
            simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
        else:
            simulation_app.close(wait_for_replicator=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
