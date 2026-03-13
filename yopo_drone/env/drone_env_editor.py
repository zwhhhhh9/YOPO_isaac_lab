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
import asyncio
import contextlib
import sys
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim  # noqa: F401

from yopo_drone.env.random_forest_scene import (
    add_random_forest_arguments,
    add_random_forest_scene,
    build_random_forest_cfg_from_args,
)
from yopo_drone.utils.robot_model import DEFAULT_ROBOT_URDF

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
    ground_cfg = sim_utils.GroundPlaneCfg(size=(500.0, 500.0), color=(0.24, 0.16, 0.10))
    ground_cfg.func(f"{world_path}/Ground", ground_cfg, translation=(0.0, 0.0, 0.0))
    print(f"Ground plane added at {world_path}/Ground (500x500, deep brown soil tone).", flush=True)


def _add_world_origin_frame(world_path: str) -> None:
    """Add a fixed world-frame marker at the scene origin."""
    try:
        import numpy as np

        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
        from isaaclab.markers.config import FRAME_MARKER_CFG
    except Exception as exc:
        print(f"World origin frame warning: {exc!r}", flush=True)
        return

    frame_marker_cfg = FRAME_MARKER_CFG.markers["frame"].copy()
    frame_marker_cfg.scale = (0.15, 0.15, 0.15)
    marker_cfg = VisualizationMarkersCfg(
        prim_path=f"{world_path}/Visuals/WorldOriginFrame",
        markers={"frame": frame_marker_cfg},
    )
    marker = VisualizationMarkers(marker_cfg)
    marker.visualize(
        translations=np.array([[0.0, 0.0, 0.0]], dtype=np.float32),
        orientations=np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
        marker_indices=np.array([0], dtype=np.int32),
    )
    print(f"World origin frame added at {world_path} origin.", flush=True)


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
    prim_path: str | None = None,
    data_types: Iterable[str] | None = None,
    clip_near: float | None = None,
    clip_far: float | None = None,
    log_name: str = "Tiled camera",
) -> Any:
    resolved_prim_path = str(args.tiled_cam_prim_path if prim_path is None else prim_path)
    resolved_data_types = list(data_types) if data_types is not None else ["rgb", "depth"]
    tiled_camera_cfg = TiledCameraCfg(
        prim_path=resolved_prim_path,
        update_period=float(args.tiled_cam_update_period),
        width=int(args.tiled_cam_width),
        height=int(args.tiled_cam_height),
        data_types=resolved_data_types,
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(args.tiled_cam_focal_length),
            focus_distance=float(args.tiled_cam_focus_distance),
            horizontal_aperture=float(args.tiled_cam_horizontal_aperture),
            clipping_range=(
                float(args.tiled_cam_clip_near if clip_near is None else clip_near),
                float(args.tiled_cam_clip_far if clip_far is None else clip_far),
            ),
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=tuple(float(v) for v in args.tiled_cam_offset_pos),
            rot=tuple(float(v) for v in args.tiled_cam_offset_rot),
            convention=str(args.tiled_cam_offset_convention),
        ),
    )
    tiled_camera = TiledCamera(tiled_camera_cfg)
    print(
        f"{log_name} created:"
        f" prim={resolved_prim_path},"
        f" data_types={resolved_data_types},"
        f" resolution=({args.tiled_cam_width}x{args.tiled_cam_height})"
    , flush=True)
    return tiled_camera


def _derive_depth_tiled_camera_prim_path(prim_path: str) -> str:
    prim_path = str(prim_path).rstrip("/")
    tail = prim_path.rsplit("/", 1)[-1]
    if tail == "TiledCamera":
        return prim_path[: -len("TiledCamera")] + "TiledDepthCamera"
    return f"{prim_path}_Depth"


def _warm_up_tiled_camera(
    tiled_camera: Any,
    sim: Any,
    *,
    sim_dt: float,
    warmup_steps: int,
    log_name: str = "Tiled camera",
) -> None:
    for _ in range(max(int(warmup_steps), 1)):
        sim.step()
    tiled_camera.update(sim_dt)

    rgb = tiled_camera.data.output.get("rgb")
    depth = tiled_camera.data.output.get("depth")
    if depth is None:
        depth = tiled_camera.data.output.get("distance_to_image_plane")
    rgb_shape = tuple(rgb.shape) if rgb is not None else None
    depth_shape = tuple(depth.shape) if depth is not None else None
    print(f"{log_name} data ready: rgb_shape={rgb_shape}, depth_shape={depth_shape}", flush=True)


def _schedule_tiled_camera_left_stack_dock(
    *,
    ui_module: Any,
    rgb_window: Any | None = None,
    depth_window: Any | None = None,
    rgb_window_name: str = "Tiled Camera",
) -> Any:
    if rgb_window is None and depth_window is None:
        return None
    left_dock_ratio = 0.36
    depth_window_name = str(getattr(depth_window, "title", "")) if depth_window is not None else ""

    def _dock_window(window_obj: Any, *, target_name: str, target_handle: Any, dock_position: Any, ratio: float) -> bool:
        if window_obj is None:
            return False
        with contextlib.suppress(Exception):
            if target_handle is not None:
                window_obj.dock_in(target_handle, dock_position, float(ratio))
                return True
        with contextlib.suppress(Exception):
            dock_in_window = getattr(window_obj, "dock_in_window", None)
            if callable(dock_in_window):
                if bool(dock_in_window(str(target_name), dock_position, float(ratio))):
                    return True
        return False

    async def _dock_windows() -> None:
        try:
            import omni.kit.app

            app = omni.kit.app.get_app()
            await app.next_update_async()
            await app.next_update_async()

            completed = False
            attempts = 0
            rgb_state = "off"
            depth_state = "off"
            for attempts in range(1, 13):
                viewport_window = ui_module.Workspace.get_window("Viewport")
                rgb_handle = ui_module.Workspace.get_window(str(rgb_window_name))
                rgb_target = rgb_window if rgb_window is not None else rgb_handle
                if rgb_target is not None and viewport_window is not None:
                    _dock_window(
                        rgb_target,
                        target_name="Viewport",
                        target_handle=viewport_window,
                        dock_position=ui_module.DockPosition.LEFT,
                        ratio=left_dock_ratio,
                    )

                await app.next_update_async()

                depth_target = depth_window
                rgb_handle = ui_module.Workspace.get_window(str(rgb_window_name))
                if depth_target is not None:
                    if rgb_handle is not None:
                        _dock_window(
                            depth_target,
                            target_name=str(rgb_window_name),
                            target_handle=rgb_handle,
                            dock_position=ui_module.DockPosition.BOTTOM,
                            ratio=0.5,
                        )
                    elif viewport_window is not None:
                        _dock_window(
                            depth_target,
                            target_name="Viewport",
                            target_handle=viewport_window,
                            dock_position=ui_module.DockPosition.LEFT,
                            ratio=left_dock_ratio,
                        )

                await app.next_update_async()

                rgb_dock_id = int(getattr(ui_module.Workspace.get_window(str(rgb_window_name)), "dock_id", 0))
                depth_dock_id = int(getattr(ui_module.Workspace.get_window(depth_window_name), "dock_id", 0)) if depth_window_name else 0
                rgb_state = "off" if rgb_window is None and rgb_handle is None else ("docked" if rgb_dock_id > 0 else "floating")
                depth_state = "off" if depth_window is None else ("docked" if depth_dock_id > 0 else "floating")
                if (rgb_window is None or rgb_dock_id > 0) and (depth_window is None or depth_dock_id > 0):
                    completed = True
                    break

            if completed:
                print(
                    "Docked tiled camera windows:"
                    f" rgb={rgb_state},"
                    f" depth={depth_state},"
                    f" attempts={attempts},"
                    " layout='left_stack'",
                    flush=True,
                )
            else:
                print(
                    "Tiled camera dock warning:"
                    f" rgb={rgb_state},"
                    f" depth={depth_state},"
                    f" attempts={attempts},"
                    " layout='left_stack_incomplete'",
                    flush=True,
                )
        except Exception as exc:
            print(f"Tiled camera dock warning: {exc!r}", flush=True)

    return asyncio.ensure_future(_dock_windows())


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
            name=str(args.tiled_cam_inset_window_name),
            width=int(args.tiled_cam_inset_width),
            height=int(args.tiled_cam_inset_height),
            dockPreference=ui.DockPreference.LEFT_BOTTOM,
        )
        with contextlib.suppress(Exception):
            inset_window.visible(True)
        with contextlib.suppress(Exception):
            inset_window.visible = True
        inset_window.viewport_api.camera_path = Sdf.Path(str(args.tiled_cam_prim_path))
        inset_window._yopo_dock_task = _schedule_tiled_camera_left_stack_dock(
            ui_module=ui,
            rgb_window=inset_window,
            rgb_window_name=str(args.tiled_cam_inset_window_name),
        )

        print(
            "Tiled camera inset attached:"
            f" window='{args.tiled_cam_inset_window_name}',"
            f" size=({args.tiled_cam_inset_width}x{args.tiled_cam_inset_height}),"
            f" camera={args.tiled_cam_prim_path},"
            " dock='scheduled',"
            " layout='left_stack_rgb_top'"
        , flush=True)
        return inset_window
    except Exception as exc:
        print(f"Tiled camera inset warning: {exc!r}", flush=True)
        return None


class _UiDepthWindowHandle:
    def __init__(
        self,
        *,
        ui_module: Any,
        window_name: str,
        width: int,
        height: int,
        pos_x: int,
        pos_y: int,
    ) -> None:
        self._ui = ui_module
        self._window_name = str(window_name)
        self._width = int(width)
        self._height = int(height)
        self._window = None
        self._provider = None
        self._ensure_window()

    @property
    def window(self) -> Any:
        return self._window

    def _ensure_window(self) -> None:
        if self._window is not None and self._provider is not None:
            with contextlib.suppress(Exception):
                self._window.visible = True
            return

        self._provider = self._ui.ByteImageProvider()
        self._window = self._ui.Window(
            self._window_name,
            width=self._width,
            height=self._height,
            visible=True,
            padding_x=0,
            padding_y=0,
            dockPreference=self._ui.DockPreference.LEFT_BOTTOM,
        )
        with self._window.frame:
            with self._ui.ZStack(style={"margin": 0, "padding": 0}):
                self._ui.ImageWithProvider(self._provider)

    def show(self, image: np.ndarray) -> None:
        self._ensure_window()
        rgba_image = _to_rgba_uint8_image(image)
        self._provider.set_bytes_data(rgba_image.flatten().data, [rgba_image.shape[1], rgba_image.shape[0]])

    def destroy(self) -> None:
        if self._window is not None:
            with contextlib.suppress(Exception):
                self._window.visible = False
            with contextlib.suppress(Exception):
                self._window.destroy()
        self._window = None
        self._provider = None


def _attach_tiled_camera_depth_inset(args: argparse.Namespace, tiled_camera: Any) -> Any:
    """Create a GUI depth window for the tiled camera."""
    if args.headless or args.disable_tiled_camera or args.disable_tiled_camera_depth_inset:
        return None
    if not _is_gui_enabled():
        return None
    try:
        import omni.ui as ui

        depth_window = _UiDepthWindowHandle(
            ui_module=ui,
            window_name=args.tiled_cam_depth_inset_window_name,
            width=int(args.tiled_cam_depth_inset_width),
            height=int(args.tiled_cam_depth_inset_height),
            pos_x=int(args.tiled_cam_depth_inset_pos_x),
            pos_y=int(args.tiled_cam_depth_inset_pos_y),
        )
        state: dict[str, Any] = {
            "window": depth_window,
            "last_update_time": 0.0,
        }
        state["dock_task"] = _schedule_tiled_camera_left_stack_dock(
            ui_module=ui,
            depth_window=depth_window.window,
            rgb_window_name=str(args.tiled_cam_inset_window_name),
        )
        _update_tiled_camera_depth_inset(args, tiled_camera, state, force=True)
        print(
            "Tiled camera depth inset attached:"
            f" window='{args.tiled_cam_depth_inset_window_name}',"
            f" size=({args.tiled_cam_depth_inset_width}x{args.tiled_cam_depth_inset_height}),"
            f" clip=({args.tiled_cam_depth_vis_near},{args.tiled_cam_depth_vis_far}),"
            " dock='scheduled',"
            " layout='left_stack_depth_bottom'"
        , flush=True)
        return state
    except Exception as exc:
        print(f"Tiled camera depth inset warning: {exc!r}", flush=True)
        return None


def _update_tiled_camera_depth_inset(
    args: argparse.Namespace,
    tiled_camera: Any,
    depth_inset_state: Any,
    *,
    force: bool = False,
) -> None:
    if depth_inset_state is None:
        return
    update_interval = max(float(args.tiled_cam_depth_inset_update_interval), 0.0)
    now = time.monotonic()
    if not force and now - float(depth_inset_state["last_update_time"]) < update_interval:
        return

    depth = tiled_camera.data.output.get("depth")
    if depth is None:
        depth = tiled_camera.data.output.get("distance_to_image_plane")
    if depth is None:
        return

    depth_np = _depth_tensor_to_numpy(depth)
    near = float(args.tiled_cam_depth_vis_near)
    far = float(args.tiled_cam_depth_vis_far)
    grayscale_image = _colorize_depth_for_ui(
        depth_np,
        near=near,
        far=far,
    )
    depth_inset_state["window"].show(grayscale_image)
    depth_inset_state["last_update_time"] = now


def _depth_tensor_to_numpy(depth_tensor: Any) -> np.ndarray:
    depth_np = depth_tensor[0, ..., 0].detach().cpu().numpy()
    return np.asarray(depth_np, dtype=np.float32)


def _colorize_depth_for_ui(depth_np: np.ndarray, *, near: float, far: float) -> np.ndarray:
    near = max(float(near), 0.0)
    far = max(float(far), near + 1e-6)
    sanitized_depth = np.asarray(depth_np, dtype=np.float32).copy()
    # Treat "no return" pixels as very far so sky/background render white.
    no_return_mask = ~np.isfinite(sanitized_depth) | (sanitized_depth <= 0.0)
    sanitized_depth[no_return_mask] = far

    clipped = np.clip(sanitized_depth, near, far)
    normalized = (clipped - near) / (far - near)
    normalized = np.clip(normalized, 0.0, 1.0)
    return (normalized * 255.0).astype(np.uint8)


def _to_rgba_uint8_image(image: np.ndarray) -> np.ndarray:
    image_uint8 = np.asarray(image, dtype=np.uint8)
    if image_uint8.ndim == 2:
        return np.dstack(
            (
                image_uint8,
                image_uint8,
                image_uint8,
                np.full(image_uint8.shape, 255, dtype=np.uint8),
            )
        )
    if image_uint8.ndim == 3 and image_uint8.shape[2] == 1:
        gray = image_uint8[..., 0]
        return np.dstack((gray, gray, gray, np.full(gray.shape, 255, dtype=np.uint8)))
    if image_uint8.ndim == 3 and image_uint8.shape[2] == 3:
        alpha = np.full(image_uint8.shape[:2] + (1,), 255, dtype=np.uint8)
        return np.concatenate((image_uint8, alpha), axis=2)
    if image_uint8.ndim == 3 and image_uint8.shape[2] == 4:
        return image_uint8
    raise ValueError(f"Unsupported image shape for UI depth inset: {image_uint8.shape}")


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

    parser.add_argument("--robot-urdf", type=str, default=DEFAULT_ROBOT_URDF)
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

    add_random_forest_arguments(parser)

    parser.add_argument("--disable-tiled-camera", action="store_true")
    parser.add_argument("--tiled-cam-prim-path", type=str, default="/World/Robot/base_link/TiledCamera")
    parser.add_argument(
        "--tiled-cam-width",
        type=int,
        default=160,
        help="Tiled camera width (default is 160).",
    )
    parser.add_argument(
        "--tiled-cam-height",
        type=int,
        default=96,
        help="Tiled camera height (default is 96).",
    )
    parser.add_argument("--tiled-cam-update-period", type=float, default=0.0)
    parser.add_argument("--tiled-cam-offset-pos", type=float, nargs=3, default=(0.20, 0.0, 0.0))
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
    parser.add_argument("--tiled-cam-depth-clip-near", type=float, default=0.05)
    parser.add_argument("--tiled-cam-depth-clip-far", type=float, default=20.0)
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
    parser.add_argument(
        "--disable-tiled-camera-depth-inset",
        action="store_true",
        help="Disable the GUI depth inset for the tiled camera.",
    )
    parser.add_argument("--tiled-cam-depth-inset-window-name", type=str, default="Tiled Camera Depth")
    parser.add_argument("--tiled-cam-depth-inset-width", type=int, default=360)
    parser.add_argument("--tiled-cam-depth-inset-height", type=int, default=240)
    parser.add_argument("--tiled-cam-depth-inset-pos-x", type=int, default=1520)
    parser.add_argument("--tiled-cam-depth-inset-pos-y", type=int, default=60)
    parser.add_argument(
        "--tiled-cam-depth-inset-update-interval",
        type=float,
        default=0.1,
        help="Seconds between GUI depth inset refreshes.",
    )
    parser.add_argument(
        "--tiled-cam-depth-vis-near",
        type=float,
        default=0.0,
        help="Near depth used for grayscale mapping in the GUI depth inset.",
    )
    parser.add_argument(
        "--tiled-cam-depth-vis-far",
        type=float,
        default=20.0,
        help="Far depth used for grayscale mapping in the GUI depth inset.",
    )

    return parser


def _create_new_stage() -> Usd.Stage:
    import omni.usd

    if not omni.usd.get_context().new_stage():
        raise RuntimeError("Cannot create a new USD stage.")

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Cannot obtain current USD stage from Isaac Sim.")
    return stage


def _get_current_stage() -> Usd.Stage:
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Cannot obtain current USD stage from Isaac Sim.")
    return stage


def initialize_scene_from_editor(
    *,
    sim_utils: Any,
    world_path: str = "/World",
    create_new_stage: bool = True,
    clear_existing_world: bool = True,
    add_lights: bool = True,
    add_ground: bool = True,
) -> Usd.Stage:
    """Initialize the current Isaac stage with the editor's basic scene primitives.

    This helper is reusable from other Isaac Python programs, e.g. planner/control
    entry points that want the same ground/light world setup as `drone_env_editor.py`
    without creating a separate `SimulationContext` or spawning the robot twice.
    """
    if not world_path.startswith("/"):
        raise ValueError("world_path must be an absolute USD path like /World")

    _ensure_pxr_imported()
    stage = _create_new_stage() if create_new_stage else _get_current_stage()
    _configure_stage(stage)

    if clear_existing_world:
        _clear_path(stage, world_path)

    _define_world(stage, world_path)
    if add_lights:
        _add_lights(world_path, sim_utils=sim_utils)
    if add_ground:
        _add_ground(world_path, sim_utils=sim_utils)
    _add_world_origin_frame(world_path)
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
        parser.error("--tiled-cam-prim-path must be an absolute USD path like /World/Robot/base_link/TiledCamera")
    if args.tiled_cam_width <= 0 or args.tiled_cam_height <= 0:
        parser.error("--tiled-cam-width/--tiled-cam-height must be positive integers")
    if args.robot_contact_offset < args.robot_rest_offset:
        parser.error("--robot-contact-offset must be >= --robot-rest-offset")
    if args.random_forest_prim_path and not args.random_forest_prim_path.startswith("/"):
        parser.error("--random-forest-prim-path must be an absolute USD path like /World/Obstacles/RandomForest")

    robot_urdf_path = _resolve_project_path(args.robot_urdf)
    if not robot_urdf_path.is_file():
        parser.error(f"--robot-urdf file not found: {robot_urdf_path}")

    forest_cfg = None
    if args.add_random_forest:
        try:
            forest_cfg = build_random_forest_cfg_from_args(args)
        except ValueError as exc:
            parser.error(str(exc))

    simulation_app, _app_owner = _create_simulation_app(headless=bool(args.headless))
    sim = None
    inset_window = None
    depth_inset_state = None
    forest_summary = None

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
        _add_world_origin_frame(args.world_path)
        if forest_cfg is not None:
            forest_summary = add_random_forest_scene(sim_utils=sim_utils, cfg=forest_cfg)
        _add_robot_from_urdf(args, robot_urdf_path, sim_utils=sim_utils)

        tiled_camera = None
        depth_tiled_camera = None
        if not args.disable_tiled_camera:
            tiled_camera = _add_tiled_camera(
                args,
                sim_utils=sim_utils,
                TiledCamera=TiledCamera,
                TiledCameraCfg=TiledCameraCfg,
                data_types=["rgb"],
                log_name="Tiled RGB camera",
            )
            depth_tiled_camera = _add_tiled_camera(
                args,
                sim_utils=sim_utils,
                TiledCamera=TiledCamera,
                TiledCameraCfg=TiledCameraCfg,
                prim_path=_derive_depth_tiled_camera_prim_path(args.tiled_cam_prim_path),
                data_types=["depth"],
                clip_near=float(args.tiled_cam_depth_clip_near),
                clip_far=float(args.tiled_cam_depth_clip_far),
                log_name="Tiled depth camera",
            )

        sim_utils.update_stage()
        sim.reset()

        if tiled_camera is not None:
            _warm_up_tiled_camera(
                tiled_camera,
                sim,
                sim_dt=float(args.sim_dt),
                warmup_steps=int(args.tiled_cam_warmup_steps),
                log_name="Tiled RGB camera",
            )
            _warm_up_tiled_camera(
                depth_tiled_camera,
                sim,
                sim_dt=float(args.sim_dt),
                warmup_steps=int(args.tiled_cam_warmup_steps),
                log_name="Tiled depth camera",
            )
            inset_window = _attach_tiled_camera_inset(args)
            depth_inset_state = _attach_tiled_camera_depth_inset(args, depth_tiled_camera)

        _set_overview_camera(sim)
        _enable_viewport_grid()
        print("Scene updated in current Isaac Lab stage.", flush=True)
        random_forest_summary = "off" if forest_summary is None else f"on({forest_summary['tree_count']} trees)"

        print(
            "Summary:"
            f" robot_urdf={robot_urdf_path},"
            f" robot_init_pos={tuple(args.robot_init_pos)},"
            f" random_forest={random_forest_summary},"
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
                        if tiled_camera is not None:
                            tiled_camera.update(float(args.sim_dt))
                        if depth_tiled_camera is not None and depth_inset_state is not None:
                            depth_tiled_camera.update(float(args.sim_dt))
                            _update_tiled_camera_depth_inset(args, depth_tiled_camera, depth_inset_state)
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
        if depth_inset_state is not None:
            with contextlib.suppress(Exception):
                depth_inset_state["window"].destroy()

        if args.headless or args.close_after_build:
            simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
        else:
            simulation_app.close(wait_for_replicator=False)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
