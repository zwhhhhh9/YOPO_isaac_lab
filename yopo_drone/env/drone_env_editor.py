#!/usr/bin/env python3
"""Edit Isaac Lab drone simulation scenes from the command line.

Example:
    # Preview directly in Isaac Lab GUI (default keeps window open):
    python yopo_drone/run.py yopo_drone/env/drone_env_editor.py \
        --ground-size 100000 100000 \
        --sun-intensity 120000

    # Headless smoke test (no GUI):
    python yopo_drone/run.py yopo_drone/env/drone_env_editor.py \
        --headless --close-after-build
"""

from __future__ import annotations

import argparse
import contextlib
from typing import Iterable

with contextlib.suppress(ModuleNotFoundError):
    import isaacsim  # noqa: F401

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


def _ensure_pxr_imported() -> None:
    global Gf, Usd, UsdGeom, UsdLux
    if all(module is not None for module in (Gf, Usd, UsdGeom, UsdLux)):
        return
    try:
        from pxr import Gf as _Gf, Usd as _Usd, UsdGeom as _UsdGeom, UsdLux as _UsdLux
    except ImportError as exc:
        raise SystemExit(
            "This script requires Isaac Sim / Isaac Lab Python runtime (pxr module not found). "
            "Rebuild the image with usd-core installed."
        ) from exc
    Gf, Usd, UsdGeom, UsdLux = _Gf, _Usd, _UsdGeom, _UsdLux


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

    parser.add_argument("--ground-size", type=float, nargs=2, metavar=("X", "Y"), default=(100000.0, 100000.0))
    parser.add_argument("--ground-thickness", type=float, default=0.25)
    parser.add_argument("--ground-top-z", type=float, default=0.0)
    parser.add_argument("--ground-color", type=float, nargs=3, default=(0.35, 0.35, 0.35))

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


def main() -> int:
    parser = _build_argparser()
    args = parser.parse_args()
    if not args.world_path.startswith("/"):
        parser.error("--world-path must be an absolute USD path like /World")

    simulation_app = SimulationApp({"headless": bool(args.headless)})

    try:
        _ensure_pxr_imported()
        stage = _create_new_stage()
        _configure_stage(stage)
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
        _add_lights(stage, args.world_path, args)
        print("Scene updated in current Isaac Lab stage.")

        print(
            "Summary:"
            f" ground=({args.ground_size[0]} x {args.ground_size[1]}),"
            f" dome_light={'off' if args.disable_dome_light else 'on'},"
            f" sun_light={'off' if args.disable_sun_light else 'on'},"
            f" distant_light={'off' if args.disable_distant_light else 'on'}"
        )

        if _is_gui_enabled() and not args.close_after_build:
            import omni.kit.app

            print("Scene is ready in GUI. Close the Isaac Lab window to exit.")
            app = omni.kit.app.get_app_interface()
            with contextlib.suppress(KeyboardInterrupt):
                while app.is_running():
                    app.update()
    finally:
        if args.headless or args.close_after_build:
            simulation_app.close(wait_for_replicator=False, skip_cleanup=True)
        else:
            simulation_app.close(wait_for_replicator=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
