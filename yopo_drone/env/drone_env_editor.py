#!/usr/bin/env python3
"""Edit Isaac Lab drone simulation scenes from the command line.

Example:
    # Preview directly in Isaac Lab GUI (default keeps window open):
    python yopo_drone/run.py yopo_drone/env/drone_env_editor.py \
        --ground-size 100000 100000 \
        --sun-intensity 120000 \
        --obstacle wall_1 8 0 2 1 12 4 \
        --grid-rows 3 --grid-cols 4 --grid-spacing 6 6 \
        --random-obstacles 15

    # Optional: export edited result to a USD file:
    python yopo_drone/run.py yopo_drone/env/drone_env_editor.py \
        --output-usd /tmp/drone_env.usd \
        --random-obstacles 10 --close-after-build
"""

from __future__ import annotations

import argparse
import contextlib
import random
import re
from pathlib import Path
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
Sdf = None
Usd = None
UsdGeom = None
UsdLux = None


def _ensure_pxr_imported() -> None:
    global Gf, Sdf, Usd, UsdGeom, UsdLux
    if all(module is not None for module in (Gf, Sdf, Usd, UsdGeom, UsdLux)):
        return
    try:
        from pxr import Gf as _Gf, Sdf as _Sdf, Usd as _Usd, UsdGeom as _UsdGeom, UsdLux as _UsdLux
    except ImportError as exc:
        raise SystemExit(
            "This script requires Isaac Sim / Isaac Lab Python runtime (pxr module not found). "
            "Rebuild the image with usd-core installed."
        ) from exc
    Gf, Sdf, Usd, UsdGeom, UsdLux = _Gf, _Sdf, _Usd, _UsdGeom, _UsdLux


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


def _add_box_obstacle(
    stage: Usd.Stage,
    obstacles_path: str,
    *,
    name: str,
    center: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
) -> None:
    path = Sdf.Path(obstacles_path).AppendChild(_sanitize_prim_name(name)).pathString
    _clear_path(stage, path)
    cube = UsdGeom.Cube.Define(stage, path)
    cube.CreateSizeAttr(1.0)
    _set_color(cube, color)
    _set_transform(UsdGeom.Xformable(cube.GetPrim()), translate=center, scale=size)


def _parse_obstacle_tokens(tokens: list[str]) -> tuple[str, tuple[float, float, float], tuple[float, float, float]]:
    if len(tokens) != 7:
        raise ValueError(
            "Each --obstacle needs 7 values: NAME X Y Z SX SY SZ"
        )
    name = tokens[0]
    values = tuple(float(v) for v in tokens[1:])
    return name, (values[0], values[1], values[2]), (values[3], values[4], values[5])


def _sanitize_prim_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", name.strip())
    if not cleaned:
        return "Obstacle"
    if cleaned[0].isdigit():
        cleaned = f"_{cleaned}"
    return cleaned


def _generate_grid_obstacles(
    stage: Usd.Stage,
    obstacles_path: str,
    args: argparse.Namespace,
) -> int:
    if args.grid_rows <= 0 or args.grid_cols <= 0:
        return 0

    count = 0
    size_x, size_y, size_z = args.grid_size
    spacing_x, spacing_y = args.grid_spacing
    origin_x, origin_y = args.grid_origin
    z = args.grid_base_z + size_z * 0.5
    start_x = origin_x - 0.5 * (args.grid_cols - 1) * spacing_x
    start_y = origin_y - 0.5 * (args.grid_rows - 1) * spacing_y

    for row in range(args.grid_rows):
        for col in range(args.grid_cols):
            jitter_x = random.uniform(-args.grid_jitter, args.grid_jitter)
            jitter_y = random.uniform(-args.grid_jitter, args.grid_jitter)
            x = start_x + col * spacing_x + jitter_x
            y = start_y + row * spacing_y + jitter_y
            _add_box_obstacle(
                stage,
                obstacles_path,
                name=f"grid_{row}_{col}",
                center=(x, y, z),
                size=(size_x, size_y, size_z),
                color=tuple(args.obstacle_color),
            )
            count += 1
    return count


def _generate_random_obstacles(
    stage: Usd.Stage,
    obstacles_path: str,
    args: argparse.Namespace,
) -> int:
    if args.random_obstacles <= 0:
        return 0

    if args.random_area is None:
        area_x = args.ground_size[0] * 0.45
        area_y = args.ground_size[1] * 0.45
    else:
        area_x, area_y = args.random_area

    min_sx, min_sy, min_sz = args.random_size_min
    max_sx, max_sy, max_sz = args.random_size_max
    if min_sx > max_sx or min_sy > max_sy or min_sz > max_sz:
        raise ValueError("--random-size-min must be <= --random-size-max for every axis")

    count = 0
    for i in range(args.random_obstacles):
        sx = random.uniform(min_sx, max_sx)
        sy = random.uniform(min_sy, max_sy)
        sz = random.uniform(min_sz, max_sz)
        x = random.uniform(-area_x, area_x)
        y = random.uniform(-area_y, area_y)
        z = args.random_base_z + sz * 0.5

        _add_box_obstacle(
            stage,
            obstacles_path,
            name=f"random_{i}",
            center=(x, y, z),
            size=(sx, sy, sz),
            color=tuple(args.obstacle_color),
        )
        count += 1
    return count


def _add_obstacles(stage: Usd.Stage, world_path: str, args: argparse.Namespace) -> int:
    obstacles_path = f"{world_path}/Obstacles"
    if args.clear_obstacles:
        _clear_path(stage, obstacles_path)
    UsdGeom.Xform.Define(stage, obstacles_path)

    added = 0
    for idx, tokens in enumerate(args.obstacle):
        name, center, size = _parse_obstacle_tokens(tokens)
        if not name:
            name = f"manual_{idx}"
        _add_box_obstacle(
            stage,
            obstacles_path,
            name=name,
            center=center,
            size=size,
            color=tuple(args.obstacle_color),
        )
        added += 1

    added += _generate_grid_obstacles(stage, obstacles_path, args)
    added += _generate_random_obstacles(stage, obstacles_path, args)
    return added


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Edit an Isaac Lab drone environment scene.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-usd", type=Path, default=None, help="Optional source USD scene.")
    parser.add_argument(
        "--output-usd",
        type=Path,
        default=None,
        help="Optional output USD scene. If omitted, only live stage in Isaac Lab is edited.",
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

    parser.add_argument(
        "--obstacle",
        nargs=7,
        action="append",
        default=[],
        metavar=("NAME", "X", "Y", "Z", "SX", "SY", "SZ"),
        help="Manual box obstacle. Repeat this argument to add more.",
    )
    parser.add_argument("--obstacle-color", type=float, nargs=3, default=(0.85, 0.25, 0.2))
    parser.add_argument("--clear-obstacles", action="store_true", help="Clear /World/Obstacles before adding.")

    parser.add_argument("--grid-rows", type=int, default=0)
    parser.add_argument("--grid-cols", type=int, default=0)
    parser.add_argument("--grid-spacing", type=float, nargs=2, default=(6.0, 6.0))
    parser.add_argument("--grid-size", type=float, nargs=3, default=(1.5, 1.5, 3.0))
    parser.add_argument("--grid-origin", type=float, nargs=2, default=(0.0, 0.0))
    parser.add_argument("--grid-base-z", type=float, default=0.0)
    parser.add_argument("--grid-jitter", type=float, default=0.0, help="Max random offset in XY for each grid obstacle.")

    parser.add_argument("--random-obstacles", type=int, default=0)
    parser.add_argument(
        "--random-area",
        type=float,
        nargs=2,
        default=None,
        metavar=("HALF_X", "HALF_Y"),
        help="Spawn area half-extent in XY. Defaults to 45%% of ground size.",
    )
    parser.add_argument("--random-size-min", type=float, nargs=3, default=(0.8, 0.8, 1.2))
    parser.add_argument("--random-size-max", type=float, nargs=3, default=(3.0, 3.0, 4.5))
    parser.add_argument("--random-base-z", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    return parser


def _load_or_create_stage(input_usd: Path | None) -> Usd.Stage:
    import omni.usd

    if input_usd is not None:
        if not input_usd.exists():
            raise RuntimeError(f"Cannot open input USD (file does not exist): {input_usd}")
        usd_context = omni.usd.get_context()
        usd_context.disable_save_to_recent_files()
        opened = usd_context.open_stage(str(input_usd))
        usd_context.enable_save_to_recent_files()
        if not opened:
            raise RuntimeError(f"Cannot open input USD: {input_usd}")
    else:
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
        random.seed(args.seed)
        input_path = args.input_usd.expanduser().resolve() if args.input_usd else None
        output_path = args.output_usd.expanduser().resolve() if args.output_usd else None
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)

        stage = _load_or_create_stage(input_path)
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
        obstacle_count = _add_obstacles(stage, args.world_path, args)

        if output_path is not None:
            if input_path is not None and output_path == input_path:
                stage.GetRootLayer().Save()
            else:
                stage.GetRootLayer().Export(str(output_path))
            print(f"Saved edited scene to: {output_path}")
        else:
            print("No USD file written (--output-usd not provided). Scene is updated in current Isaac Lab stage.")

        print(
            "Summary:"
            f" ground=({args.ground_size[0]} x {args.ground_size[1]}),"
            f" obstacles_added={obstacle_count},"
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
