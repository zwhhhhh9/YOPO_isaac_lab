#!/usr/bin/env python3
"""Random forest obstacle scene generator for Isaac Lab.

This mirrors the original YOPO random-forest idea in
`Simulator/src/src/maps.cpp::forest()`:
1. place tree centers on a jittered grid controlled by `tree_dist`
2. apply a random per-tree uniform scale
3. apply a small random roll/pitch and free yaw rotation

Instead of loading a PLY tree, this scene builds each tree from simple Isaac Lab
primitives so it can be spawned directly into the USD stage.
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class RandomForestSceneCfg:
    """Configuration for the YOPO-style random forest scene."""

    prim_path: str
    seed: int = 0
    size_x: float = 80.0
    size_y: float = 80.0
    tile_radius: int = 2
    tree_dist: float = 4.0
    scale_min: float = 0.5
    scale_max: float = 1.0
    tilt_deg_max: float = 10.0
    spawn_clearance_radius: float = 2.5
    spawn_clearance_center: tuple[float, float] = (0.0, 0.0)
    canopy_collision: bool = False
    trunk_radius_scale_min: float = 0.75
    trunk_radius_scale_max: float = 1.85
    canopy_scale_min: float = 0.75
    canopy_scale_max: float = 1.95
    base_trunk_radius: float = 0.18
    base_trunk_height: float = 5.5
    base_lower_canopy_radius: float = 1.20
    base_lower_canopy_height: float = 2.60
    base_upper_canopy_radius: float = 0.85
    base_upper_canopy_height: float = 1.80


def add_random_forest_arguments(parser: argparse.ArgumentParser) -> None:
    """Register CLI flags used by drone_env_editor."""
    group = parser.add_argument_group("random forest")
    group.add_argument(
        "--add-random-forest",
        action="store_true",
        help="Spawn a YOPO-style random forest obstacle scene.",
    )
    group.add_argument(
        "--random-forest-prim-path",
        type=str,
        default="",
        help="Root prim path for the generated forest. Empty means <world-path>/Obstacles/RandomForest.",
    )
    group.add_argument(
        "--random-forest-seed",
        type=int,
        default=0,
        help="Deterministic seed for tree placement and randomization.",
    )
    group.add_argument(
        "--random-forest-size-x",
        type=float,
        default=80.0,
        help="Single forest tile width along X in meters.",
    )
    group.add_argument(
        "--random-forest-size-y",
        type=float,
        default=80.0,
        help="Single forest tile width along Y in meters.",
    )
    group.add_argument(
        "--random-forest-tile-radius",
        type=int,
        default=2,
        help="Static repeat radius in tiles. Total tiles = (2r+1)^2.",
    )
    group.add_argument(
        "--random-forest-tree-dist",
        type=float,
        default=4.0,
        help="Cell size / average spacing between tree centers, matching YOPO's tree_dist idea.",
    )
    group.add_argument(
        "--random-forest-clearance-radius",
        type=float,
        default=2.5,
        help="Keep this radius around the robot start position free of trees.",
    )
    group.add_argument(
        "--random-forest-scale-min",
        type=float,
        default=0.5,
        help="Minimum per-tree uniform scale.",
    )
    group.add_argument(
        "--random-forest-scale-max",
        type=float,
        default=1.0,
        help="Maximum per-tree uniform scale.",
    )
    group.add_argument(
        "--random-forest-tilt-deg-max",
        type=float,
        default=10.0,
        help="Maximum absolute roll/pitch tilt in degrees for each tree.",
    )
    group.add_argument(
        "--random-forest-canopy-collision",
        action="store_true",
        help="Enable collision on the tree canopy as well as the trunk.",
    )
    group.add_argument(
        "--random-forest-trunk-radius-scale-min",
        type=float,
        default=0.75,
        help="Minimum per-tree multiplier applied to trunk radius only.",
    )
    group.add_argument(
        "--random-forest-trunk-radius-scale-max",
        type=float,
        default=1.85,
        help="Maximum per-tree multiplier applied to trunk radius only.",
    )
    group.add_argument(
        "--random-forest-canopy-scale-min",
        type=float,
        default=0.75,
        help="Minimum per-tree multiplier applied to canopy size.",
    )
    group.add_argument(
        "--random-forest-canopy-scale-max",
        type=float,
        default=1.95,
        help="Maximum per-tree multiplier applied to canopy size.",
    )


def build_random_forest_cfg_from_args(args: argparse.Namespace) -> RandomForestSceneCfg:
    """Build a forest config from drone_env_editor CLI arguments."""
    prim_path = str(args.random_forest_prim_path).strip() or f"{args.world_path}/Obstacles/RandomForest"
    robot_init_pos = getattr(args, "robot_init_pos", (0.0, 0.0, 0.0))
    clearance_center = (float(robot_init_pos[0]), float(robot_init_pos[1]))
    cfg = RandomForestSceneCfg(
        prim_path=prim_path,
        seed=int(args.random_forest_seed),
        size_x=float(args.random_forest_size_x),
        size_y=float(args.random_forest_size_y),
        tile_radius=int(args.random_forest_tile_radius),
        tree_dist=float(args.random_forest_tree_dist),
        scale_min=float(args.random_forest_scale_min),
        scale_max=float(args.random_forest_scale_max),
        tilt_deg_max=float(args.random_forest_tilt_deg_max),
        spawn_clearance_radius=float(args.random_forest_clearance_radius),
        spawn_clearance_center=clearance_center,
        canopy_collision=bool(args.random_forest_canopy_collision),
        trunk_radius_scale_min=float(args.random_forest_trunk_radius_scale_min),
        trunk_radius_scale_max=float(args.random_forest_trunk_radius_scale_max),
        canopy_scale_min=float(args.random_forest_canopy_scale_min),
        canopy_scale_max=float(args.random_forest_canopy_scale_max),
    )
    _validate_cfg(cfg)
    return cfg


def add_random_forest_scene(*, sim_utils: Any, cfg: RandomForestSceneCfg) -> dict[str, Any]:
    """Populate the current stage with a static tiled random forest."""
    _validate_cfg(cfg)

    stage, UsdGeom = _get_stage_and_usdgeom()
    _clear_path(stage, cfg.prim_path)
    UsdGeom.Xform.Define(stage, cfg.prim_path)

    trunk_cfg, lower_canopy_cfg, upper_canopy_cfg = _build_tree_cfgs(sim_utils, cfg)

    tile_count = 0
    total_tree_count = 0
    for tile_x in range(-cfg.tile_radius, cfg.tile_radius + 1):
        for tile_y in range(-cfg.tile_radius, cfg.tile_radius + 1):
            total_tree_count += _add_forest_tile(
                cfg=cfg,
                stage=stage,
                UsdGeom=UsdGeom,
                tile_x=tile_x,
                tile_y=tile_y,
                trunk_cfg=trunk_cfg,
                lower_canopy_cfg=lower_canopy_cfg,
                upper_canopy_cfg=upper_canopy_cfg,
            )
            tile_count += 1

    rows = max(int(cfg.size_x / cfg.tree_dist), 1)
    cols = max(int(cfg.size_y / cfg.tree_dist), 1)
    total_span_x = cfg.size_x * (2 * cfg.tile_radius + 1)
    total_span_y = cfg.size_y * (2 * cfg.tile_radius + 1)
    summary = {
        "prim_path": cfg.prim_path,
        "tree_count": total_tree_count,
        "tile_count": tile_count,
        "rows": rows,
        "cols": cols,
        "tile_size_x": cfg.size_x,
        "tile_size_y": cfg.size_y,
        "total_span_x": total_span_x,
        "total_span_y": total_span_y,
        "tree_dist": cfg.tree_dist,
        "clearance_radius": cfg.spawn_clearance_radius,
    }
    print(
        "Random forest added:"
        f" prim={cfg.prim_path},"
        f" tiles={tile_count},"
        f" trees={total_tree_count},"
        f" per_tile_grid=({rows}x{cols}),"
        f" tile_size=({cfg.size_x}x{cfg.size_y}),"
        f" total_span=({total_span_x}x{total_span_y}),"
        f" tree_dist={cfg.tree_dist},"
        f" clearance_radius={cfg.spawn_clearance_radius},"
        f" trunk_radius_scale=({cfg.trunk_radius_scale_min},{cfg.trunk_radius_scale_max}),"
        f" canopy_scale=({cfg.canopy_scale_min},{cfg.canopy_scale_max})"
    , flush=True)
    return summary


def _build_tree_cfgs(sim_utils: Any, cfg: RandomForestSceneCfg) -> tuple[Any, Any, Any]:
    trunk_cfg = sim_utils.CylinderCfg(
        radius=cfg.base_trunk_radius,
        height=cfg.base_trunk_height,
        axis="Z",
        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.35, 0.23, 0.12), roughness=0.95),
    )
    lower_canopy_cfg = sim_utils.ConeCfg(
        radius=cfg.base_lower_canopy_radius,
        height=cfg.base_lower_canopy_height,
        axis="Z",
        collision_props=_build_canopy_collision_cfg(sim_utils, cfg.canopy_collision),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.12, 0.34, 0.16), roughness=0.9),
    )
    upper_canopy_cfg = sim_utils.ConeCfg(
        radius=cfg.base_upper_canopy_radius,
        height=cfg.base_upper_canopy_height,
        axis="Z",
        collision_props=_build_canopy_collision_cfg(sim_utils, cfg.canopy_collision),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.16, 0.40, 0.18), roughness=0.85),
    )
    return trunk_cfg, lower_canopy_cfg, upper_canopy_cfg


def _add_forest_tile(
    *,
    cfg: RandomForestSceneCfg,
    stage: Any,
    UsdGeom: Any,
    tile_x: int,
    tile_y: int,
    trunk_cfg: Any,
    lower_canopy_cfg: Any,
    upper_canopy_cfg: Any,
) -> int:
    tile_path = f"{cfg.prim_path}/Tile_{_tile_name_part(tile_x)}_{_tile_name_part(tile_y)}"
    tile_center_x = float(tile_x) * cfg.size_x
    tile_center_y = float(tile_y) * cfg.size_y
    tile_root = UsdGeom.Xform.Define(stage, tile_path)
    _set_transform(UsdGeom.Xformable(tile_root.GetPrim()), translate=(tile_center_x, tile_center_y, 0.0))

    rng = random.Random(_tile_seed(cfg.seed, tile_x, tile_y))
    positions = _generate_tile_positions(cfg, rng, tile_center_x=tile_center_x, tile_center_y=tile_center_y)

    for tree_index, (local_x, local_y) in enumerate(positions):
        tree_path = f"{tile_path}/Tree_{tree_index:04d}"
        tree_root = UsdGeom.Xform.Define(stage, tree_path)
        scale_factor = rng.uniform(cfg.scale_min, cfg.scale_max)
        trunk_radius_scale = rng.uniform(cfg.trunk_radius_scale_min, cfg.trunk_radius_scale_max)
        lower_canopy_scale = rng.uniform(cfg.canopy_scale_min, cfg.canopy_scale_max)
        upper_canopy_scale = rng.uniform(cfg.canopy_scale_min, min(cfg.canopy_scale_max, lower_canopy_scale))
        roll = rng.uniform(-cfg.tilt_deg_max, cfg.tilt_deg_max)
        pitch = rng.uniform(-cfg.tilt_deg_max, cfg.tilt_deg_max)
        yaw = rng.uniform(0.0, 360.0)
        trunk_height = cfg.base_trunk_height * scale_factor
        trunk_radius = cfg.base_trunk_radius * scale_factor * trunk_radius_scale
        lower_canopy_height = cfg.base_lower_canopy_height * scale_factor * lower_canopy_scale
        lower_canopy_radius = cfg.base_lower_canopy_radius * scale_factor * lower_canopy_scale
        upper_canopy_height = cfg.base_upper_canopy_height * scale_factor * upper_canopy_scale
        upper_canopy_radius = cfg.base_upper_canopy_radius * scale_factor * upper_canopy_scale
        trunk_tree_cfg = trunk_cfg.copy().replace(radius=trunk_radius, height=trunk_height)
        lower_canopy_tree_cfg = lower_canopy_cfg.copy().replace(
            radius=lower_canopy_radius,
            height=lower_canopy_height,
        )
        upper_canopy_tree_cfg = upper_canopy_cfg.copy().replace(
            radius=upper_canopy_radius,
            height=upper_canopy_height,
        )
        _set_transform(
            UsdGeom.Xformable(tree_root.GetPrim()),
            translate=(local_x, local_y, 0.0),
            rotate_xyz=(roll, pitch, yaw),
        )

        trunk_tree_cfg.func(
            f"{tree_path}/Trunk",
            trunk_tree_cfg,
            translation=(0.0, 0.0, trunk_height * 0.5),
        )
        lower_canopy_tree_cfg.func(
            f"{tree_path}/LowerCanopy",
            lower_canopy_tree_cfg,
            translation=(0.0, 0.0, trunk_height + lower_canopy_height * 0.35),
        )
        upper_canopy_tree_cfg.func(
            f"{tree_path}/UpperCanopy",
            upper_canopy_tree_cfg,
            translation=(0.0, 0.0, trunk_height + lower_canopy_height * 0.90),
        )

    return len(positions)


def _build_canopy_collision_cfg(sim_utils: Any, enabled: bool) -> Any | None:
    if not enabled:
        return None
    return sim_utils.CollisionPropertiesCfg(collision_enabled=True)


def _validate_cfg(cfg: RandomForestSceneCfg) -> None:
    if not cfg.prim_path.startswith("/"):
        raise ValueError("Random forest prim_path must be an absolute USD path.")
    if cfg.size_x <= 0.0 or cfg.size_y <= 0.0:
        raise ValueError("Random forest size_x/size_y must be positive.")
    if cfg.tile_radius < 0:
        raise ValueError("Random forest tile_radius must be >= 0.")
    if cfg.tree_dist <= 0.0:
        raise ValueError("Random forest tree_dist must be positive.")
    if cfg.scale_min <= 0.0 or cfg.scale_max <= 0.0:
        raise ValueError("Random forest scale_min/scale_max must be positive.")
    if cfg.scale_min > cfg.scale_max:
        raise ValueError("Random forest scale_min must be <= scale_max.")
    if cfg.trunk_radius_scale_min <= 0.0 or cfg.trunk_radius_scale_max <= 0.0:
        raise ValueError("Random forest trunk_radius_scale_min/max must be positive.")
    if cfg.trunk_radius_scale_min > cfg.trunk_radius_scale_max:
        raise ValueError("Random forest trunk_radius_scale_min must be <= trunk_radius_scale_max.")
    if cfg.canopy_scale_min <= 0.0 or cfg.canopy_scale_max <= 0.0:
        raise ValueError("Random forest canopy_scale_min/max must be positive.")
    if cfg.canopy_scale_min > cfg.canopy_scale_max:
        raise ValueError("Random forest canopy_scale_min must be <= canopy_scale_max.")
    if cfg.tilt_deg_max < 0.0:
        raise ValueError("Random forest tilt_deg_max must be non-negative.")
    if cfg.spawn_clearance_radius < 0.0:
        raise ValueError("Random forest spawn_clearance_radius must be non-negative.")


def _generate_tile_positions(
    cfg: RandomForestSceneCfg,
    rng: random.Random,
    *,
    tile_center_x: float,
    tile_center_y: float,
) -> list[tuple[float, float]]:
    rows = max(int(cfg.size_x / cfg.tree_dist), 1)
    cols = max(int(cfg.size_y / cfg.tree_dist), 1)
    x_offset = cfg.size_x / 2.0
    y_offset = cfg.size_y / 2.0
    clearance_x, clearance_y = cfg.spawn_clearance_center
    positions: list[tuple[float, float]] = []

    for row_idx in range(rows):
        for col_idx in range(cols):
            local_x = row_idx * cfg.tree_dist + rng.uniform(0.0, cfg.tree_dist) - x_offset
            local_y = col_idx * cfg.tree_dist + rng.uniform(0.0, cfg.tree_dist) - y_offset
            world_x = tile_center_x + local_x
            world_y = tile_center_y + local_y
            dx = world_x - clearance_x
            dy = world_y - clearance_y
            if dx * dx + dy * dy < cfg.spawn_clearance_radius * cfg.spawn_clearance_radius:
                continue
            positions.append((local_x, local_y))

    return positions


def _tile_name_part(value: int) -> str:
    if value >= 0:
        return f"p{value}"
    return f"n{abs(value)}"


def _tile_seed(base_seed: int, tile_x: int, tile_y: int) -> int:
    mask = (1 << 64) - 1
    seed = int(base_seed) & mask
    seed ^= ((tile_x & mask) * 0x9E3779B185EBCA87) & mask
    seed ^= ((tile_y & mask) * 0xC2B2AE3D27D4EB4F) & mask
    return seed


def _get_stage_and_usdgeom():
    import omni.usd
    from pxr import UsdGeom

    stage = omni.usd.get_context().get_stage()
    if stage is None:
        raise RuntimeError("Cannot obtain current USD stage from Isaac Sim.")
    return stage, UsdGeom


def _clear_path(stage: Any, path: str) -> None:
    prim = stage.GetPrimAtPath(path)
    if prim and prim.IsValid():
        stage.RemovePrim(path)


def _get_or_create_op(xformable: Any, op_type: Any, UsdGeom: Any) -> Any:
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
    xformable: Any,
    *,
    translate: tuple[float, float, float] | None = None,
    rotate_xyz: tuple[float, float, float] | None = None,
    scale: tuple[float, float, float] | None = None,
) -> None:
    from pxr import Gf, UsdGeom

    if translate is not None:
        _get_or_create_op(xformable, UsdGeom.XformOp.TypeTranslate, UsdGeom).Set(Gf.Vec3d(*translate))
    if rotate_xyz is not None:
        _get_or_create_op(xformable, UsdGeom.XformOp.TypeRotateXYZ, UsdGeom).Set(Gf.Vec3d(*rotate_xyz))
    if scale is not None:
        _get_or_create_op(xformable, UsdGeom.XformOp.TypeScale, UsdGeom).Set(Gf.Vec3d(*scale))
