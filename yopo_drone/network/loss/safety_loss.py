from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as f
from scipy.ndimage import distance_transform_edt

from .config import ensure_config

_PLY_TYPE_TO_NUMPY = {
    "char": "i1",
    "uchar": "u1",
    "short": "i2",
    "ushort": "u2",
    "int": "i4",
    "uint": "u4",
    "float": "f4",
    "double": "f8",
}


def _read_ply_points(path: Path) -> np.ndarray:
    with path.open("rb") as ply_file:
        header_lines: list[str] = []
        while True:
            raw_line = ply_file.readline()
            if not raw_line:
                raise ValueError(f"Invalid PLY file without end_header: {path}")
            line = raw_line.decode("ascii").strip()
            header_lines.append(line)
            if line == "end_header":
                break

        fmt: str | None = None
        vertex_count: int | None = None
        vertex_properties: list[tuple[str, str]] = []
        current_element: str | None = None
        for line in header_lines:
            parts = line.split()
            if not parts:
                continue
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                current_element = parts[1]
                if current_element == "vertex":
                    vertex_count = int(parts[2])
                    vertex_properties = []
            elif parts[0] == "property" and current_element == "vertex":
                if parts[1] == "list":
                    raise ValueError(f"PLY list properties are not supported for vertex data: {path}")
                vertex_properties.append((parts[1], parts[2]))

        if fmt is None or vertex_count is None:
            raise ValueError(f"Failed to parse PLY header: {path}")
        if not vertex_properties:
            raise ValueError(f"PLY file does not contain vertex properties: {path}")

        property_names = [name for _, name in vertex_properties]
        for axis in ("x", "y", "z"):
            if axis not in property_names:
                raise ValueError(f"PLY vertex data is missing '{axis}' in {path}")

        if fmt == "ascii":
            data = np.loadtxt(ply_file, dtype=np.float64, max_rows=vertex_count)
            if data.ndim == 1:
                data = data.reshape(1, -1)
            indices = [property_names.index(axis) for axis in ("x", "y", "z")]
            points = data[:, indices]
        elif fmt in {"binary_little_endian", "binary_big_endian"}:
            endian = "<" if fmt == "binary_little_endian" else ">"
            dtype = np.dtype([(name, endian + _PLY_TYPE_TO_NUMPY[data_type]) for data_type, name in vertex_properties])
            data = np.fromfile(ply_file, dtype=dtype, count=vertex_count)
            points = np.stack([data["x"], data["y"], data["z"]], axis=1)
        else:
            raise ValueError(f"Unsupported PLY format '{fmt}' in {path}")

    return np.asarray(points, dtype=np.float32).reshape(-1, 3)


class SafetyLoss(nn.Module):
    def __init__(self, l_matrix: th.Tensor, config=None, dataset_root: str | Path | None = None):
        super().__init__()
        self.cfg = ensure_config(config)
        self.traj_num = int(self.cfg["traj_num"])
        self.map_expand_min = np.asarray(self.cfg["map_expand_min"], dtype=np.float32)
        self.map_expand_max = np.asarray(self.cfg["map_expand_max"], dtype=np.float32)
        self.d0 = float(self.cfg["d0"])
        self.r = float(self.cfg["r"])

        self._l = l_matrix
        self.sgm_time = float(self.cfg["sgm_time"])
        self.eval_points = 30
        self.device = self._l.device
        self.time_integral = True

        self.voxel_size = 0.2
        self.min_bounds: th.Tensor | None = None
        self.max_bounds: th.Tensor | None = None
        self.sdf_shapes: th.Tensor | None = None
        self.dataset_root = Path(dataset_root).expanduser().resolve() if dataset_root else self.cfg.dataset_root()

        print(f"Building ESDF map from {self.dataset_root} ...")
        self.sdf_maps = self.get_sdf_from_pointclouds(self.dataset_root)
        print("Map built!")

    def forward(self, df: th.Tensor, dp: th.Tensor, map_id: th.Tensor | np.ndarray | list[int] | int) -> th.Tensor:
        batch_size = int(dp.shape[0])
        map_id_tensor = self._expand_map_id(map_id, batch_size)
        l_matrix = self._l.unsqueeze(0).expand(batch_size, -1, -1)
        coefficients = self.get_coefficient_from_derivative(dp, df, l_matrix)

        dt = self.sgm_time / self.eval_points
        t_list = th.linspace(dt, self.sgm_time, self.eval_points, device=self.device)
        t_list = t_list.view(1, -1, 1).expand(batch_size, -1, -1)

        pos_coeff = self.get_position_from_coeff(coefficients, t_list)
        cost, _ = self.get_distance_cost(pos_coeff, map_id_tensor)

        if self.time_integral:
            return cost.mean(dim=-1)

        vel_coeff = self.get_velocity_from_coeff(coefficients, t_list).norm(dim=-1)
        line_integral_cost = (cost * vel_coeff * dt).sum(dim=1)
        line_length = (vel_coeff * dt).sum(dim=1)
        return line_integral_cost / line_length.clamp_min(1e-6)

    def get_distance_cost(self, pos: th.Tensor, map_id: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        batch_size, num_points, _ = pos.shape

        sdf_maps, local_origin, local_shape = self.get_batch_sdf(pos, map_id)
        grid = (pos - local_origin.unsqueeze(1)) / self.voxel_size
        grid_point = 2.0 * grid / (local_shape - 1).unsqueeze(1) - 1.0

        grid_point = grid_point.view(batch_size, 1, 1, num_points, 3)
        grid_point = th.clamp(grid_point, min=-0.99, max=0.99)

        dist_query = f.grid_sample(
            sdf_maps,
            grid_point,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        dist_query = dist_query.view(batch_size, num_points)
        return self.cost_function(dist_query), dist_query

    def cost_function(self, distance: th.Tensor) -> th.Tensor:
        return th.exp(-(distance - self.d0) / self.r)

    def get_coefficient_from_derivative(self, dp: th.Tensor, df: th.Tensor, l_matrix: th.Tensor) -> th.Tensor:
        coefficient = th.zeros(dp.shape[0], 18, device=self.device)
        for axis in range(3):
            derivative = th.cat([df[:, axis, :], dp[:, axis, :]], dim=1).unsqueeze(-1)
            coeff_axis = (l_matrix @ derivative).squeeze(-1)
            coefficient[:, 6 * axis : 6 * (axis + 1)] = coeff_axis
        return coefficient

    def get_position_from_coeff(self, coeff: th.Tensor, t: th.Tensor) -> th.Tensor:
        t_power = th.stack([th.ones_like(t), t, t**2, t**3, t**4, t**5], dim=-1).squeeze(-2)

        coeff_x = coeff[:, 0:6]
        coeff_y = coeff[:, 6:12]
        coeff_z = coeff[:, 12:18]

        x = th.sum(t_power * coeff_x.unsqueeze(1), dim=-1)
        y = th.sum(t_power * coeff_y.unsqueeze(1), dim=-1)
        z = th.sum(t_power * coeff_z.unsqueeze(1), dim=-1)
        return th.stack([x, y, z], dim=-1)

    def get_velocity_from_coeff(self, coeff: th.Tensor, t: th.Tensor) -> th.Tensor:
        t_power = th.stack([th.ones_like(t), 2 * t, 3 * t**2, 4 * t**3, 5 * t**4], dim=-1).squeeze(-2)

        coeff_x = coeff[:, 1:6]
        coeff_y = coeff[:, 7:12]
        coeff_z = coeff[:, 13:18]

        vx = th.sum(t_power * coeff_x.unsqueeze(1), dim=-1)
        vy = th.sum(t_power * coeff_y.unsqueeze(1), dim=-1)
        vz = th.sum(t_power * coeff_z.unsqueeze(1), dim=-1)
        return th.stack([vx, vy, vz], dim=-1)

    def get_batch_sdf(self, pos: th.Tensor, map_id: th.Tensor) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        if self.min_bounds is None or self.sdf_shapes is None:
            raise RuntimeError("SDF maps are not initialized.")

        min_bounds = self.min_bounds[map_id]
        sdf_shapes = self.sdf_shapes[map_id]

        min_pos = pos.amin(dim=1)
        max_pos = pos.amax(dim=1)
        min_indices = ((min_pos - min_bounds) / self.voxel_size).int()
        max_indices = ((max_pos - min_bounds) / self.voxel_size).int()
        spans = max_indices - min_indices
        max_spans = spans.amax(dim=0)
        centers = (min_indices + max_indices) // 2
        min_indices = centers - max_spans // 2 - 5
        max_indices = centers + max_spans // 2 + 5

        new_min_indices = min_indices.clamp(min=0)
        underflow_amount = new_min_indices - min_indices
        min_indices = new_min_indices
        max_indices = max_indices + underflow_amount

        new_max_indices = th.minimum(max_indices, sdf_shapes.int())
        overflow_amount = max_indices - new_max_indices
        max_indices = new_max_indices
        min_indices = min_indices - overflow_amount

        if (min_indices < 0).any():
            min_underflow = th.minimum(min_indices, th.zeros_like(min_indices))
            shift = (-min_underflow).max(dim=0).values
            min_indices = min_indices + shift

        sdf_maps = th.stack(
            [
                self.sdf_maps[map_index][
                    0,
                    :,
                    min_index[2] : max_index[2],
                    min_index[1] : max_index[1],
                    min_index[0] : max_index[0],
                ]
                for map_index, min_index, max_index in zip(
                    map_id.tolist(),
                    min_indices.tolist(),
                    max_indices.tolist(),
                )
            ]
        )
        local_origin = min_indices * self.voxel_size + min_bounds
        local_shape = max_indices - min_indices
        return sdf_maps, local_origin, local_shape

    def get_sdf_from_pointclouds(self, path: Path) -> list[th.Tensor]:
        sorted_files = self.read_sorted_ply_files(path)
        if not sorted_files:
            raise FileNotFoundError(
                f"No pointcloud PLY files were found under {path}. "
                "Expected pointcloud.ply, pointcloud_<n>.ply, or pointcloud-<n>.ply."
            )

        sdf_maps: list[th.Tensor] = []
        min_bounds: list[np.ndarray] = []
        max_bounds: list[np.ndarray] = []
        sdf_shapes: list[tuple[int, int, int]] = []

        for file_path in sorted_files:
            points = _read_ply_points(file_path)
            min_bound = points.min(axis=0) - self.map_expand_min
            max_bound = points.max(axis=0) + self.map_expand_max
            print(
                f"    {file_path.name}: "
                f"x=({points[:, 0].min():.2f}, {points[:, 0].max():.2f}), "
                f"y=({points[:, 1].min():.2f}, {points[:, 1].max():.2f}), "
                f"z=({points[:, 2].min():.2f}, {points[:, 2].max():.2f})"
            )

            sdf_shape = np.ceil((max_bound - min_bound) / self.voxel_size).astype(int)
            voxel_indices = ((points - min_bound) / self.voxel_size).astype(int)
            valid_mask = np.all((voxel_indices >= 0) & (voxel_indices < sdf_shape), axis=1)
            voxel_indices = voxel_indices[valid_mask]

            occupancy = np.zeros(sdf_shape, dtype=np.uint8)
            occupancy[tuple(voxel_indices.T)] = 1

            obstacle_mask = occupancy == 1
            free_mask = occupancy == 0

            dist_to_obstacle = distance_transform_edt(free_mask) * self.voxel_size
            dist_inside_obstacle = distance_transform_edt(obstacle_mask) * self.voxel_size
            dist_to_obstacle[obstacle_mask] = -dist_inside_obstacle[obstacle_mask]

            sdf_tensor = (
                th.from_numpy(dist_to_obstacle)
                .float()
                .unsqueeze(0)
                .unsqueeze(0)
                .permute(0, 1, 4, 3, 2)
                .to(self.device)
            )

            sdf_maps.append(sdf_tensor)
            sdf_shapes.append(tuple(sdf_tensor.shape[-3:][::-1]))
            min_bounds.append(min_bound)
            max_bounds.append(max_bound)

        self.min_bounds = th.tensor(np.asarray(min_bounds), device=self.device).float()
        self.max_bounds = th.tensor(np.asarray(max_bounds), device=self.device).float()
        self.sdf_shapes = th.tensor(np.asarray(sdf_shapes), device=self.device).float()
        return sdf_maps

    def read_sorted_ply_files(self, path: Path) -> list[Path]:
        path = Path(path).expanduser().resolve()
        if path.is_file():
            return [path]

        candidates: set[Path] = set()
        for pattern in ("pointcloud.ply", "pointcloud_*.ply", "pointcloud-*.ply"):
            candidates.update(path.rglob(pattern))

        def sort_key(file_path: Path) -> tuple[str, int, str]:
            rel_parent = str(file_path.parent.relative_to(path)) if file_path.parent != path else "."
            match = re.search(r"(\d+)$", file_path.stem)
            index = int(match.group(1)) if match else -1
            return rel_parent, index, file_path.name

        return sorted(candidates, key=sort_key)

    def _expand_map_id(self, map_id: th.Tensor | np.ndarray | list[int] | int, batch_size: int) -> th.Tensor:
        tensor = th.as_tensor(map_id, device=self.device).reshape(-1).long()
        if tensor.numel() == batch_size:
            return tensor
        if tensor.numel() == 1:
            return tensor.expand(batch_size)
        if batch_size % tensor.numel() == 0:
            repeat = batch_size // tensor.numel()
            return tensor.repeat_interleave(repeat)
        raise ValueError(f"Cannot expand map_id with shape {tuple(tensor.shape)} to batch size {batch_size}.")
