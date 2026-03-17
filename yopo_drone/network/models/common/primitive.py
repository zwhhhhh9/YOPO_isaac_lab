"""Lattice primitive definitions migrated from YOPO."""

from __future__ import annotations

import torch
from scipy.spatial.transform import Rotation as R

from .config import Config, make_config


class LatticePrimitive:
    """
    Grid index layout in image space (row-major, bottom-left origin):

        +---+---+---+
        | 8 | 7 | 6 |
        +---+---+---+
        | 5 | 4 | 3 |
        +---+---+---+
        | 2 | 1 | 0 |
        +---+---+---+
    """

    def __init__(self, config: Config | None = None):
        self.cfg = make_config(config)
        ratio = 1.0 if self.cfg["train"] else self.cfg["velocity"] / self.cfg["vel_max_train"]
        self.vel_max = ratio * self.cfg["vel_max_train"]
        self.acc_max = ratio * ratio * self.cfg["acc_max_train"]
        self.segment_time = self.cfg["sgm_time"] / ratio
        self.horizon_num = int(self.cfg["horizon_num"])
        self.vertical_num = int(self.cfg["vertical_num"])
        self.radio_num = int(self.cfg["radio_num"])
        self.traj_num = int(self.cfg["traj_num"])
        self.horizon_fov = float(self.cfg["horizon_camera_fov"])
        self.vertical_fov = float(self.cfg["vertical_camera_fov"])
        self.horizon_anchor_fov = float(self.cfg["horizon_anchor_fov"])
        self.vertical_anchor_fov = float(self.cfg["vertical_anchor_fov"])
        self.radio_range = float(self.cfg["radio_range"])

        if self.horizon_num == 1:
            direction_diff = 0.0
        else:
            direction_diff = (self.horizon_fov / 180.0 * torch.pi) / self.horizon_num

        if self.vertical_num == 1:
            altitude_diff = 0.0
        else:
            altitude_diff = (self.vertical_fov / 180.0 * torch.pi) / self.vertical_num
        radio_diff = self.radio_range / self.radio_num

        lattice_pos_list = []
        lattice_angle_list = []
        lattice_rbp_list = []

        for radio_index in range(self.radio_num):
            for vertical_index in range(self.vertical_num):
                for horizon_index in range(self.horizon_num):
                    search_radio = (radio_index + 1) * radio_diff
                    alpha = torch.tensor(
                        -direction_diff * (self.horizon_num - 1) / 2 + horizon_index * direction_diff,
                        dtype=torch.float32,
                    )
                    beta = torch.tensor(
                        -altitude_diff * (self.vertical_num - 1) / 2 + vertical_index * altitude_diff,
                        dtype=torch.float32,
                    )

                    pos_node = torch.tensor(
                        [
                            torch.cos(beta) * torch.cos(alpha) * search_radio,
                            torch.cos(beta) * torch.sin(alpha) * search_radio,
                            torch.sin(beta) * search_radio,
                        ],
                        dtype=torch.float32,
                    )
                    lattice_pos_list.append(pos_node)
                    lattice_angle_list.append(torch.tensor([alpha, beta], dtype=torch.float32))

                    rotation = R.from_euler("ZYX", [alpha.item(), -beta.item(), 0.0], degrees=False)
                    lattice_rbp_list.append(torch.tensor(rotation.as_matrix(), dtype=torch.float32))

        self.lattice_pos_node = torch.stack(lattice_pos_list)
        self.lattice_angle_node = torch.stack(lattice_angle_list)
        self.lattice_rbp_node = torch.stack(lattice_rbp_list)
        self.yaw_diff = 0.5 * self.horizon_anchor_fov / 180.0 * torch.pi
        self.pitch_diff = 0.5 * self.vertical_anchor_fov / 180.0 * torch.pi

    def get_state_lattice(self, lattice_id=None, *, device: torch.device | None = None) -> torch.Tensor:
        values = self.lattice_pos_node if lattice_id is None else self.lattice_pos_node[lattice_id]
        return values.to(device=device) if device is not None else values

    def get_angle_lattice(self, lattice_id=None, *, device: torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        values = self.lattice_angle_node if lattice_id is None else self.lattice_angle_node[lattice_id]
        if device is not None:
            values = values.to(device=device)
        return values[..., 0], values[..., 1]

    def get_rotation(self, lattice_id=None, *, device: torch.device | None = None) -> torch.Tensor:
        values = self.lattice_rbp_node if lattice_id is None else self.lattice_rbp_node[lattice_id]
        return values.to(device=device) if device is not None else values

    def convert_image_grid_lattice_id(self, action_id):
        return self.traj_num - action_id - 1

