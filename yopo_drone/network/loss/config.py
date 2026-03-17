"""Python-native configuration for the migrated YOPO loss package."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


@dataclass
class YopoLossConfig:
    velocity: float = 6.0
    vel_max_train: float = 6.0
    acc_max_train: float = 6.0
    wg: float = 0.15
    ws: float = 10.0
    wa: float = 1.0
    wc: float = 1.5
    dataset_path: str = "yopo_drone/network/data_train"
    image_height: int = 96
    image_width: int = 160
    horizon_num: int = 5
    vertical_num: int = 3
    horizon_camera_fov: float = 90.0
    vertical_camera_fov: float = 60.0
    horizon_anchor_fov: float = 30.0
    vertical_anchor_fov: float = 30.0
    radio_range: float = 5.0
    radio_num: int = 1
    d0: float = 1.6
    r: float = 0.8
    vx_mean_unit: float = 0.4
    vy_mean_unit: float = 0.0
    vz_mean_unit: float = 0.0
    vx_std_unit: float = 2.0
    vy_std_unit: float = 0.45
    vz_std_unit: float = 0.3
    ax_mean_unit: float = 0.0
    ay_mean_unit: float = 0.0
    az_mean_unit: float = 0.0
    ax_std_unit: float = 0.5
    ay_std_unit: float = 0.5
    az_std_unit: float = 0.3
    goal_pitch_std: float = 10.0
    goal_yaw_std: float = 20.0
    goal_sampling_mode: str = "uniform_full_yaw"
    goal_yaw_uniform_min_deg: float = -180.0
    goal_yaw_uniform_max_deg: float = 180.0
    goal_pitch_uniform_min_deg: float = -30.0
    goal_pitch_uniform_max_deg: float = 30.0
    map_expand_min: tuple[float, float, float] = (0.0, 0.0, 0.2)
    map_expand_max: tuple[float, float, float] = (0.0, 0.0, 6.0)
    train: bool = True

    @property
    def goal_length(self) -> float:
        return 2.0 * self.radio_range

    @property
    def sgm_time(self) -> float:
        return 2.0 * self.radio_range / self.vel_max_train

    @property
    def traj_num(self) -> int:
        return self.horizon_num * self.vertical_num * self.radio_num

    def dataset_root(self) -> Path:
        env_override = os.environ.get("YOPO_LOSS_DATASET_PATH")
        raw_path = Path(env_override) if env_override else Path(self.dataset_path)
        if raw_path.is_absolute():
            return raw_path
        return (_project_root() / raw_path).resolve()


class Config:
    """Dictionary-style wrapper kept close to the original YOPO interface."""

    _DERIVED_KEYS = {"goal_length", "sgm_time", "traj_num"}

    def __init__(self, values: YopoLossConfig | None = None):
        self._values = values or YopoLossConfig()

    def __getitem__(self, key: str) -> Any:
        if key in self._DERIVED_KEYS:
            return getattr(self._values, key)
        if hasattr(self._values, key):
            return getattr(self._values, key)
        raise KeyError(key)

    def __setitem__(self, key: str, value: Any) -> None:
        if key in self._DERIVED_KEYS:
            raise KeyError(f"{key} is derived from other configuration values.")
        if not hasattr(self._values, key):
            raise KeyError(key)
        setattr(self._values, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def update(self, values: Mapping[str, Any]) -> None:
        for key, value in values.items():
            if key in self._DERIVED_KEYS or key == "dataset_root":
                continue
            self[key] = value

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self._values)
        data["goal_length"] = self._values.goal_length
        data["sgm_time"] = self._values.sgm_time
        data["traj_num"] = self._values.traj_num
        data["dataset_root"] = str(self._values.dataset_root())
        return data

    def dataset_root(self) -> Path:
        return self._values.dataset_root()


def ensure_config(config: Config | YopoLossConfig | Mapping[str, Any] | None = None) -> Config:
    if config is None:
        return cfg
    if isinstance(config, Config):
        return config
    if isinstance(config, YopoLossConfig):
        return Config(config)
    resolved = Config()
    resolved.update(config)
    return resolved


cfg = Config()
