#!/usr/bin/env python3
"""Plot planned/reference speed against actual flight speed from YOPO telemetry CSV."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

VISUALIZATIONS_DIR = Path(__file__).resolve().parent
PNG_OUTPUTS_DIR = VISUALIZATIONS_DIR / "png_outputs"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("telemetry_csv", type=Path, help="Telemetry CSV produced by eval_ego.py.")
    parser.add_argument("output_png", type=Path, help="Path to the generated PNG.")
    parser.add_argument(
        "--title",
        type=str,
        default="YOPO Planned Speed vs Actual Speed",
        help="Figure title.",
    )
    return parser.parse_args()


def _load_columns(csv_path: Path) -> dict[str, np.ndarray]:
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    if not rows:
        raise ValueError(f"No telemetry rows found in {csv_path}.")

    columns: dict[str, list[float]] = {}
    for key in rows[0].keys():
        values: list[float] = []
        for row in rows:
            raw = row.get(key, "")
            try:
                values.append(float(raw))
            except (TypeError, ValueError):
                values.append(float("nan"))
        columns[key] = values
    return {key: np.asarray(values, dtype=np.float64) for key, values in columns.items()}


def _with_timestamp_suffix(path: Path) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = path.suffix or ".png"
    return path.with_name(f"{path.stem}_{timestamp}{suffix}")


def _resolve_output_png(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if resolved.parent == VISUALIZATIONS_DIR:
        resolved = PNG_OUTPUTS_DIR / resolved.name
    return _with_timestamp_suffix(resolved)


def _reconstruct_reference_speed(data: dict[str, np.ndarray]) -> np.ndarray:
    time_s = data["timestamp"]
    target = np.column_stack((data["target_x"], data["target_y"], data["target_z"]))
    if len(time_s) < 2:
        return np.zeros_like(time_s)
    grad_x = np.gradient(target[:, 0], time_s, edge_order=1)
    grad_y = np.gradient(target[:, 1], time_s, edge_order=1)
    grad_z = np.gradient(target[:, 2], time_s, edge_order=1)
    return np.linalg.norm(np.column_stack((grad_x, grad_y, grad_z)), axis=1)


def main() -> int:
    args = _parse_args()
    telemetry_csv = args.telemetry_csv.expanduser().resolve()
    output_png = _resolve_output_png(args.output_png)
    output_png.parent.mkdir(parents=True, exist_ok=True)

    data = _load_columns(telemetry_csv)
    time_s = data["timestamp"]
    actual_speed = data["speed_xyz"]

    if "target_speed_xyz" in data and np.any(np.isfinite(data["target_speed_xyz"])):
        planned_speed = data["target_speed_xyz"]
        planned_label = "Planned/Reference speed"
    else:
        planned_speed = _reconstruct_reference_speed(data)
        planned_label = "Planned speed (reconstructed from target path)"

    actual_peak_idx = int(np.nanargmax(actual_speed))
    planned_peak_idx = int(np.nanargmax(planned_speed))

    finite_error = np.isfinite(actual_speed) & np.isfinite(planned_speed)
    speed_error = actual_speed[finite_error] - planned_speed[finite_error]
    rmse = float(np.sqrt(np.mean(np.square(speed_error)))) if speed_error.size > 0 else float("nan")

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax_speed, ax_error) = plt.subplots(
        2,
        1,
        figsize=(14, 8),
        sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_speed.plot(time_s, planned_speed, color="#1f77b4", linewidth=2.0, label=planned_label)
    ax_speed.plot(time_s, actual_speed, color="#d62728", linewidth=1.8, alpha=0.9, label="Actual flight speed")
    ax_speed.scatter(
        [time_s[planned_peak_idx]],
        [planned_speed[planned_peak_idx]],
        color="#1f77b4",
        s=36,
        zorder=5,
    )
    ax_speed.scatter(
        [time_s[actual_peak_idx]],
        [actual_speed[actual_peak_idx]],
        color="#d62728",
        s=36,
        zorder=5,
    )
    ax_speed.annotate(
        f"planned peak {planned_speed[planned_peak_idx]:.2f} m/s @ {time_s[planned_peak_idx]:.2f}s",
        xy=(time_s[planned_peak_idx], planned_speed[planned_peak_idx]),
        xytext=(8, 10),
        textcoords="offset points",
        color="#1f77b4",
        fontsize=9,
    )
    ax_speed.annotate(
        f"actual peak {actual_speed[actual_peak_idx]:.2f} m/s @ {time_s[actual_peak_idx]:.2f}s",
        xy=(time_s[actual_peak_idx], actual_speed[actual_peak_idx]),
        xytext=(8, -16),
        textcoords="offset points",
        color="#d62728",
        fontsize=9,
    )
    ax_speed.set_ylabel("Speed (m/s)")
    ax_speed.set_title(args.title)
    ax_speed.legend(loc="upper right")
    ax_speed.text(
        0.01,
        0.02,
        f"Telemetry: {telemetry_csv.name}\nRMSE(actual-planned) = {rmse:.3f} m/s",
        transform=ax_speed.transAxes,
        fontsize=9,
        va="bottom",
        ha="left",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )

    ax_error.plot(time_s, actual_speed - planned_speed, color="#2ca02c", linewidth=1.5)
    ax_error.axhline(0.0, color="#555555", linewidth=1.0, linestyle="--")
    ax_error.set_xlabel("Time (s)")
    ax_error.set_ylabel("Actual - Planned\n(m/s)")

    fig.tight_layout()
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved speed comparison plot to: {output_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
