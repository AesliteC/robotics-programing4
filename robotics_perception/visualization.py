"""Visualization helpers for calibration homework."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np


def draw_checkerboard_corners(
    image: np.ndarray,
    checkerboard_size: Tuple[int, int],
    corners: np.ndarray,
    success: bool = True,
) -> np.ndarray:
    """Draw detected checkerboard corners on a copy of the input image."""
    vis = image.copy()
    cv2.drawChessboardCorners(vis, checkerboard_size, corners.reshape(-1, 1, 2), success)
    return vis


def draw_horizontal_epipolar_lines(
    left_img: np.ndarray,
    right_img: np.ndarray,
    step: int = 40,
) -> np.ndarray:
    """Concatenate rectified stereo images and draw horizontal reference lines."""
    if left_img.ndim == 2:
        left_vis = cv2.cvtColor(left_img, cv2.COLOR_GRAY2BGR)
    else:
        left_vis = left_img.copy()
    if right_img.ndim == 2:
        right_vis = cv2.cvtColor(right_img, cv2.COLOR_GRAY2BGR)
    else:
        right_vis = right_img.copy()

    h = min(left_vis.shape[0], right_vis.shape[0])
    left_vis = left_vis[:h]
    right_vis = right_vis[:h]
    canvas = np.hstack([left_vis, right_vis])

    for y in range(0, h, step):
        cv2.line(canvas, (0, y), (canvas.shape[1], y), (0, 255, 0), 1)
    return canvas


def save_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), image)


def save_disparity_colormap(path: str | Path, disparity: np.ndarray) -> None:
    """Save disparity as a visible colormap image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(disparity)
    if valid.any():
        disp_min = float(np.nanpercentile(disparity[valid], 2))
        disp_max = float(np.nanpercentile(disparity[valid], 98))
        norm = np.clip((disparity - disp_min) / (disp_max - disp_min + 1e-6), 0, 1)
    else:
        norm = np.zeros_like(disparity, dtype=np.float32)
    plt.imsave(path, norm, cmap="magma")


def save_depth_colormap(path: str | Path, depth: np.ndarray, max_depth: float | None = None) -> None:
    """Save depth as a visible colormap image."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    valid = np.isfinite(depth) & (depth > 0)
    if max_depth is None and valid.any():
        max_depth = float(np.nanpercentile(depth[valid], 95))
    if max_depth is None or max_depth <= 0:
        max_depth = 1.0
    norm = np.zeros_like(depth, dtype=np.float32)
    norm[valid] = np.clip(depth[valid] / max_depth, 0, 1)
    plt.imsave(path, norm, cmap="viridis")
