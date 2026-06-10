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
    norm = np.zeros_like(disparity, dtype=np.float32)
    if valid.any():
        disp_min = float(np.nanpercentile(disparity[valid], 2))
        disp_max = float(np.nanpercentile(disparity[valid], 98))
        norm[valid] = np.clip((disparity[valid] - disp_min) / (disp_max - disp_min + 1e-6), 0, 1)
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


def save_disparity_depth_figure(path: str | Path, disparity: np.ndarray, depth: np.ndarray) -> None:
    """Save side-by-side disparity and depth maps with colorbars."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    valid_disparity = np.isfinite(disparity) & (disparity > 0)
    valid_depth = np.isfinite(depth) & (depth > 0)

    if valid_disparity.any():
        disp_vmin = float(np.nanpercentile(disparity[valid_disparity], 2))
        disp_vmax = float(np.nanpercentile(disparity[valid_disparity], 98))
    else:
        disp_vmin, disp_vmax = 0.0, 1.0

    if valid_depth.any():
        depth_vmin = float(np.nanpercentile(depth[valid_depth], 2))
        depth_vmax = float(np.nanpercentile(depth[valid_depth], 98))
    else:
        depth_vmin, depth_vmax = 0.0, 1.0

    if disp_vmax <= disp_vmin:
        disp_vmax = disp_vmin + 1.0
    if depth_vmax <= depth_vmin:
        depth_vmax = depth_vmin + 1.0

    disparity_cmap = plt.get_cmap("viridis").copy()
    depth_cmap = plt.get_cmap("magma").copy()
    disparity_cmap.set_bad(color="black")
    depth_cmap.set_bad(color="black")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    disp_plot = axes[0].imshow(
        np.ma.masked_where(~valid_disparity, disparity),
        cmap=disparity_cmap,
        vmin=disp_vmin,
        vmax=disp_vmax,
    )
    axes[0].set_title("Disparity map (px)")
    axes[0].axis("off")
    fig.colorbar(disp_plot, ax=axes[0], fraction=0.046, pad=0.04)

    depth_plot = axes[1].imshow(
        np.ma.masked_where(~valid_depth, depth),
        cmap=depth_cmap,
        vmin=depth_vmin,
        vmax=depth_vmax,
    )
    axes[1].set_title("Depth map (m)")
    axes[1].axis("off")
    fig.colorbar(depth_plot, ax=axes[1], fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
