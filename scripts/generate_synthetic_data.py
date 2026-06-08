#!/usr/bin/env python3
"""Generate a small synthetic checkerboard dataset for debugging calibration code.

This script is intentionally provided as a utility, not as the main homework
solution. It creates planar checkerboard images using projective warping.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def make_checkerboard_texture(cols_inner: int, rows_inner: int, square_px: int = 80, margin_px: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Create checkerboard texture and its board-plane square corner coordinates.

    Returns:
        texture: grayscale checkerboard image.
        src_quad: four outer board corners in texture pixel coordinates.
    """
    squares_x = cols_inner + 1
    squares_y = rows_inner + 1
    w = squares_x * square_px + 2 * margin_px
    h = squares_y * square_px + 2 * margin_px
    texture = np.full((h, w), 230, dtype=np.uint8)

    for y in range(squares_y):
        for x in range(squares_x):
            color = 30 if (x + y) % 2 == 0 else 230
            x0 = margin_px + x * square_px
            y0 = margin_px + y * square_px
            texture[y0 : y0 + square_px, x0 : x0 + square_px] = color

    src_quad = np.array(
        [
            [margin_px, margin_px],
            [margin_px + squares_x * square_px, margin_px],
            [margin_px + squares_x * square_px, margin_px + squares_y * square_px],
            [margin_px, margin_px + squares_y * square_px],
        ],
        dtype=np.float32,
    )
    return texture, src_quad


def rodrigues(rx: float, ry: float, rz: float) -> np.ndarray:
    rvec = np.array([rx, ry, rz], dtype=np.float64)
    R, _ = cv2.Rodrigues(rvec)
    return R


def project_board_outer_corners(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    board_width: float,
    board_height: float,
) -> np.ndarray:
    corners_3d = np.array(
        [
            [0.0, 0.0, 0.0],
            [board_width, 0.0, 0.0],
            [board_width, board_height, 0.0],
            [0.0, board_height, 0.0],
        ],
        dtype=np.float64,
    )
    pts_cam = (R @ corners_3d.T).T + t.reshape(1, 3)
    pts = (K @ pts_cam.T).T
    pts = pts[:, :2] / pts[:, 2:3]
    return pts.astype(np.float32)


def render_board(
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_size: tuple[int, int],
    texture: np.ndarray,
    src_quad: np.ndarray,
    board_width: float,
    board_height: float,
) -> np.ndarray:
    w, h = image_size
    dst_quad = project_board_outer_corners(K, R, t, board_width, board_height)
    H = cv2.getPerspectiveTransform(src_quad, dst_quad)
    background = np.full((h, w), 190, dtype=np.uint8)
    warped = cv2.warpPerspective(texture, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=190)
    mask = cv2.warpPerspective(np.full_like(texture, 255), H, (w, h), flags=cv2.INTER_NEAREST, borderValue=0)
    image = background.copy()
    image[mask > 0] = warped[mask > 0]
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    noise = np.random.normal(0, 2.0, image.shape).astype(np.int16)
    image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return image


def valid_pose(K: np.ndarray, R: np.ndarray, t: np.ndarray, image_size: tuple[int, int], board_width: float, board_height: float) -> bool:
    w, h = image_size
    pts = project_board_outer_corners(K, R, t, board_width, board_height)
    margin = 20
    return bool(np.all(pts[:, 0] > margin) and np.all(pts[:, 0] < w - margin) and np.all(pts[:, 1] > margin) and np.all(pts[:, 1] < h - margin))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic checkerboard calibration data")
    parser.add_argument("--output_dir", type=str, default="sample_data/synthetic")
    parser.add_argument("--num_images", type=int, default=20)
    parser.add_argument("--checkerboard_cols", type=int, default=9)
    parser.add_argument("--checkerboard_rows", type=int, default=6)
    parser.add_argument("--square_size", type=float, default=0.025)
    parser.add_argument("--image_width", type=int, default=640)
    parser.add_argument("--image_height", type=int, default=480)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    out = Path(args.output_dir)
    single_dir = out / "single"
    left_dir = out / "stereo" / "left"
    right_dir = out / "stereo" / "right"
    for d in [single_dir, left_dir, right_dir]:
        d.mkdir(parents=True, exist_ok=True)

    image_size = (args.image_width, args.image_height)
    K_left = np.array([[580.0, 0.0, args.image_width / 2], [0.0, 580.0, args.image_height / 2], [0.0, 0.0, 1.0]])
    K_right = np.array([[580.0, 0.0, args.image_width / 2 - 2.0], [0.0, 580.0, args.image_height / 2 + 1.0], [0.0, 0.0, 1.0]])
    baseline = 0.08
    T_right_left = np.array([-baseline, 0.0, 0.0], dtype=np.float64)  # right camera center relative to left frame convention for rendering

    texture, src_quad = make_checkerboard_texture(args.checkerboard_cols, args.checkerboard_rows)
    board_width = (args.checkerboard_cols + 1) * args.square_size
    board_height = (args.checkerboard_rows + 1) * args.square_size

    poses = []
    attempts = 0
    while len(poses) < args.num_images and attempts < args.num_images * 50:
        attempts += 1
        rx = rng.uniform(-0.45, 0.45)
        ry = rng.uniform(-0.45, 0.45)
        rz = rng.uniform(-0.50, 0.50)
        R = rodrigues(rx, ry, rz)
        t = np.array([
            rng.uniform(-0.12, 0.06),
            rng.uniform(-0.08, 0.05),
            rng.uniform(0.55, 0.95),
        ], dtype=np.float64)
        if valid_pose(K_left, R, t, image_size, board_width, board_height):
            poses.append((R, t))

    if len(poses) < args.num_images:
        raise RuntimeError("Could not sample enough valid board poses. Try reducing num_images.")

    for i, (R, t_left) in enumerate(poses):
        left_img = render_board(K_left, R, t_left, image_size, texture, src_quad, board_width, board_height)
        # Approximate right-camera rendering by changing the board translation in the right camera frame.
        t_right = t_left + np.array([baseline, 0.0, 0.0], dtype=np.float64)
        right_img = render_board(K_right, R, t_right, image_size, texture, src_quad, board_width, board_height)

        cv2.imwrite(str(single_dir / f"calib_{i:03d}.png"), left_img)
        cv2.imwrite(str(left_dir / f"pair_{i:03d}.png"), left_img)
        cv2.imwrite(str(right_dir / f"pair_{i:03d}.png"), right_img)

    gt = {
        "K_left": K_left.tolist(),
        "K_right": K_right.tolist(),
        "dist_left": [0, 0, 0, 0, 0],
        "dist_right": [0, 0, 0, 0, 0],
        "baseline_m": baseline,
        "checkerboard_cols": args.checkerboard_cols,
        "checkerboard_rows": args.checkerboard_rows,
        "square_size_m": args.square_size,
        "image_size": list(image_size),
        "note": "Synthetic images are for debugging. Real calibration images are recommended for the final report.",
    }
    with open(out / "ground_truth.json", "w", encoding="utf-8") as f:
        json.dump(gt, f, indent=2)

    print(f"Generated synthetic dataset at {out}")


if __name__ == "__main__":
    main()
