#!/usr/bin/env python3
"""Entry point for stereo calibration, rectification, disparity, and depth."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from robotics_perception.io_utils import save_json
from robotics_perception.stereo_calibration import (
    calibrate_stereo_camera,
    collect_stereo_points,
    compute_disparity_sgbm,
    disparity_to_depth,
    rectify_pair,
    stereo_rectify,
)
from robotics_perception.visualization import (
    draw_horizontal_epipolar_lines,
    save_depth_colormap,
    save_disparity_colormap,
    save_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stereo calibration starter script")
    parser.add_argument("--left_dir", type=str, required=True, help="Folder containing left images")
    parser.add_argument("--right_dir", type=str, required=True, help="Folder containing right images")
    parser.add_argument("--checkerboard_cols", type=int, default=9, help="Number of inner corners along width")
    parser.add_argument("--checkerboard_rows", type=int, default=6, help="Number of inner corners along height")
    parser.add_argument("--square_size", type=float, default=0.025, help="Checkerboard square size in meters")
    parser.add_argument("--output_dir", type=str, default="outputs/stereo", help="Output folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkerboard_size = (args.checkerboard_cols, args.checkerboard_rows)
    object_points, left_points, right_points, image_size, used_pairs = collect_stereo_points(
        left_dir=args.left_dir,
        right_dir=args.right_dir,
        checkerboard_size=checkerboard_size,
        square_size=args.square_size,
    )
    print(f"Using {len(used_pairs)} valid stereo pairs")

    stereo = calibrate_stereo_camera(object_points, left_points, right_points, image_size)
    print("Left K:")
    print(stereo.left.K)
    print("Right K:")
    print(stereo.right.K)
    print("Stereo translation T:")
    print(stereo.T.ravel())
    print(f"Baseline: {stereo.baseline:.6f} meters if square_size is meters")

    # TODO(student): adapt this block to your stereo_rectify return format.
    rectification = stereo_rectify(stereo)

    # Suggested return format:
    # R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right = rectification
    R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right = rectification

    save_json(
        output_dir / "stereo_summary.json",
        {
            "left_K": stereo.left.K,
            "right_K": stereo.right.K,
            "left_dist": stereo.left.dist,
            "right_dist": stereo.right.dist,
            "R": stereo.R,
            "T": stereo.T,
            "E": stereo.E,
            "F": stereo.F,
            "baseline": stereo.baseline,
            "image_size": stereo.image_size,
            "R1": R1,
            "R2": R2,
            "P1": P1,
            "P2": P2,
            "Q": Q,
        },
    )

    # Process one or a few example pairs.
    for idx, (left_path, right_path) in enumerate(used_pairs[:3]):
        left_img = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_img = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        rect_l, rect_r = rectify_pair(left_img, right_img, map1_left, map2_left, map1_right, map2_right)
        epi_vis = draw_horizontal_epipolar_lines(rect_l, rect_r)
        save_image(output_dir / "rectified" / f"pair_{idx:03d}.png", np.hstack([rect_l, rect_r]))
        save_image(output_dir / "epipolar_lines" / f"pair_{idx:03d}.png", epi_vis)

        disparity = compute_disparity_sgbm(rect_l, rect_r)
        depth = disparity_to_depth(disparity, fx=stereo.left.fx, baseline=stereo.baseline)
        save_disparity_colormap(output_dir / "disparity" / f"pair_{idx:03d}.png", disparity)
        save_depth_colormap(output_dir / "depth" / f"pair_{idx:03d}.png", depth)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
