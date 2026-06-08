#!/usr/bin/env python3
"""Entry point for single-camera calibration homework."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from robotics_perception.calibration import (
    calibrate_single_camera,
    collect_calibration_points,
    find_checkerboard_corners,
    list_images,
    save_calibration_npz,
)
from robotics_perception.camera_model import undistort_image
from robotics_perception.io_utils import save_json
from robotics_perception.visualization import draw_checkerboard_corners, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-camera calibration starter script")
    parser.add_argument("--image_dir", type=str, required=True, help="Folder containing calibration images")
    parser.add_argument("--checkerboard_cols", type=int, default=9, help="Number of inner corners along width")
    parser.add_argument("--checkerboard_rows", type=int, default=6, help="Number of inner corners along height")
    parser.add_argument("--square_size", type=float, default=0.025, help="Checkerboard square size in meters")
    parser.add_argument("--output_dir", type=str, default="outputs/single", help="Output folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    checkerboard_size = (args.checkerboard_cols, args.checkerboard_rows)
    image_paths = list_images(args.image_dir)
    print(f"Found {len(image_paths)} images in {args.image_dir}")

    object_points, image_points, image_size, used_paths = collect_calibration_points(
        image_paths=image_paths,
        checkerboard_size=checkerboard_size,
        square_size=args.square_size,
    )
    print(f"Using {len(used_paths)} valid calibration images")

    # Save detected corner visualizations.
    for path in used_paths[:10]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        ok, corners = find_checkerboard_corners(image, checkerboard_size)
        if ok and corners is not None:
            vis = draw_checkerboard_corners(image, checkerboard_size, corners)
            save_image(output_dir / "corners" / path.name, vis)

    camera, rvecs, tvecs, mean_error, per_view_errors = calibrate_single_camera(
        object_points, image_points, image_size
    )

    print("Camera intrinsic matrix K:")
    print(camera.K)
    print("Distortion coefficients:")
    print(camera.dist.ravel())
    print(f"Mean reprojection error: {mean_error:.4f} px")

    save_calibration_npz(output_dir / "single_camera_calibration.npz", camera, rvecs, tvecs, mean_error, per_view_errors)
    save_json(
        output_dir / "single_camera_summary.json",
        {
            "K": camera.K,
            "dist": camera.dist,
            "image_size": camera.image_size,
            "mean_reprojection_error_px": mean_error,
            "per_view_errors_px": per_view_errors,
            "num_valid_images": len(used_paths),
        },
    )

    # Save a few undistortion examples.
    for path in used_paths[:5]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        undistorted = undistort_image(image, camera)
        side_by_side = np.hstack([image, undistorted])
        save_image(output_dir / "undistorted" / path.name, side_by_side)

    print(f"Results saved to {output_dir}")


if __name__ == "__main__":
    main()
