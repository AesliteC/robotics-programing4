#!/usr/bin/env python3
"""Generate a report demonstration sequence for the calibration pipeline."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from robotics_perception.calibration import (
    calibrate_single_camera,
    collect_calibration_points,
    find_checkerboard_corners,
    list_images,
)
from robotics_perception.camera_model import undistort_image
from robotics_perception.stereo_calibration import (
    calibrate_stereo_camera,
    collect_stereo_points,
    compute_disparity_sgbm,
    disparity_to_depth,
    rectify_pair,
    stereo_rectify,
)
from robotics_perception.visualization import draw_checkerboard_corners, draw_horizontal_epipolar_lines, save_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a short calibration demonstration sequence")
    parser.add_argument("--data_root", type=str, default="sample_data/synthetic")
    parser.add_argument("--output_dir", type=str, default="outputs/demonstration")
    parser.add_argument("--checkerboard_cols", type=int, default=9)
    parser.add_argument("--checkerboard_rows", type=int, default=6)
    parser.add_argument("--square_size", type=float, default=0.025)
    parser.add_argument("--canvas_width", type=int, default=1280)
    parser.add_argument("--canvas_height", type=int, default=560)
    return parser.parse_args()


def label_panel(image: np.ndarray, label: str) -> np.ndarray:
    panel = image.copy()
    cv2.rectangle(panel, (0, 0), (panel.shape[1], 34), (0, 0, 0), -1)
    cv2.putText(panel, label, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return panel


def make_frame(image: np.ndarray, title: str, width: int, height: int) -> np.ndarray:
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    title_h = 56
    cv2.rectangle(canvas, (0, 0), (width, title_h), (32, 42, 54), -1)
    cv2.putText(canvas, title, (24, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.82, (255, 255, 255), 2, cv2.LINE_AA)

    max_w = width - 48
    max_h = height - title_h - 36
    scale = min(max_w / image.shape[1], max_h / image.shape[0])
    new_size = (int(round(image.shape[1] * scale)), int(round(image.shape[0] * scale)))
    resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    x0 = (width - resized.shape[1]) // 2
    y0 = title_h + (max_h - resized.shape[0]) // 2
    canvas[y0 : y0 + resized.shape[0], x0 : x0 + resized.shape[1]] = resized
    return canvas


def colorize_map(values: np.ndarray, cmap: int, lower_percentile: float, upper_percentile: float) -> np.ndarray:
    valid = np.isfinite(values) & (values > 0)
    norm = np.zeros(values.shape, dtype=np.uint8)
    if valid.any():
        vmin = float(np.nanpercentile(values[valid], lower_percentile))
        vmax = float(np.nanpercentile(values[valid], upper_percentile))
        scaled = np.clip((values[valid] - vmin) / (vmax - vmin + 1e-6), 0.0, 1.0)
        norm[valid] = np.round(scaled * 255).astype(np.uint8)
    color = cv2.applyColorMap(norm, cmap)
    color[~valid] = (0, 0, 0)
    return color


def write_video(path: Path, frames: list[np.ndarray], fps: float = 1.0) -> bool:
    if not frames:
        return False
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not writer.isOpened():
        return False
    for frame in frames:
        writer.write(frame)
    writer.release()
    return True


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    frames_dir = output_dir / "frames"
    checkerboard_size = (args.checkerboard_cols, args.checkerboard_rows)

    if output_dir.exists():
        shutil.rmtree(output_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    single_paths = list_images(data_root / "single")
    object_points, image_points, image_size, used_paths = collect_calibration_points(
        image_paths=single_paths,
        checkerboard_size=checkerboard_size,
        square_size=args.square_size,
    )
    camera, _, _, _, _ = calibrate_single_camera(object_points, image_points, image_size)

    single_image = cv2.imread(str(used_paths[0]), cv2.IMREAD_COLOR)
    ok, corners = find_checkerboard_corners(single_image, checkerboard_size)
    if not ok or corners is None:
        raise RuntimeError(f"Checkerboard not found in {used_paths[0]}")

    corners_vis = draw_checkerboard_corners(single_image, checkerboard_size, corners)
    undistorted = undistort_image(single_image, camera)
    undistortion_vis = np.hstack(
        [
            label_panel(single_image, "Original image"),
            label_panel(undistorted, "Undistorted image"),
        ]
    )

    object_points_s, left_points, right_points, image_size_s, used_pairs = collect_stereo_points(
        left_dir=data_root / "stereo" / "left",
        right_dir=data_root / "stereo" / "right",
        checkerboard_size=checkerboard_size,
        square_size=args.square_size,
    )
    stereo = calibrate_stereo_camera(object_points_s, left_points, right_points, image_size_s)
    rectification = stereo_rectify(stereo)
    _, _, _, _, _, map1_left, map2_left, map1_right, map2_right = rectification

    left_img = cv2.imread(str(used_pairs[0][0]), cv2.IMREAD_COLOR)
    right_img = cv2.imread(str(used_pairs[0][1]), cv2.IMREAD_COLOR)
    rect_l, rect_r = rectify_pair(left_img, right_img, map1_left, map2_left, map1_right, map2_right)
    rectified_vis = draw_horizontal_epipolar_lines(rect_l, rect_r)
    disparity = compute_disparity_sgbm(rect_l, rect_r)
    depth = disparity_to_depth(disparity, fx=stereo.left.fx, baseline=stereo.baseline)
    disparity_vis = colorize_map(
        disparity,
        cv2.COLORMAP_MAGMA if hasattr(cv2, "COLORMAP_MAGMA") else cv2.COLORMAP_JET,
        2,
        98,
    )
    depth_vis = colorize_map(
        depth,
        cv2.COLORMAP_VIRIDIS if hasattr(cv2, "COLORMAP_VIRIDIS") else cv2.COLORMAP_JET,
        2,
        95,
    )

    frame_specs = [
        ("01_detected_corners.png", "1. Detected checkerboard corners", corners_vis),
        ("02_undistorted_image.png", "2. Original image and undistorted result", undistortion_vis),
        ("03_stereo_rectification.png", "3. Rectified stereo pair with horizontal epipolar lines", rectified_vis),
        ("04_disparity_map.png", "4. SGBM disparity map", disparity_vis),
        ("05_depth_map.png", "5. Depth map computed from disparity", depth_vis),
    ]
    frames = [
        make_frame(image, title, args.canvas_width, args.canvas_height)
        for _, title, image in frame_specs
    ]
    for (filename, _, _), frame in zip(frame_specs, frames):
        save_image(frames_dir / filename, frame)

    contact_sheet = np.vstack(frames)
    save_image(output_dir / "demonstration_sequence.png", contact_sheet)
    video_ok = write_video(output_dir / "demonstration.mp4", frames)

    readme_lines = [
        "# Calibration Demonstration",
        "",
        "This folder contains the required demonstration sequence for the camera calibration homework.",
        "",
        "Frame order:",
        "",
        "1. `frames/01_detected_corners.png`: detected checkerboard corners.",
        "2. `frames/02_undistorted_image.png`: original image and undistorted image.",
        "3. `frames/03_stereo_rectification.png`: rectified stereo pair with horizontal epipolar lines.",
        "4. `frames/04_disparity_map.png`: SGBM disparity map.",
        "5. `frames/05_depth_map.png`: depth map computed from disparity.",
        "",
        "`demonstration_sequence.png` is a single contact sheet containing the same ordered frames.",
    ]
    if video_ok:
        readme_lines.append("`demonstration.mp4` is a short video made from the same frames.")
    else:
        readme_lines.append("The numbered PNG files are the primary image sequence demonstration.")
    (output_dir / "README.md").write_text("\n".join(readme_lines) + "\n", encoding="utf-8")

    print(f"Demonstration frames saved to {frames_dir}")
    print(f"Contact sheet saved to {output_dir / 'demonstration_sequence.png'}")
    if video_ok:
        print(f"Video saved to {output_dir / 'demonstration.mp4'}")


if __name__ == "__main__":
    main()
