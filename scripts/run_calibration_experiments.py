#!/usr/bin/env python3
"""Run calibration experiments and save report-ready tables and plots."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
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
from robotics_perception.stereo_calibration import (
    calibrate_stereo_camera,
    collect_stereo_points,
    compute_disparity_sgbm,
    disparity_to_depth,
    rectify_pair,
    stereo_rectify,
)
from robotics_perception.visualization import (
    draw_checkerboard_corners,
    draw_horizontal_epipolar_lines,
    save_depth_colormap,
    save_disparity_depth_figure,
    save_disparity_colormap,
    save_image,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run report experiments for camera calibration homework")
    parser.add_argument("--data_root", type=str, default="sample_data/synthetic")
    parser.add_argument("--output_root", type=str, default="outputs/experiments")
    parser.add_argument("--experiment_data_root", type=str, default="experiment_data")
    parser.add_argument("--checkerboard_cols", type=int, default=9)
    parser.add_argument("--checkerboard_rows", type=int, default=6)
    parser.add_argument("--square_size", type=float, default=0.025)
    parser.add_argument("--counts", type=int, nargs="+", default=[5, 8, 12])
    return parser.parse_args()


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    try:
        rel_src = os.path.relpath(src.resolve(), dst.parent.resolve())
        dst.symlink_to(rel_src)
    except OSError:
        shutil.copy2(src, dst)


def prepare_subsets(data_root: Path, experiment_root: Path, counts: list[int]) -> None:
    if experiment_root.exists():
        shutil.rmtree(experiment_root)
    single_paths = sorted((data_root / "single").glob("*.png"))
    left_paths = sorted((data_root / "stereo" / "left").glob("*.png"))
    right_paths = sorted((data_root / "stereo" / "right").glob("*.png"))

    for count in counts:
        single_dir = experiment_root / f"single_{count}"
        for path in single_paths[:count]:
            link_or_copy(path, single_dir / path.name)

        left_dir = experiment_root / f"stereo_{count}" / "left"
        right_dir = experiment_root / f"stereo_{count}" / "right"
        for path in left_paths[:count]:
            link_or_copy(path, left_dir / path.name)
        for path in right_paths[:count]:
            link_or_copy(path, right_dir / path.name)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_single_experiment(
    image_dir: Path,
    output_dir: Path,
    checkerboard_size: tuple[int, int],
    square_size: float,
) -> dict[str, Any]:
    image_paths = list_images(image_dir)
    object_points, image_points, image_size, used_paths = collect_calibration_points(
        image_paths=image_paths,
        checkerboard_size=checkerboard_size,
        square_size=square_size,
    )
    camera, rvecs, tvecs, mean_error, per_view_errors = calibrate_single_camera(
        object_points,
        image_points,
        image_size,
    )
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

    for path in used_paths[:3]:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        ok, corners = find_checkerboard_corners(image, checkerboard_size)
        if ok and corners is not None:
            save_image(output_dir / "corners" / path.name, draw_checkerboard_corners(image, checkerboard_size, corners))
            save_image(output_dir / "undistorted" / path.name, np.hstack([image, undistort_image(image, camera)]))

    return {
        "name": output_dir.name,
        "num_images": len(used_paths),
        "mean_reprojection_error_px": float(mean_error),
        "fx": float(camera.K[0, 0]),
        "fy": float(camera.K[1, 1]),
        "cx": float(camera.K[0, 2]),
        "cy": float(camera.K[1, 2]),
        "k1": float(camera.dist.ravel()[0]) if camera.dist.size > 0 else np.nan,
        "k2": float(camera.dist.ravel()[1]) if camera.dist.size > 1 else np.nan,
        "p1": float(camera.dist.ravel()[2]) if camera.dist.size > 2 else np.nan,
        "p2": float(camera.dist.ravel()[3]) if camera.dist.size > 3 else np.nan,
        "k3": float(camera.dist.ravel()[4]) if camera.dist.size > 4 else np.nan,
    }


def stereo_summary_dict(stereo: Any, rectification: tuple[np.ndarray, ...]) -> dict[str, Any]:
    R1, R2, P1, P2, Q, _, _, _, _ = rectification
    return {
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
    }


def run_stereo_experiment(
    left_dir: Path,
    right_dir: Path,
    output_dir: Path,
    checkerboard_size: tuple[int, int],
    square_size: float,
    gt_baseline: float,
) -> tuple[dict[str, Any], Any, tuple[np.ndarray, ...], list[tuple[Path, Path]]]:
    object_points, left_points, right_points, image_size, used_pairs = collect_stereo_points(
        left_dir=left_dir,
        right_dir=right_dir,
        checkerboard_size=checkerboard_size,
        square_size=square_size,
    )
    stereo = calibrate_stereo_camera(object_points, left_points, right_points, image_size)
    rectification = stereo_rectify(stereo)
    save_json(output_dir / "stereo_summary.json", stereo_summary_dict(stereo, rectification))

    R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right = rectification
    for idx, (left_path, right_path) in enumerate(used_pairs[:3]):
        left_img = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_img = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        rect_l, rect_r = rectify_pair(left_img, right_img, map1_left, map2_left, map1_right, map2_right)
        save_image(output_dir / "rectified" / f"pair_{idx:03d}.png", np.hstack([rect_l, rect_r]))
        save_image(output_dir / "epipolar_lines" / f"pair_{idx:03d}.png", draw_horizontal_epipolar_lines(rect_l, rect_r))
        disparity = compute_disparity_sgbm(rect_l, rect_r)
        depth = disparity_to_depth(disparity, fx=stereo.left.fx, baseline=stereo.baseline)
        save_disparity_colormap(output_dir / "disparity" / f"pair_{idx:03d}.png", disparity)
        save_depth_colormap(output_dir / "depth" / f"pair_{idx:03d}.png", depth)
        save_disparity_depth_figure(output_dir / "disparity_depth" / f"pair_{idx:03d}.png", disparity, depth)

    return (
        {
            "name": output_dir.name,
            "num_pairs": len(used_pairs),
            "baseline_m": float(stereo.baseline),
            "baseline_abs_error_m": float(abs(stereo.baseline - gt_baseline)),
            "baseline_rel_error": float(abs(stereo.baseline - gt_baseline) / gt_baseline),
            "left_fx": float(stereo.left.fx),
            "left_fy": float(stereo.left.fy),
            "right_fx": float(stereo.right.fx),
            "right_fy": float(stereo.right.fy),
            "tx": float(stereo.T.reshape(3)[0]),
            "ty": float(stereo.T.reshape(3)[1]),
            "tz": float(stereo.T.reshape(3)[2]),
        },
        stereo,
        rectification,
        used_pairs,
    )


def plot_single_summary(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xs = [row["num_images"] for row in rows]
    ys = [row["mean_reprojection_error_px"] for row in rows]
    plt.figure(figsize=(6, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Number of calibration images")
    plt.ylabel("Mean reprojection error (px)")
    plt.title("Single-camera calibration error vs image count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_stereo_summary(rows: list[dict[str, Any]], gt_baseline: float, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    xs = [row["num_pairs"] for row in rows]
    baselines = [row["baseline_m"] for row in rows]
    errors = [row["baseline_abs_error_m"] for row in rows]
    fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(xs, baselines, marker="o", color="tab:blue", label="Estimated baseline")
    ax1.axhline(gt_baseline, color="tab:green", linestyle="--", label="Ground truth baseline")
    ax1.set_xlabel("Number of stereo pairs")
    ax1.set_ylabel("Baseline (m)")
    ax1.grid(True, alpha=0.3)
    ax2 = ax1.twinx()
    ax2.plot(xs, errors, marker="s", color="tab:red", label="Absolute error")
    ax2.set_ylabel("Absolute error (m)")
    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines + lines2, labels + labels2, loc="best")
    plt.title("Stereo baseline vs stereo pair count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_sgbm_experiment(
    output_dir: Path,
    stereo: Any,
    rectification: tuple[np.ndarray, ...],
    pair: tuple[Path, Path],
) -> list[dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    _, _, _, _, _, map1_left, map2_left, map1_right, map2_right = rectification
    left_img = cv2.imread(str(pair[0]), cv2.IMREAD_COLOR)
    right_img = cv2.imread(str(pair[1]), cv2.IMREAD_COLOR)
    rect_l, rect_r = rectify_pair(left_img, right_img, map1_left, map2_left, map1_right, map2_right)
    configs = [
        ("sgbm_small_block", 128, 3),
        ("sgbm_default", 128, 5),
        ("sgbm_large_block", 128, 9),
    ]
    rows = []
    for name, num_disparities, block_size in configs:
        disparity = compute_disparity_sgbm(
            rect_l,
            rect_r,
            num_disparities=num_disparities,
            block_size=block_size,
        )
        depth = disparity_to_depth(disparity, fx=stereo.left.fx, baseline=stereo.baseline)
        save_disparity_colormap(output_dir / f"{name}_disparity.png", disparity)
        save_depth_colormap(output_dir / f"{name}_depth.png", depth)
        if name == "sgbm_default":
            save_disparity_depth_figure(output_dir / f"{name}_disparity_depth.png", disparity, depth)
        valid_disp = np.isfinite(disparity) & (disparity > 0)
        valid_depth = np.isfinite(depth) & (depth > 0)
        rows.append(
            {
                "name": name,
                "num_disparities": num_disparities,
                "block_size": block_size,
                "valid_disparity_ratio": float(np.mean(valid_disp)),
                "median_disparity_px": float(np.nanmedian(disparity[valid_disp])) if valid_disp.any() else np.nan,
                "median_depth_m": float(np.nanmedian(depth[valid_depth])) if valid_depth.any() else np.nan,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output_root)
    experiment_root = Path(args.experiment_data_root)
    checkerboard_size = (args.checkerboard_cols, args.checkerboard_rows)

    output_root.mkdir(parents=True, exist_ok=True)
    prepare_subsets(data_root, experiment_root, args.counts)

    with open(data_root / "ground_truth.json", "r", encoding="utf-8") as f:
        ground_truth = json.load(f)
    gt_baseline = float(ground_truth["baseline_m"])

    single_rows = []
    for count in args.counts:
        single_rows.append(
            run_single_experiment(
                experiment_root / f"single_{count}",
                output_root / f"single_{count}",
                checkerboard_size,
                args.square_size,
            )
        )
    write_csv(
        output_root / "tables" / "single_image_count_summary.csv",
        single_rows,
        ["name", "num_images", "mean_reprojection_error_px", "fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3"],
    )
    save_json(output_root / "tables" / "single_image_count_summary.json", {"rows": single_rows})
    plot_single_summary(single_rows, output_root / "plots" / "single_reprojection_error_vs_images.png")

    stereo_rows = []
    stereo_artifacts: dict[int, tuple[Any, tuple[np.ndarray, ...], list[tuple[Path, Path]]]] = {}
    for count in args.counts:
        row, stereo, rectification, used_pairs = run_stereo_experiment(
            experiment_root / f"stereo_{count}" / "left",
            experiment_root / f"stereo_{count}" / "right",
            output_root / f"stereo_{count}",
            checkerboard_size,
            args.square_size,
            gt_baseline,
        )
        stereo_rows.append(row)
        stereo_artifacts[count] = (stereo, rectification, used_pairs)
    write_csv(
        output_root / "tables" / "stereo_pair_count_summary.csv",
        stereo_rows,
        ["name", "num_pairs", "baseline_m", "baseline_abs_error_m", "baseline_rel_error", "left_fx", "left_fy", "right_fx", "right_fy", "tx", "ty", "tz"],
    )
    save_json(output_root / "tables" / "stereo_pair_count_summary.json", {"rows": stereo_rows})
    plot_stereo_summary(stereo_rows, gt_baseline, output_root / "plots" / "stereo_baseline_error_vs_pairs.png")

    full_count = max(args.counts)
    stereo, rectification, used_pairs = stereo_artifacts[full_count]
    sgbm_rows = run_sgbm_experiment(output_root / "sgbm", stereo, rectification, used_pairs[0])
    write_csv(
        output_root / "tables" / "sgbm_parameter_summary.csv",
        sgbm_rows,
        ["name", "num_disparities", "block_size", "valid_disparity_ratio", "median_disparity_px", "median_depth_m"],
    )
    save_json(output_root / "tables" / "sgbm_parameter_summary.json", {"rows": sgbm_rows})

    baseline_row = next(row for row in stereo_rows if row["num_pairs"] == full_count)
    save_json(
        output_root / "tables" / "baseline_vs_ground_truth.json",
        {
            "estimated_baseline_m": baseline_row["baseline_m"],
            "ground_truth_baseline_m": gt_baseline,
            "absolute_error_m": baseline_row["baseline_abs_error_m"],
            "relative_error": baseline_row["baseline_rel_error"],
        },
    )

    print(f"Experiment data prepared at {experiment_root}")
    print(f"Experiment outputs saved to {output_root}")


if __name__ == "__main__":
    main()
