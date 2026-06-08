"""Stereo calibration, rectification, disparity, and depth utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from robotics_perception.camera_model import CameraParameters, build_checkerboard_object_points
from robotics_perception.calibration import find_checkerboard_corners, list_images


@dataclass
class StereoParameters:
    """Container for stereo calibration results."""

    left: CameraParameters
    right: CameraParameters
    R: np.ndarray
    T: np.ndarray
    E: np.ndarray
    F: np.ndarray
    image_size: Tuple[int, int]

    @property
    def baseline(self) -> float:
        """Stereo baseline length in the same unit as checkerboard square size."""
        return float(np.linalg.norm(self.T.reshape(3)))


def collect_stereo_points(
    left_dir: str | Path,
    right_dir: str | Path,
    checkerboard_size: Tuple[int, int],
    square_size: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray], tuple[int, int], list[tuple[Path, Path]]]:
    """Collect matched checkerboard corners from synchronized stereo images."""
    left_paths = list_images(left_dir)
    right_paths = list_images(right_dir)

    if len(left_paths) != len(right_paths):
        print(f"[WARN] Number of left/right images differs: {len(left_paths)} vs {len(right_paths)}")

    object_template = build_checkerboard_object_points(checkerboard_size, square_size)
    object_points_list: list[np.ndarray] = []
    left_points_list: list[np.ndarray] = []
    right_points_list: list[np.ndarray] = []
    used_pairs: list[tuple[Path, Path]] = []
    image_size: tuple[int, int] | None = None

    for left_path, right_path in zip(left_paths, right_paths):
        left_img = cv2.imread(str(left_path), cv2.IMREAD_COLOR)
        right_img = cv2.imread(str(right_path), cv2.IMREAD_COLOR)
        if left_img is None or right_img is None:
            print(f"[WARN] Could not read pair: {left_path}, {right_path}")
            continue

        h, w = left_img.shape[:2]
        if image_size is None:
            image_size = (w, h)

        ok_l, corners_l = find_checkerboard_corners(left_img, checkerboard_size)
        ok_r, corners_r = find_checkerboard_corners(right_img, checkerboard_size)

        if ok_l and ok_r and corners_l is not None and corners_r is not None:
            object_points_list.append(object_template.copy())
            left_points_list.append(corners_l.astype(np.float32))
            right_points_list.append(corners_r.astype(np.float32))
            used_pairs.append((left_path, right_path))
        else:
            print(f"[WARN] Checkerboard not found in pair: {left_path.name}, {right_path.name}")

    if image_size is None:
        raise RuntimeError("No readable stereo images found.")

    return object_points_list, left_points_list, right_points_list, image_size, used_pairs


def calibrate_stereo_camera(
    object_points_list: list[np.ndarray],
    left_points_list: list[np.ndarray],
    right_points_list: list[np.ndarray],
    image_size: tuple[int, int],
) -> StereoParameters:
    """Estimate left/right camera intrinsics and stereo extrinsics.

    A common workflow is:
      1. Calibrate left camera.
      2. Calibrate right camera.
      3. Run cv2.stereoCalibrate with fixed intrinsics.
    """
    # TODO(student): implement stereo calibration.
    # Hint:
    #   - Use cv2.calibrateCamera for left and right cameras.
    #   - Use cv2.stereoCalibrate with cv2.CALIB_FIX_INTRINSIC.
    #   - Return StereoParameters(...).
    raise NotImplementedError("calibrate_stereo_camera is not implemented")


def stereo_rectify(
    stereo: StereoParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute stereo rectification transforms.

    Suggested return format:
        R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right
    """
    # TODO(student): implement cv2.stereoRectify and cv2.initUndistortRectifyMap.
    # You may change the return type if you document it clearly.
    raise NotImplementedError("stereo_rectify is not implemented")


def rectify_pair(
    left_img: np.ndarray,
    right_img: np.ndarray,
    map1_left: np.ndarray,
    map2_left: np.ndarray,
    map1_right: np.ndarray,
    map2_right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply precomputed rectification maps to a stereo pair."""
    rect_l = cv2.remap(left_img, map1_left, map2_left, interpolation=cv2.INTER_LINEAR)
    rect_r = cv2.remap(right_img, map1_right, map2_right, interpolation=cv2.INTER_LINEAR)
    return rect_l, rect_r


def compute_disparity_sgbm(
    rectified_left: np.ndarray,
    rectified_right: np.ndarray,
    min_disparity: int = 0,
    num_disparities: int = 128,
    block_size: int = 5,
) -> np.ndarray:
    """Compute disparity from rectified stereo images.

    Returns:
        disparity: Floating-point disparity map in pixels.
    """
    # TODO(student): implement StereoSGBM disparity computation.
    # Hint:
    #   - Convert images to grayscale.
    #   - num_disparities must be divisible by 16.
    #   - OpenCV returns fixed-point disparity scaled by 16.
    raise NotImplementedError("compute_disparity_sgbm is not implemented")


def disparity_to_depth(disparity: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    """Convert disparity to depth using Z = fx * B / d.

    Invalid or non-positive disparity should be assigned np.nan or zero.
    """
    # TODO(student): implement disparity-to-depth conversion.
    raise NotImplementedError("disparity_to_depth is not implemented")
