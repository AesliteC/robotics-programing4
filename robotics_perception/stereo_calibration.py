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
    retl, Kl, distl, rvecsl, tvecsl = cv2.calibrateCamera(object_points_list, left_points_list, image_size, None, None)
    retr, Kr, distr, rvecsr, tvecsr = cv2.calibrateCamera(object_points_list, right_points_list, image_size, None, None)
    ret, Kl, distl, Kr, distr, R, T, E, F = cv2.stereoCalibrate(
        object_points_list,
        left_points_list,
        right_points_list,
        Kl,
        distl,
        Kr,
        distr,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC
    )
    return StereoParameters(
        CameraParameters(Kl, distl, image_size),
        CameraParameters(Kr, distr, image_size),
        R,
        T,
        E,
        F,
        image_size
    )

def stereo_rectify(
    stereo: StereoParameters,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute stereo rectification transforms.

    Suggested return format:
        R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right
    """
    # TODO(student): implement cv2.stereoRectify and cv2.initUndistortRectifyMap.
    # You may change the return type if you document it clearly.
    left = stereo.left
    right = stereo.right
    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        left.K, left.dist, right.K, right.dist, stereo.image_size, stereo.R, stereo.T
    )
    map1_left, map2_left = cv2.initUndistortRectifyMap(left.K, left.dist, R1, P1, stereo.image_size, cv2.CV_32FC1)
    map1_right, map2_right = cv2.initUndistortRectifyMap(right.K, right.dist, R2, P2, stereo.image_size, cv2.CV_32FC1)
    return R1, R2, P1, P2, Q, map1_left, map2_left, map1_right, map2_right


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
    if rectified_left.ndim == 3:
        left_gray = cv2.cvtColor(rectified_left, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = rectified_left
    if rectified_right.ndim == 3:
        right_gray = cv2.cvtColor(rectified_right, cv2.COLOR_BGR2GRAY)
    else:
        right_gray = rectified_right
    num_disparities = int(np.ceil(num_disparities / 16) * 16)
    block_size = max(3, int(block_size) | 1)

    def compute_with_min_disparity(matcher_min_disparity: int) -> np.ndarray:
        matcher = cv2.StereoSGBM_create(
            minDisparity=matcher_min_disparity,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * block_size * block_size,
            P2=32 * block_size * block_size,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32,
        )
        disp = matcher.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disp[disp <= matcher_min_disparity] = np.nan
        return disp

    disparity = compute_with_min_disparity(min_disparity)
    if min_disparity == 0:
        signed_min_disparity = -num_disparities // 2
        signed_disparity = compute_with_min_disparity(signed_min_disparity)
        signed_disparity = np.abs(signed_disparity)
        signed_disparity[signed_disparity <= 0] = np.nan
        valid = np.isfinite(disparity) & (disparity > 0)
        signed_valid = np.isfinite(signed_disparity) & (signed_disparity > 0)
        if np.mean(signed_valid) > np.mean(valid) * 1.2:
            disparity = signed_disparity
    return disparity


def disparity_to_depth(disparity: np.ndarray, fx: float, baseline: float) -> np.ndarray:
    """Convert disparity to depth using Z = fx * B / d.

    Invalid or non-positive disparity should be assigned np.nan or zero.
    """
    # TODO(student): implement disparity-to-depth conversion.
    depth = np.full(disparity.shape, np.nan, dtype=np.float32)
    valid = np.isfinite(disparity) & (disparity > 0)
    depth[valid] = float(fx) * float(baseline) / disparity[valid]
    return depth
