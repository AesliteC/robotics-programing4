"""Single-camera calibration utilities.

Students should complete the TODO sections for calibration and error analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

from robotics_perception.camera_model import (
    CameraParameters,
    build_checkerboard_object_points,
    compute_reprojection_error,
)


def list_images(image_dir: str | Path) -> list[Path]:
    """List common image files in a folder."""
    image_dir = Path(image_dir)
    exts = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    paths: list[Path] = []
    for ext in exts:
        paths.extend(sorted(image_dir.glob(ext)))
    return sorted(paths)


def find_checkerboard_corners(
    image: np.ndarray,
    checkerboard_size: Tuple[int, int],
    refine: bool = True,
) -> tuple[bool, np.ndarray | None]:
    """Detect checkerboard inner corners.

    Args:
        image: BGR or grayscale image.
        checkerboard_size: Number of inner corners as (cols, rows).
        refine: Whether to run sub-pixel corner refinement.

    Returns:
        success: True if the checkerboard was found.
        corners: Array with shape (N, 2) if successful, otherwise None.
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    success, corners = cv2.findChessboardCorners(gray, checkerboard_size, flags)

    if not success:
        return False, None

    if refine:
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            50,
            1e-4,
        )
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    return True, corners.reshape(-1, 2)


def collect_calibration_points(
    image_paths: list[Path],
    checkerboard_size: Tuple[int, int],
    square_size: float,
) -> tuple[list[np.ndarray], list[np.ndarray], tuple[int, int], list[Path]]:
    """Collect object points and image points from calibration images."""
    object_template = build_checkerboard_object_points(checkerboard_size, square_size)

    object_points_list: list[np.ndarray] = []
    image_points_list: list[np.ndarray] = []
    used_paths: list[Path] = []
    image_size: tuple[int, int] | None = None

    for path in image_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            print(f"[WARN] Could not read image: {path}")
            continue

        h, w = image.shape[:2]
        if image_size is None:
            image_size = (w, h)

        success, corners = find_checkerboard_corners(image, checkerboard_size)
        if success and corners is not None:
            object_points_list.append(object_template.copy())
            image_points_list.append(corners.astype(np.float32))
            used_paths.append(path)
        else:
            print(f"[WARN] Checkerboard not found: {path}")

    if image_size is None:
        raise RuntimeError("No readable images found.")

    return object_points_list, image_points_list, image_size, used_paths


def calibrate_single_camera(
    object_points_list: list[np.ndarray],
    image_points_list: list[np.ndarray],
    image_size: tuple[int, int],
) -> tuple[CameraParameters, list[np.ndarray], list[np.ndarray], float, list[float]]:
    """Estimate camera intrinsics, distortion, and per-view extrinsics.

    Returns:
        camera: CameraParameters containing K, distortion, and image size.
        rvecs: Rotation vectors, one per calibration image.
        tvecs: Translation vectors, one per calibration image.
        mean_error: Mean reprojection error in pixels.
        per_view_errors: Reprojection error per image.
    """
    if len(object_points_list) < 5:
        raise ValueError("At least 5 valid calibration images are recommended.")

    # TODO(student): implement camera calibration.
    # Hint:
    #   ret, K, dist, rvecs, tvecs = cv2.calibrateCamera(...)
    #   camera = CameraParameters(K=K, dist=dist, image_size=image_size)
    #   mean_error, per_view_errors = compute_reprojection_error(...)
    #   return camera, rvecs, tvecs, mean_error, per_view_errors
    raise NotImplementedError("calibrate_single_camera is not implemented")


def save_calibration_npz(
    output_path: str | Path,
    camera: CameraParameters,
    rvecs: list[np.ndarray],
    tvecs: list[np.ndarray],
    mean_error: float,
    per_view_errors: list[float],
) -> None:
    """Save calibration results to a NumPy .npz file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        K=camera.K,
        dist=camera.dist,
        image_size=np.array(camera.image_size),
        rvecs=np.array(rvecs, dtype=object),
        tvecs=np.array(tvecs, dtype=object),
        mean_error=float(mean_error),
        per_view_errors=np.array(per_view_errors, dtype=np.float64),
    )


def load_camera_npz(path: str | Path) -> CameraParameters:
    """Load camera intrinsics and distortion from an .npz file."""
    data = np.load(path, allow_pickle=True)
    return CameraParameters(
        K=data["K"],
        dist=data["dist"],
        image_size=tuple(data["image_size"].tolist()),
    )
