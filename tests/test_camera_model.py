import numpy as np

from robotics_perception.camera_model import build_checkerboard_object_points, compute_reprojection_error


def test_build_checkerboard_object_points_shape_and_values():
    pts = build_checkerboard_object_points((3, 2), 0.1)
    assert pts.shape == (6, 3)
    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.1, 0.0, 0.0],
            [0.2, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.1, 0.1, 0.0],
            [0.2, 0.1, 0.0],
        ]
    )
    assert np.allclose(pts, expected)


def test_compute_reprojection_error_zero_case():
    # This is a minimal sanity test students can use after implementing the function.
    obj = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype=np.float64)
    K = np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    dist = np.zeros((5, 1), dtype=np.float64)
    rvec = np.zeros((3, 1), dtype=np.float64)
    tvec = np.zeros((3, 1), dtype=np.float64)
    img = np.array([[50.0, 40.0], [60.0, 40.0], [50.0, 50.0]], dtype=np.float64)

    mean, per_view = compute_reprojection_error([obj], [img], [rvec], [tvec], K, dist)
    assert abs(mean) < 1e-9
    assert len(per_view) == 1
    assert abs(per_view[0]) < 1e-9
