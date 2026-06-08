# Robotics Perception Homework: Camera Calibration and Stereo Depth

This starter codebase is for the perception programming assignment on camera model, single-camera calibration, stereo calibration, rectification, disparity, and depth estimation.

Students should complete the `TODO` sections and generate a short report with calibration results and analysis.

## Directory structure

```text
RoboticsPerceptionCalibrationStarter/
├── configs/
│   └── checkerboard.yaml
├── robotics_perception/
│   ├── camera_model.py
│   ├── calibration.py
│   ├── stereo_calibration.py
│   ├── visualization.py
│   └── io_utils.py
├── scripts/
│   ├── generate_synthetic_data.py
│   ├── single_camera_entry.py
│   └── stereo_entry.py
├── tests/
│   └── test_camera_model.py
├── requirements.txt
└── README.md
```

## Installation

Create a Python environment and install dependencies:

```bash
pip install -r requirements.txt
```

Recommended Python version: 3.9 or newer.

## Step 1: Generate synthetic calibration images

A small synthetic dataset can be generated for debugging. Students may also use real checkerboard images collected by themselves.

```bash
python scripts/generate_synthetic_data.py \
  --output_dir sample_data/synthetic \
  --num_images 20 \
  --checkerboard_cols 9 \
  --checkerboard_rows 6 \
  --square_size 0.025
```

This creates:

```text
sample_data/synthetic/single/*.png
sample_data/synthetic/stereo/left/*.png
sample_data/synthetic/stereo/right/*.png
sample_data/synthetic/ground_truth.json
```

## Step 2: Single-camera calibration

Complete the TODO sections in:

```text
robotics_perception/camera_model.py
robotics_perception/calibration.py
scripts/single_camera_entry.py
```

Then run:

```bash
python scripts/single_camera_entry.py \
  --image_dir sample_data/synthetic/single \
  --checkerboard_cols 9 \
  --checkerboard_rows 6 \
  --square_size 0.025 \
  --output_dir outputs/single
```

Expected outputs include detected corner visualizations, undistorted images, calibration parameters, and reprojection error.

## Step 3: Stereo calibration, rectification, disparity, and depth

Complete the TODO sections in:

```text
robotics_perception/stereo_calibration.py
scripts/stereo_entry.py
```

Then run:

```bash
python scripts/stereo_entry.py \
  --left_dir sample_data/synthetic/stereo/left \
  --right_dir sample_data/synthetic/stereo/right \
  --checkerboard_cols 9 \
  --checkerboard_rows 6 \
  --square_size 0.025 \
  --output_dir outputs/stereo
```

Expected outputs include stereo parameters, rectified image pairs, epipolar-line visualizations, disparity maps, and depth maps.

## Assignment expectations

Students should report:

- Camera intrinsic matrix and distortion coefficients.
- Reprojection error and interpretation.
- Example calibration target detections.
- Before/after undistortion images.
- Stereo extrinsic parameters and baseline.
- Rectification visualization with horizontal epipolar lines.
- Disparity and depth maps.
- Discussion of error sources and how calibration affects robot perception.

## Notes

- You may use OpenCV functions for corner detection and optimization.
- You must explain the camera model and reprojection error in your report.
- Keep units consistent. If `square_size` is in meters, the stereo baseline and depth are also in meters.
