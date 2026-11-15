# Camera-LiDAR Calibration - Submission contents

Camera: Dev0 only. LiDAR-to-camera transform convention: p_C = R p_L + t.

## Main folders
- source/: Python implementation, requirements, setup and tests.
- results/lidar_colored_ply/: ALL PLY outputs from processed LiDAR scans, with detected sphere points red.
- results/lidar_centers.csv: centre and fit quality for EVERY processed scan, including failures.
- results/camera_centers.csv: Dev0 ellipse detections and 3D camera centres.
- results/paired_centers.csv: matching Dev0 camera and LiDAR centres.
- results/camera_overlays/: selected QA images, not full input JPG dataset.
- results/registration_overlays/: selected QA projections.
- results/calibration_result.json and T_camera_lidar.txt: rigid transform and fit metrics.
- results/selected_frames.txt and registration_residuals.csv: selected matches and errors.
- REPORT_Camera_LiDAR.pdf: concise report (if present).

## Reproduction
Install requirements from source/requirements.txt and see source/README.md.
Input camera photos and raw XYZ clouds are not included; obtain them from the assignment data.
The calibration fits were validated on Dev0 and 7 manually reviewed frames.

## Packaging summary
- LiDAR scans/PLY: 160
- Dev0 camera CSV rows: 160
- Dev0 paired detections: 51
- Selected frame IDs: test_fn81, test_fn82, test_fn83, test_fn86, test_fn87, test_fn88, test_fn89
- Included camera overlays: 7
- Included registration overlays: 7
- Included PDF report: yes

## Important limitation
This package includes exactly the results found in the chosen results folder.
If the original assignment requires nearly 400 frames over groups 1, 2, 3,
processing only data group 1 is not full all-frame coverage.
In-sample registration RMSE is NOT ground-truth calibration accuracy.
Equal fn file numbers are used for pairing; precise time alignment is unverified.
