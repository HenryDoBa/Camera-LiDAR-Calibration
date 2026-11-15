# Camera–LiDAR Calibration — final code for the supplied ELTE dataset

**Course:** 3D Sensing and Sensor Fusion

**Input checked:** 15 paired files from `camera_lidar_sample.zip`, containing `1_pictures/Dev0_Image_w1920_h1200_fn*.jpg` and `1_cartesians/test_fn*.xyz` (XYZ plus 3 extra intensity/color columns). Intrinsic matrix `K=[[1250,0,960],[0,1250,600],[0,0,1]]`. Known sphere radius **0.30 m**. Only camera **Dev0** is used.

## Important findings

- The 15-frame sample contains a fully visible yellow ball in frames **fn52, fn64, fn75, fn87, fn98**. In **fn41** it is cut by the image border. Other sampled frames mostly show no ball; empty detections are normal, not a program error.
- The sample produces **5 candidate 3D–3D pairs**. It is **not sufficient** to satisfy the instructor's requirement of **at least six manually checked correspondences**. Run the software on the original complete extracted folders, select at least six true matches, and inspect the point clouds carefully before submitting the transform.
- `fn` numbers match across the two modalities. They are **frame IDs, not decimal timestamp values**; the assignment mentions encoded timestamps, but none are evident from the sample names. `--pair-mode frameid` matches equal IDs without claiming verified clock synchronization. Confirm capture-time alignment if metadata become available.
- This pipeline is tuned to the **yellow** sphere in these recordings. Other colors/lighting may require adjusting `calib/yellow.py`. The camera model assumes *zero unreported lens distortion*.

## Environment (Ubuntu)

```bash
cd camera_lidar_final
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Python 3.10+ recommended. Optional for GUI 3D inspection: `python -m pip install open3d` (or open PLY in MeshLab/CloudCompare). All other core features do not require Open3D.

## Run on the 15-frame sample

If `camera_lidar_sample.zip` is in `~/Downloads`:

```bash
bash run_sample.sh ~/Downloads/camera_lidar_sample.zip
```

It extracts the ZIP under `sample_data/` and creates `results_sample/`: CSVs with detection quality and 3D centers, 15 colorized PLY files, 15 camera overlays, and `review.html` for visual inspection. Only the six LiDAR frames with user-provided approximate seed locations are fitted, to avoid false positive background spheres. Five camera detections pass the color/edge/ellipse consistency tests.

**The sample script deliberately does not register five pairs**, because the task requires at least six manually accepted good pairs.

## Run on ALL extracted group-1 images and LiDAR clouds

The two original extraction directories should be exactly:

```text
~/Downloads/1_pictures/Dev0_Image_w1920_h1200_fn1.jpg ...
~/Downloads/1_cartesians/test_fn1.xyz ...
```

Run:

```bash
bash run_full.sh ~/Downloads
```

The script runs both algorithms on *every file* and writes `results_full/`. The LiDAR track uses sparse preliminary seeds in `config/sample_lidar_seeds.csv` and follows the target between seeded frames; before any valid seed or after repeated misses, it writes a gray-only PLY and a `failed` row rather than fabricating a sphere. Some frames with a real ball may still be missed and must be corrected with additional rows in the seed CSV. There is no guaranteed automatic detection rate.

Open `results_full/review.html`; click the overlays and check the matching `results_full/lidar_colored_ply/test_fnNN.ply` in MeshLab or CloudCompare. **Red points must be on the ball, not on the human or background.** Frames with incorrect ellipses or point clouds must not be selected.

Create the file `selected_frames.txt` in the project root, listing **at least six accepted LiDAR stems**, e.g.:

```text
test_fn52
test_fn55
test_fn64
test_fn75
test_fn87
test_fn98
```

The above list is **only an example**: `fn55` has NOT been supplied or checked; do not copy this list without examining the complete dataset. Prefer well-spread 3D positions and more than six approved frames.

Then:

```bash
python run.py register --out results_full --selected selected_frames.txt --register-threshold 0.12
python run.py validate --out results_full
```

The result is `results_full/T_camera_lidar.txt`, `T_lidar_camera.txt`, `calibration_result.json`, `registration_residuals.csv`, `registration_overlays/`, and `registration_3d.png`.

**Transformation definition** (meters):

```text
p_camera = R_camera_lidar @ p_lidar + t_camera_lidar
```

`register` refuses fewer than six approved pairs. The Kabsch/SVD rigid transform is robustly estimated through RANSAC, rejecting 3D center correspondences beyond the given threshold. Check inlier count, residual errors, near-collinear motion warnings, and reprojection overlays before reporting `R` and `t`.

## How it meets the four marking criteria

1. **LiDAR**: RANSAC with radius 0.30 m, robust nonlinear least-squares, colored binary PLY (red inlier vertices) for **each** input frame, and per-frame centers/errors.
2. **Camera**: yellow HSV ROI proposal, Canny edges, RANSAC robust ellipse fitting; quality metrics and overlaid ellipse images. The preliminary ROI proposal is *not* the ellipse fit.
3. **3D sphere center**: reconstruct the metric center from the fitted silhouette conic and the given intrinsics + known radius (full 3D, not naive ellipse center backprojection).
4. **Extrinsics**: match corresponding `fn` IDs, manually review at least six reliable pairs, use RANSAC+Kabsch SVD to recover a proper rotation and translation, save both 4x4 transforms, inspect residuals.

## Diagnostics and corrections

If the wrong sphere/nearby human is fitted on a frame, **add or edit** a seed row in `config/sample_lidar_seeds.csv` (the CSV must have `filename,x,y,z`). The seed is only an approximate center in meters, not a ground-truth measurement. Manual inspection is essential. Radius and inlier counts alone **do not prove** the fitted points belong to the ball. Re-run LiDAR fitting, then `review` after changing seeds. If no camera sphere is detected in a frame that actually contains one, inspect HSV thresholds in `calib/yellow.py`, or use the manual ROI route:

```bash
python run.py camera --camera-dir ~/Downloads/1_pictures --out results_full --roi-csv camera_rois.csv
```

Manual camera ROI columns: `filename,x,y,w,h`. With this alternate mode, you should carefully verify images because the algorithm may choose background contours in some frames. The automatic yellow detector is better for the current dataset.

`--pair-mode timestamp` is available if *actual* comparable Unix timestamps appear in new file names. **Never** label fn IDs as timestamps in the report. Non-lens distortion parameters were given, so the implementation cannot model unknown radial/tangential distortion.

## Submission

The teacher asks for **one ZIP** containing all results and source. The project ZIP supplied here is **finalized source code tested on the sample, not a completed 25-point submission**. Add `results_full`, a report based on the measured final `R,t` and manually approved frames, and any needed demonstration figures once the full dataset has been run. See `REPORT_TEMPLATE.md`.

Unit tests: `python -m pytest -q tests` (pytest is a listed development dependency).
