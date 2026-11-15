#!/usr/bin/env python3
"""Camera--LiDAR calibration assignment, run `python run.py --help`."""
from __future__ import annotations
import argparse
import csv
import html
import json
import os
from pathlib import Path
import numpy as np
import cv2

from calib.geometry import K, RADIUS_M, fit_sphere_ransac, sphere_center_from_ellipse, ransac_transform
from calib.vision import detect_ellipse, dynamic_roi, annotate
from calib.yellow import detect_yellow_sphere
from calib.io_utils import (IMAGE_EXT, LIDAR_EXT, list_files, read_points,
                            write_colored_ply, read_csv, write_csv,
                            read_overrides, pair_rows, timestamp_from_name)


def limit(files, n):
    return files[:n] if n > 0 else files


def process_lidar(args):
    files = limit(list_files(args.lidar_dir, LIDAR_EXT), args.max_frames)
    if not files:
        raise RuntimeError('No point cloud files found; inspect --lidar-dir')
    out = Path(args.out)
    ply_dir = out / 'lidar_colored_ply'
    manual_seeds = read_overrides(args.seed_csv, ('x', 'y', 'z'))
    current = np.asarray(args.seed, float) if args.seed is not None else None
    if current is None and not manual_seeds:
        print('WARNING: no LiDAR seed. No detections will be attempted unless --allow-unseeded is used.')
        print('Best practice: supply --seed-csv with anchors for frames that contain the sphere.')
    rows = []
    missed = 0
    for idx, file in enumerate(files, 1):
        pts = read_points(file, bin_dim=args.bin_dim)
        manually_seeded = file.stem in manual_seeds
        if manually_seeded:
            current = np.array(manual_seeds[file.stem], dtype=float)
            missed = 0
        inliers = np.zeros(len(pts), bool)
        fit = None
        if current is not None:
            crop = np.linalg.norm(pts - current, axis=1) <= args.roi_radius
            sub = pts[crop]
        else:
            crop = np.ones(len(pts), bool)
            sub = pts
        if len(sub) >= args.min_inliers and (current is not None or args.allow_unseeded) and (not args.manual_only or manually_seeded):
            fit = fit_sphere_ransac(sub, radius=RADIUS_M,
                                    threshold=args.threshold, iterations=args.iterations,
                                    min_inliers=args.min_inliers,
                                    seed=42 + idx)
        # A good match should stay reasonably near previous target location.
        if fit is not None and current is not None and np.linalg.norm(fit.center - current) > args.roi_radius * 0.9:
            fit = None
        status = 'failed'
        center = [np.nan] * 3
        rmse = np.nan
        n = 0
        if fit is not None:
            center = fit.center.tolist()
            current = fit.center.copy()
            inliers[np.flatnonzero(crop)[fit.inliers]] = True
            n = int(inliers.sum())
            rmse = fit.rmse
            status = 'ok'
            missed = 0
        else:
            missed += 1
            if missed >= args.reset_after_failures and not args.manual_only:
                current = None  # avoid latching indefinitely onto a person/background object
        write_colored_ply(ply_dir / f'{file.stem}.ply', pts, inliers)
        rows.append({'stem': file.stem, 'filename': file.name,
                     'status': status, 'x': center[0], 'y': center[1], 'z': center[2],
                     'inliers': n, 'sphere_rmse_m': rmse, 'total_points': len(pts),
                     'timestamp_s': timestamp_from_name(file.stem)})
        print(f'[LiDAR {idx}/{len(files)}] {file.name}: {status}, sphere points={n}')
    path = out / 'lidar_centers.csv'
    write_csv(path, ['stem', 'filename', 'status', 'x', 'y', 'z',
                     'inliers', 'sphere_rmse_m', 'total_points', 'timestamp_s'], rows)
    print(f'Wrote {path}, plus {len(files)} colored PLY point clouds')


def process_camera(args):
    files = limit(list_files(args.camera_dir, IMAGE_EXT), args.max_frames)
    if not files:
        raise RuntimeError('No image files found; inspect --camera-dir')
    out = Path(args.out)
    overlay_dir = out / 'camera_overlays'
    overlay_dir.mkdir(parents=True, exist_ok=True)
    manual_rois = read_overrides(args.roi_csv, ('x', 'y', 'w', 'h'))
    current_roi = tuple(args.roi) if args.roi is not None else None
    previous = None
    rows = []
    for idx, file in enumerate(files, 1):
        image = cv2.imread(str(file), cv2.IMREAD_COLOR)
        if image is None:
            print(f'WARNING: image could not be decoded: {file}')
            continue
        if idx == 1:
            print(f'First image dimensions: {image.shape[1]} x {image.shape[0]}')
            if image.shape[1] < 960 or image.shape[0] < 600:
                print('WARNING: supplied intrinsic matrix assumes unscaled camera images.')
        if file.stem in manual_rois:
            roi = manual_rois[file.stem]
            previous = None  # manual ROI overrides stale tracker state
        elif previous is not None:
            roi = dynamic_roi(previous, image.shape, factor=args.track_factor)
        else:
            roi = current_roi
        if idx == 1 and args.interactive_roi and file.stem not in manual_rois:
            try:
                selected = cv2.selectROI('Select rectangle around sphere; ENTER to confirm', image,
                                         fromCenter=False, showCrosshair=True)
                cv2.destroyWindow('Select rectangle around sphere; ENTER to confirm')
            except cv2.error as exc:
                raise RuntimeError('Interactive ROI needs a graphical desktop; use --roi x y w h') from exc
            if selected[2] <= 0 or selected[3] <= 0:
                raise RuntimeError('No valid ROI selected')
            current_roi = roi = selected
            print('Initial camera ROI:', *map(int, selected))
        if args.yellow_auto:
            ellipse, roi, quality, detection_status = detect_yellow_sphere(
                image, edge_threshold=args.edge_threshold)
            # No detection is a valid outcome: do not infer a 3D point from background edges.
            previous = None
        else:
            ellipse, quality, _ = detect_ellipse(image, roi=roi, previous=previous,
                                                 threshold=args.edge_threshold)
            if ellipse is None and current_roi is not None and previous is not None and file.stem not in manual_rois:
                ellipse, quality, _ = detect_ellipse(image, roi=current_roi,
                                                     previous=None,
                                                     threshold=args.edge_threshold)
                if ellipse is not None:
                    roi = current_roi
        row = {'stem': file.stem, 'filename': file.name, 'status': 'failed',
               'cx_px': np.nan, 'cy_px': np.nan, 'axis1_px': np.nan,
               'axis2_px': np.nan, 'angle_deg': np.nan,
               'cam_x_m': np.nan, 'cam_y_m': np.nan, 'cam_z_m': np.nan,
               'edge_inliers': 0, 'ellipse_rmse_px': np.nan,
               'angular_coverage': 0, 'timestamp_s': timestamp_from_name(file.stem)}
        if ellipse is not None:
            try:
                center3d = sphere_center_from_ellipse(ellipse, K_matrix=K, radius=RADIUS_M)
                (cx, cy), (a, b), angle = ellipse
                row.update({'status': 'ok', 'cx_px': cx, 'cy_px': cy,
                            'axis1_px': a, 'axis2_px': b, 'angle_deg': angle,
                            'cam_x_m': center3d[0], 'cam_y_m': center3d[1],
                            'cam_z_m': center3d[2],
                            'edge_inliers': quality['edge_inliers'],
                            'ellipse_rmse_px': quality['ellipse_rmse_px'],
                            'angular_coverage': quality['angular_coverage']})
                previous = ellipse
            except (ValueError, FloatingPointError) as exc:
                print(f'WARNING: invalid ellipse geometry in {file.name}: {exc}')
                ellipse = None
        overlay = annotate(image, ellipse, roi,
                           status=f'{file.name}: {row["status"]}')
        cv2.imwrite(str(overlay_dir / f'{file.stem}.jpg'), overlay,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        rows.append(row)
        print(f'[Camera {idx}/{len(files)}] {file.name}: {row["status"]}')
    path = out / 'camera_centers.csv'
    cols = ['stem', 'filename', 'status', 'cx_px', 'cy_px', 'axis1_px',
            'axis2_px', 'angle_deg', 'cam_x_m', 'cam_y_m', 'cam_z_m',
            'edge_inliers', 'ellipse_rmse_px', 'angular_coverage', 'timestamp_s']
    write_csv(path, cols, rows)
    print(f'Wrote {path}, plus {len(rows)} camera overlay images')


def build_pairs(args):
    out = Path(args.out)
    l = read_csv(out / 'lidar_centers.csv')
    c = read_csv(out / 'camera_centers.csv')
    matches = pair_rows(l, c, mode=args.pair_mode, max_dt=args.max_dt,
                        offset=args.camera_time_offset)
    rows = []
    for a, b, delta in matches:
        rows.append({'lidar_stem': a['stem'], 'camera_stem': b['stem'],
                     'lidar_filename': a['filename'], 'camera_filename': b['filename'],
                     'dt_s': delta if delta is not None else '',
                     'lidar_x': a['x'], 'lidar_y': a['y'], 'lidar_z': a['z'],
                     'camera_x': b['cam_x_m'], 'camera_y': b['cam_y_m'],
                     'camera_z': b['cam_z_m'],
                     'lidar_inliers': a['inliers'],
                     'lidar_rmse_m': a['sphere_rmse_m'],
                     'ellipse_rmse_px': b['ellipse_rmse_px'],
                     'angular_coverage': b['angular_coverage']})
    write_csv(out / 'paired_centers.csv',
              ['lidar_stem', 'camera_stem', 'lidar_filename', 'camera_filename',
               'dt_s', 'lidar_x', 'lidar_y', 'lidar_z', 'camera_x', 'camera_y',
               'camera_z', 'lidar_inliers', 'lidar_rmse_m', 'ellipse_rmse_px',
               'angular_coverage'], rows)
    return rows


def review(args):
    rows = build_pairs(args)
    out = Path(args.out)
    body = ['<!doctype html><html><head><meta charset="utf-8">',
            '<title>Calibration frame review</title>',
            '<style>body{font-family:Arial,sans-serif;max-width:1250px;margin:auto;padding:25px}'
            'table{border-collapse:collapse;width:100%}td,th{border:1px solid #ccc;padding:7px}'
            'img{width:250px;max-height:180px;object-fit:contain}code{font-size:12px}'
            '</style></head><body><h1>Camera–LiDAR frame review</h1>',
            '<p>Manually inspect image overlays AND corresponding red-inlier PLY clouds. '
            'Pick >=6 diverse and reliable pairs, not just adjacent frames. '
            'Copy LiDAR stems into <code>selected_frames.txt</code>, one per line.</p>',
            '<table><tr><th>LiDAR stem (copy to selection)</th><th>Camera overlay</th>'
            '<th>LiDAR check</th><th>Pair quality</th></tr>']
    for p in rows:
        stem = html.escape(p['lidar_stem'])
        camera = html.escape(p['camera_stem'])
        body.append(f'<tr><td><code>{stem}</code></td>'
                    f'<td><a href="camera_overlays/{camera}.jpg"><img '
                    f'src="camera_overlays/{camera}.jpg"></a></td>'
                    f'<td><code>lidar_colored_ply/{stem}.ply</code><br>'
                    f'{int(float(p["lidar_inliers"]))} sphere points</td>'
                    f'<td>Time difference: {html.escape(str(p["dt_s"])) if p["dt_s"] != "" else "not available (matched by fn ID)"}<br>'
                    f'LiDAR RMSE: {html.escape(str(p["lidar_rmse_m"]))} m<br>'
                    f'Ellipse RMSE: {html.escape(str(p["ellipse_rmse_px"]))} px<br>'
                    f'Coverage: {html.escape(str(p["angular_coverage"]))}</td></tr>')
    body += ['</table></body></html>']
    path = out / 'review.html'
    path.write_text('\n'.join(body), encoding='utf-8')
    print(f'Paired {len(rows)} successful sphere and ellipse detections.')
    print(f'Open in browser: {path.resolve()}')
    print('Inspect sphere points in PLY with MeshLab or Open3D, and select >=6 valid frames.')


def register(args):
    out = Path(args.out)
    pairs = read_csv(out / 'paired_centers.csv')
    selected_path = Path(args.selected)
    if not selected_path.exists():
        raise FileNotFoundError(f'Please manually create {selected_path} with >=6 LiDAR stems')
    lines = [line.strip() for line in selected_path.read_text().splitlines()
             if line.strip() and not line.lstrip().startswith('#')]
    selected = {Path(name).stem for name in lines}
    chosen = [p for p in pairs if p['lidar_stem'] in selected]
    if len(chosen) < 6:
        raise ValueError(f'Only {len(chosen)} selected frames matched; need >=6 good paired frames')
    if len(chosen) != len(selected):
        print(f'WARNING: {len(selected)-len(chosen)} selected filenames were not paired.')
    X = np.array([[float(p[f'lidar_{axis}']) for axis in 'xyz'] for p in chosen])
    Y = np.array([[float(p[f'camera_{axis}']) for axis in 'xyz'] for p in chosen])
    geometry_singular_values = np.linalg.svd(X - X.mean(axis=0), full_matrices=False)[1]
    if geometry_singular_values[1] < 0.03:
        raise ValueError('Sphere centers are nearly collinear. Select more spatially diverse frames.')
    R, t, inliers, residuals = ransac_transform(X, Y, threshold=args.register_threshold,
                                                 iterations=args.register_iterations)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    inv = np.linalg.inv(T)
    selected_rows = []
    for p, ok, err in zip(chosen, inliers, residuals):
        selected_rows.append({**p, 'registration_inlier': int(ok), 'residual_m': err})
    write_csv(out / 'registration_residuals.csv',
              list(selected_rows[0]), selected_rows)
    info = {'transform_definition': 'p_camera = R_camera_lidar @ p_lidar + t_camera_lidar',
            'camera_from_lidar': T.tolist(), 'lidar_from_camera': inv.tolist(),
            'rotation_camera_lidar': R.tolist(), 'translation_camera_lidar_m': t.tolist(),
            'selected_pairs': len(chosen), 'ransac_inlier_pairs': int(inliers.sum()),
            'rmse_inlier_m': float(np.sqrt(np.mean(residuals[inliers] ** 2))),
            'rmse_all_m': float(np.sqrt(np.mean(residuals ** 2))),
            'lidar_motion_singular_values_m': geometry_singular_values.tolist(),
            'camera_intrinsic_K': K.tolist(), 'sphere_radius_m': RADIUS_M,
            'note': 'Verify correspondences, ellipse overlays, PLY points, and camera distortion before submission.'}
    (out / 'calibration_result.json').write_text(json.dumps(info, indent=2), encoding='utf-8')
    np.savetxt(out / 'T_camera_lidar.txt', T, fmt='%.10f')
    np.savetxt(out / 'T_lidar_camera.txt', inv, fmt='%.10f')
    print('\nTRANSFORM (camera <-- LiDAR):\n', T)
    print(f'RANSAC inliers: {inliers.sum()} / {len(chosen)}; '
          f'inlier RMSE = {info["rmse_inlier_m"]:.4f} m')
    print('LiDAR center motion singular values:', geometry_singular_values)
    if geometry_singular_values[2] < 0.03:
        print('WARNING: selected sphere centers are nearly coplanar; prefer 3D motion.')
    print('Saved calibration_result.json, T_camera_lidar.txt, registration_residuals.csv')


def validate(args):
    """Visual QC: project both 3D centers to the selected camera and compare."""
    out = Path(args.out)
    T = np.loadtxt(out / 'T_camera_lidar.txt')
    R, t = T[:3, :3], T[:3, 3]
    rows = read_csv(out / 'registration_residuals.csv')
    qc = out / 'registration_overlays'
    qc.mkdir(parents=True, exist_ok=True)
    all_L, all_C, errors = [], [], []
    for row in rows:
        center_l = np.array([float(row[f'lidar_{axis}']) for axis in 'xyz'])
        center_c = np.array([float(row[f'camera_{axis}']) for axis in 'xyz'])
        predicted = R @ center_l + t
        all_L.append(predicted)
        all_C.append(center_c)
        errors.append(float(np.linalg.norm(predicted - center_c)))
        image_file = out / 'camera_overlays' / f'{row["camera_stem"]}.jpg'
        frame = cv2.imread(str(image_file))
        if frame is None or predicted[2] <= 0 or center_c[2] <= 0:
            continue
        p1 = K @ (predicted / predicted[2])
        p2 = K @ (center_c / center_c[2])
        # Green = projected transformed LiDAR center, blue = reconstructed camera center.
        pt1 = tuple(np.round(p1[:2]).astype(int))
        pt2 = tuple(np.round(p2[:2]).astype(int))
        cv2.circle(frame, pt1, 10, (0, 255, 0), -1)
        cv2.circle(frame, pt2, 7, (255, 80, 0), -1)
        cv2.line(frame, pt1, pt2, (255, 255, 0), 2)
        cv2.putText(frame, f'3D error = {errors[-1]:.3f} m', (35, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, .85, (0, 255, 255), 2)
        cv2.imwrite(str(qc / f'{row["lidar_stem"]}.jpg'), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    if all_L:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        A = np.array(all_L)
        B = np.array(all_C)
        fig = plt.figure(figsize=(9, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(B[:, 0], B[:, 1], B[:, 2], label='3D camera centers', marker='o')
        ax.scatter(A[:, 0], A[:, 1], A[:, 2], label='LiDAR centers transformed', marker='x')
        for v, w in zip(A, B):
            ax.plot([v[0], w[0]], [v[1], w[1]], [v[2], w[2]], alpha=.6)
        ax.set_xlabel('X camera [m]')
        ax.set_ylabel('Y camera [m]')
        ax.set_zlabel('Z camera [m]')
        ax.set_title('Selected 3D centers after camera-LiDAR registration')
        ax.legend(loc='best')
        fig.tight_layout()
        fig.savefig(out / 'registration_3d.png', dpi=165)
        plt.close(fig)
    print('Saved registration overlays, registration_3d.png')
    print(f'Mean 3D residual: {np.mean(errors):.4f} m; max: {np.max(errors):.4f} m')


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    l = sub.add_parser('lidar', help='Robust 0.30 m sphere fit; save red-sphere PLYs')
    l.add_argument('--lidar-dir', required=True, type=Path)
    l.add_argument('--out', default='results')
    l.add_argument('--seed', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                   help='Approximate first-frame sphere center in LiDAR coordinates [m]')
    l.add_argument('--seed-csv', type=Path, help='Optional per-frame manual LiDAR center seeds')
    l.add_argument('--roi-radius', type=float, default=0.85)
    l.add_argument('--threshold', type=float, default=0.022,
                   help='Sphere shell residual threshold in meters')
    l.add_argument('--min-inliers', type=int, default=12)
    l.add_argument('--iterations', type=int, default=750)
    l.add_argument('--allow-unseeded', action='store_true', help='Attempt a risky whole-cloud sphere fit without a seed (NOT recommended)')
    l.add_argument('--manual-only', action='store_true', help='Fit ONLY frames listed in --seed-csv; useful for a sparse sample')
    l.add_argument('--reset-after-failures', type=int, default=3, help='Forget tracked seed after N failed frames')
    l.add_argument('--bin-dim', type=int, default=4, choices=[3, 4, 5])
    l.add_argument('--max-frames', type=int, default=0, help='0 = process all')
    c = sub.add_parser('camera', help='Canny + robust ellipse fit + exact 3D sphere center')
    c.add_argument('--camera-dir', required=True, type=Path,
                   help='ONE camera folder; do not mix the two cameras')
    c.add_argument('--out', default='results')
    c.add_argument('--yellow-auto', action='store_true', help='Detect bright yellow sphere; HSV proposal + Canny + ellipse RANSAC')
    c.add_argument('--roi', type=int, nargs=4, metavar=('X', 'Y', 'W', 'H'))
    c.add_argument('--interactive-roi', action='store_true')
    c.add_argument('--roi-csv', type=Path, help='Optional per-frame camera ROI overrides')
    c.add_argument('--track-factor', type=float, default=3.7)
    c.add_argument('--edge-threshold', type=float, default=2.75)
    c.add_argument('--max-frames', type=int, default=0)
    r = sub.add_parser('review', help='Match timestamps; create HTML visual frame review')
    r.add_argument('--out', default='results')
    r.add_argument('--pair-mode', choices=['frameid', 'timestamp', 'name', 'index'], default='frameid')
    r.add_argument('--max-dt', type=float, default=0.06,
                   help='Maximum camera vs LiDAR timestamp difference [s]')
    r.add_argument('--camera-time-offset', type=float, default=0.0,
                   help='Added to camera timestamp before matching [s]')
    t = sub.add_parser('register', help='Calibrate from >=6 manually reviewed frames')
    t.add_argument('--out', default='results')
    t.add_argument('--selected', default='selected_frames.txt')
    t.add_argument('--register-threshold', type=float, default=0.12,
                   help='3D correspondence RANSAC threshold [m]')
    t.add_argument('--register-iterations', type=int, default=1500)
    v = sub.add_parser('validate', help='Plot 3D correspondences and image projection QC')
    v.add_argument('--out', default='results')
    return parser


def main():
    args = make_parser().parse_args()
    {'lidar': process_lidar, 'camera': process_camera,
     'review': review, 'register': register,
     'validate': validate}[args.command](args)


if __name__ == '__main__':
    main()
