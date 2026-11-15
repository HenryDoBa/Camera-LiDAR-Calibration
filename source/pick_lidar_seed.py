#!/usr/bin/env python3
"""Optional interactive Open3D: SHIFT+left-click 3+ LiDAR sphere surface points,
press Q to exit; output is their mean as a rough ROI seed, NOT the fitted center.
"""
import argparse
import numpy as np
from calib.io_utils import read_points


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--file', required=True)
    parser.add_argument('--bin-dim', type=int, default=4)
    parser.add_argument('--max-points', type=int, default=150000)
    a = parser.parse_args()
    try:
        import open3d as o3d
    except ImportError as e:
        raise SystemExit('Requires open3d: pip install open3d') from e
    X = read_points(a.file, bin_dim=a.bin_dim)
    if len(X) > a.max_points:
        X = X[np.random.default_rng(42).choice(len(X), a.max_points, replace=False)]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(X)
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window(window_name='SHIFT-click sphere points, then Q')
    vis.add_geometry(cloud)
    vis.run()
    idx = vis.get_picked_points()
    vis.destroy_window()
    if not idx:
        raise SystemExit('No points selected')
    seed = X[idx].mean(axis=0)
    print('Approximate seed in meters: --seed', *(f'{v:.5f}' for v in seed))


if __name__ == '__main__':
    main()
