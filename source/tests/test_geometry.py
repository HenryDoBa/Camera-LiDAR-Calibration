import numpy as np
import cv2
from calib.geometry import (RADIUS_M, K, fit_sphere_ransac,
                            sphere_center_from_ellipse, rigid_transform, ransac_transform)
from calib.vision import detect_ellipse
from calib.io_utils import read_points, write_colored_ply, timestamp_from_name


def ellipse_from_sphere(center, r=0.30):
    """Analytic silhouette conic -> OpenCV ellipse, used only by tests."""
    c = np.asarray(center)
    cone = np.outer(c, c) - (c @ c - r * r) * np.eye(3)
    inv = np.linalg.inv(K)
    C = inv.T @ cone @ inv
    A, b, f = C[:2, :2], C[:2, 2], C[2, 2]
    mid = -np.linalg.solve(A, b)
    scale = float(mid @ A @ mid - f)
    M = A / scale
    vals, vecs = np.linalg.eigh(M)
    lengths = 2.0 / np.sqrt(vals)
    theta = np.degrees(np.arctan2(vecs[1, 0], vecs[0, 0]))
    return ((float(mid[0]), float(mid[1])), (float(lengths[0]), float(lengths[1])),
            float(theta))


def test_sphere_3d_from_ellipse():
    for c in ([0, 0, 2.0], [.5, .3, 4.5], [-.4, .2, 1.5], [1.0, -.4, 5]):
        ellipse = ellipse_from_sphere(c)
        actual = sphere_center_from_ellipse(ellipse)
        assert np.allclose(actual, c, atol=1e-7), (c, actual)


def test_fixed_radius_sphere_ransac():
    rng = np.random.default_rng(45)
    center = np.array([2.4, -0.2, 1.3])
    U = rng.normal(size=(200, 3))
    U /= np.linalg.norm(U, axis=1)[:, None]
    U = U[U[:, 0] > -0.3]
    surf = center + 0.3 * U + rng.normal(scale=0.004, size=U.shape)
    outliers = center + rng.uniform(-0.75, 0.75, size=(160, 3))
    fit = fit_sphere_ransac(np.vstack([surf, outliers]), threshold=.022,
                            min_inliers=30, iterations=550)
    assert fit is not None
    assert np.linalg.norm(fit.center - center) < 0.015
    assert fit.inliers.sum() >= 0.8 * len(surf)


def test_kabsch_and_ransac():
    rng = np.random.default_rng(42)
    X = rng.uniform([-1,-1,0], [1,1,2], size=(24, 3))
    rvec = np.array([0.13, 0.16, 0.29])
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([0.25, -0.13, 0.4])
    Y = X @ R.T + t + rng.normal(scale=0.003, size=X.shape)
    Y[3] += [1.2, 1.0, -0.4]
    Rout, tout, mask, residuals = ransac_transform(X, Y, threshold=.04)
    assert mask.sum() == 23
    assert np.linalg.norm(Rout - R) < 0.008
    assert np.linalg.norm(tout - t) < 0.009


def test_synthetic_image_ellipse_detection():
    ellipse = ellipse_from_sphere([0.30, -0.12, 3.4])
    image = np.full((1200, 1920, 3), 15, np.uint8)
    cv2.ellipse(image, ellipse, (195, 180, 145), -1, cv2.LINE_AA)
    cx, cy = ellipse[0]
    roi = (int(cx - 160), int(cy - 150), 320, 300)
    fitted, quality, _ = detect_ellipse(image, roi=roi)
    assert fitted is not None
    recovered = sphere_center_from_ellipse(fitted)
    assert np.linalg.norm(recovered - [0.30, -0.12, 3.4]) < 0.10
    assert quality['angular_coverage'] > .7


def test_ply_roundtrip(tmp_path):
    xyz = np.array([[1, 2, 3], [4, 5, 6]], dtype=float)
    p = tmp_path / 'test.ply'
    write_colored_ply(p, xyz, np.array([True, False]))
    assert np.allclose(read_points(p), xyz)


def test_timestamps():
    assert abs(timestamp_from_name('camera_1700000000123.jpg') - 1700000000.123) < 1e-6
    assert abs(timestamp_from_name('lidar_1700000000.123456.png') - 1700000000.123456) < 1e-6
