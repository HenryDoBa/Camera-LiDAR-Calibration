"""Calibration geometry: fixed-radius sphere fit, pinhole sphere inversion, and SE(3)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares

RADIUS_M = 0.30
K = np.array([[1250.0, 0.0, 960.0], [0.0, 1250.0, 600.0], [0.0, 0.0, 1.0]])

@dataclass
class SphereFit:
    center: np.ndarray
    inliers: np.ndarray
    rmse: float


def sphere_centers_three_points(p: np.ndarray, radius: float = RADIUS_M):
    """Both possible sphere centers through three noncollinear surface points."""
    a, b, c = np.asarray(p, dtype=float)
    ab, ac = b - a, c - a
    normal = np.cross(ab, ac)
    n2 = float(normal @ normal)
    if n2 < 1e-12:
        return []
    # Circumcenter offset in the plane spanned by ab and ac.
    off = (np.cross(normal, ab) * (ac @ ac) +
           np.cross(ac, normal) * (ab @ ab)) / (2.0 * n2)
    circle_center = a + off
    circle_r2 = float(off @ off)
    h2 = radius * radius - circle_r2
    if h2 < -1e-9:
        return []
    h = np.sqrt(max(0.0, h2))
    d = normal / np.sqrt(n2)
    return [circle_center + h * d, circle_center - h * d]


def fit_sphere_ransac(points, radius=RADIUS_M, threshold=0.025, iterations=900,
                       min_inliers=12, seed=42, sample_limit=2500):
    """Robust known-radius sphere fit. IMPORTANT: crop to spherical-target ROI first.

    Returns None when evidence is insufficient; indices are relative to points.
    """
    xyz = np.asarray(points, dtype=float)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    if len(xyz) < max(6, min_inliers):
        return None
    rng = np.random.default_rng(seed)
    sample = xyz if len(xyz) <= sample_limit else xyz[rng.choice(len(xyz), sample_limit, replace=False)]
    best_center, best_n, best_error = None, 0, float('inf')
    for _ in range(iterations):
        try:
            triple = sample[rng.choice(len(sample), 3, replace=False)]
        except ValueError:
            break
        for cen in sphere_centers_three_points(triple, radius):
            errs = np.abs(np.linalg.norm(sample - cen, axis=1) - radius)
            mask = errs < threshold
            n = int(mask.sum())
            mse = float(np.mean(errs[mask] ** 2)) if n else float('inf')
            if n > best_n or (n == best_n and mse < best_error):
                best_center, best_n, best_error = cen, n, mse
    if best_center is None:
        return None
    # Refine on full ROI, using an M-estimator, then recompute consensus.
    for _ in range(3):
        e = np.abs(np.linalg.norm(xyz - best_center, axis=1) - radius)
        mask = e <= threshold
        if mask.sum() < min_inliers:
            return None
        subset = xyz[mask]
        res = least_squares(lambda center: np.linalg.norm(subset - center, axis=1) - radius,
                            best_center, loss='soft_l1', f_scale=max(0.005, threshold / 2),
                            max_nfev=70)
        best_center = res.x
    err = np.abs(np.linalg.norm(xyz - best_center, axis=1) - radius)
    mask = err < threshold
    if mask.sum() < min_inliers:
        return None
    return SphereFit(best_center, mask, float(np.sqrt(np.mean(err[mask] ** 2))))


def ellipse_to_conic(ellipse):
    """OpenCV ellipse: ((cx, cy), (full_axis1, full_axis2), angle_degrees)."""
    (cx, cy), (w, h), theta_deg = ellipse
    if min(w, h) <= 0:
        raise ValueError('Ellipse axes must be > 0')
    ang = np.deg2rad(theta_deg)
    co, si = np.cos(ang), np.sin(ang)
    rot = np.array([[co, -si], [si, co]])
    A = rot @ np.diag([4.0 / w ** 2, 4.0 / h ** 2]) @ rot.T
    m = np.array([cx, cy], dtype=float)
    return np.block([[A, (-A @ m)[:, None]], [(-m @ A)[None, :],
                     np.array([[m @ A @ m - 1.0]])]])


def sphere_center_from_ellipse(ellipse, K_matrix=K, radius=RADIUS_M):
    """Exact pinhole reconstruction from spherical silhouette, with no distortion.

    The calibrated conic of a sphere is proportional to
      C = cc^T - (||c||^2-r^2) I,
    whose single positive eigenvector is the center direction, with
      distance = r * sqrt(1 - mean(negative_evals) / positive_eval).
    This corrects the bias of interpreting the ellipse center as projected c.
    """
    C_pixels = ellipse_to_conic(ellipse)
    Q = K_matrix.T @ C_pixels @ K_matrix
    Q = 0.5 * (Q + Q.T)
    lam, vec = np.linalg.eigh(Q)
    if np.sum(lam > 0) == 2 and np.sum(lam < 0) == 1:
        lam, vec = np.linalg.eigh(-Q)
    if np.sum(lam > 0) != 1 or np.sum(lam < 0) != 2:
        raise ValueError('Fitted ellipse does not produce a valid tangent cone')
    p = np.argmax(lam)
    axis = vec[:, p]
    if axis[2] < 0:
        axis = -axis
    neg = np.mean(lam[lam < 0])
    depth = radius * np.sqrt(1.0 - neg / lam[p])
    return depth * axis


def rigid_transform(src: np.ndarray, dst: np.ndarray):
    """Column-vector convention: dst ~= R @ src + t (Kabsch / SVD)."""
    X, Y = np.asarray(src, float), np.asarray(dst, float)
    if X.shape != Y.shape or X.ndim != 2 or X.shape[1] != 3 or len(X) < 3:
        raise ValueError('Need >=3 corresponding 3D points')
    xm, ym = X.mean(axis=0), Y.mean(axis=0)
    H = (X - xm).T @ (Y - ym)
    U, _, Vt = np.linalg.svd(H)
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    t = ym - R @ xm
    return R, t


def ransac_transform(src, dst, threshold=0.12, iterations=1500, seed=42):
    X, Y = np.asarray(src, float), np.asarray(dst, float)
    if len(X) < 6:
        raise ValueError('Assignment requires >=6 manually approved frame pairs')
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(iterations):
        idx = rng.choice(len(X), 3, replace=False)
        # Exclude nearly collinear triplets (rotation cannot be determined stably).
        if np.linalg.norm(np.cross(X[idx[1]] - X[idx[0]], X[idx[2]] - X[idx[0]])) < 1e-3:
            continue
        R, t = rigid_transform(X[idx], Y[idx])
        residual = np.linalg.norm(X @ R.T + t - Y, axis=1)
        inside = residual < threshold
        n = int(inside.sum())
        score = (n, -float(np.sum(np.minimum(residual, threshold) ** 2)))
        if best is None or score > best[0]:
            best = (score, inside)
    if best is None or best[1].sum() < 3:
        raise ValueError('Insufficient non-collinear reliable 3D matches')
    inliers = best[1]
    for _ in range(3):
        R, t = rigid_transform(X[inliers], Y[inliers])
        residual = np.linalg.norm(X @ R.T + t - Y, axis=1)
        new_inliers = residual < threshold
        if new_inliers.sum() < 3 or np.array_equal(new_inliers, inliers):
            break
        inliers = new_inliers
    R, t = rigid_transform(X[inliers], Y[inliers])
    residual = np.linalg.norm(X @ R.T + t - Y, axis=1)
    return R, t, inliers, residual
