"""OpenCV Canny edge detection and robust silhouette ellipse fitting."""
from __future__ import annotations
import cv2
import numpy as np


def ellipse_residual(points, ellipse):
    """Approximate geometric radial residual, pixels, for OpenCV ellipse."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    (cx, cy), (w, h), angle = ellipse
    if min(w, h) < 2:
        return np.full(len(pts), np.inf)
    theta = np.deg2rad(angle)
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    local = (pts - np.array([cx, cy])) @ R.T
    rho = np.sqrt((local[:, 0] / (w / 2)) ** 2 +
                  (local[:, 1] / (h / 2)) ** 2)
    return np.linalg.norm(local, axis=1) * np.abs(1 - 1 / np.maximum(rho, 1e-10))


def angular_coverage(points, ellipse, sectors=16):
    (cx, cy), (_, _), angle = ellipse
    theta = np.deg2rad(angle)
    pts = np.asarray(points).reshape(-1, 2) - [cx, cy]
    ang = np.arctan2(pts[:, 1], pts[:, 0]) - theta
    buckets = np.floor(((ang + np.pi) % (2 * np.pi)) / (2 * np.pi) * sectors).astype(int)
    return len(np.unique(buckets)) / sectors


def fit_ellipse_ransac(points, iterations=140, threshold=2.75, min_axis=10,
                       seed=2026, min_coverage=0.50):
    """Fit from Canny contour pixels, robust to a moderate fraction of outliers."""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 15:
        return None
    rng = np.random.default_rng(seed)
    best, best_mask, best_score = None, None, (-1, -np.inf)
    # Include a full-contour initial fit, then random minimal-subset fits.
    for it in range(iterations + 1):
        sub = pts if it == 0 else pts[rng.choice(len(pts), 8, replace=False)]
        try:
            el = cv2.fitEllipse(sub.reshape(-1, 1, 2))
        except cv2.error:
            continue
        (_, _), (w, h), _ = el
        if min(w, h) < min_axis or min(w, h) / max(w, h) < 0.52:
            continue
        errs = ellipse_residual(pts, el)
        mask = errs < threshold
        n = int(mask.sum())
        if n < 15 or angular_coverage(pts[mask], el) < min_coverage:
            continue
        score = (n, -float(np.mean(errs[mask] ** 2)))
        if score > best_score:
            best, best_mask, best_score = el, mask, score
    if best is None:
        return None
    for _ in range(2):
        try:
            new = cv2.fitEllipse(pts[best_mask].reshape(-1, 1, 2))
        except cv2.error:
            break
        e = ellipse_residual(pts, new)
        m = e < threshold
        if m.sum() < 15 or angular_coverage(pts[m], new) < min_coverage:
            break
        best, best_mask = new, m
    rmse = float(np.sqrt(np.mean(ellipse_residual(pts[best_mask], best) ** 2)))
    return best, best_mask, rmse, angular_coverage(pts[best_mask], best)


def _clamp_roi(roi, image_shape):
    H, W = image_shape[:2]
    x, y, w, h = map(int, roi)
    x = min(max(0, x), W - 1)
    y = min(max(0, y), H - 1)
    return x, y, max(1, min(w, W - x)), max(1, min(h, H - y))


def dynamic_roi(ellipse, shape, factor=3.7):
    (cx, cy), (w, h), _ = ellipse
    side = max(w, h) * factor
    return _clamp_roi((cx - side / 2, cy - side / 2, side, side), shape)


def detect_ellipse(image, roi=None, previous=None, threshold=2.75):
    """Return ellipse, quality dict, Canny image (full shape).

    IMPORTANT: the ROI should isolate the ball (or its predicted position).
    In cluttered scenes manual ROI corrections may be necessary.
    """
    H, W = image.shape[:2]
    roi = _clamp_roi(roi or (0, 0, W, H), image.shape)
    x, y, w, h = roi
    part = image[y:y + h, x:x + w]
    gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY) if part.ndim == 3 else part
    gray = cv2.GaussianBlur(gray, (5, 5), 1.15)
    median = float(np.median(gray))
    low = max(20, min(100, int(0.65 * median)))
    high = max(70, min(220, int(1.45 * median)))
    edges = cv2.Canny(gray, low, high)
    edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    chosen, chosen_quality, best = None, None, -float('inf')
    target = np.array(previous[0] if previous is not None else (x + w / 2, y + h / 2))
    for cnt in cnts:
        n = len(cnt)
        if n < 22 or cv2.arcLength(cnt, True) < 35:
            continue
        # Sparse decimation improves speed without changing the silhouette.
        p = cnt.reshape(-1, 2)[::max(1, n // 320)].astype(np.float32)
        if len(p) < 15:
            continue
        p[:, 0] += x
        p[:, 1] += y
        result = fit_ellipse_ransac(p, iterations=45, threshold=threshold,
                                    min_axis=10, seed=2026 + n % 97)
        if result is None:
            continue
        el, mask, rmse, coverage = result
        (cx, cy), (ew, eh), _ = el
        # Candidate may be slightly outside ROI if boundary is partially occluded.
        if not (x - ew / 3 <= cx < x + w + ew / 3 and
                y - eh / 3 <= cy < y + h + eh / 3):
            continue
        if ew > 1.7 * w or eh > 1.7 * h:
            continue
        dist = float(np.linalg.norm(np.array([cx, cy]) - target))
        size = max(ew, eh)
        if previous is not None:
            old_size = max(previous[1])
            if size < old_size * 0.4 or size > old_size * 2.5:
                continue
        # Prefer a closed, large, smooth boundary near the predicted sphere.
        score = 28 * coverage + 0.10 * int(mask.sum()) - 2.1 * rmse - 0.10 * dist
        if previous is not None:
            score -= 6 * abs(np.log(size / max(previous[1])))
        if score > best:
            best, chosen = score, el
            chosen_quality = {'edge_inliers': int(mask.sum()),
                              'ellipse_rmse_px': round(rmse, 4),
                              'angular_coverage': round(coverage, 3),
                              'candidate_score': round(score, 3)}
    full = np.zeros((H, W), dtype=np.uint8)
    full[y:y + h, x:x + w] = edges
    return chosen, chosen_quality, full


def annotate(image, ellipse, roi=None, status=None):
    out = image.copy()
    if roi is not None:
        x, y, w, h = _clamp_roi(roi, out.shape)
        cv2.rectangle(out, (x, y), (x + w, y + h), (0, 200, 255), 2)
    if ellipse is not None:
        cv2.ellipse(out, ellipse, (20, 60, 255), 3)
        center = tuple(map(int, ellipse[0]))
        cv2.drawMarker(out, center, (0, 255, 0), cv2.MARKER_CROSS, 18, 2)
    if status:
        cv2.putText(out, status[:85], (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(out, status[:85], (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0, 255, 255), 2, cv2.LINE_AA)
    return out
