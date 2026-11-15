"""Sphere silhouette detector for the provided bright-yellow-ball camera sequence.

The HSV step ONLY proposes an ROI; final ellipse is fitted to Canny edges by RANSAC.
Strong false positives and partially cropped spheres are deliberately rejected.
"""
from __future__ import annotations
import cv2
import numpy as np
from .vision import fit_ellipse_ransac, ellipse_residual


def detect_yellow_sphere(image, edge_threshold=3.5, min_color_area=2600):
    H, W = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bright = cv2.inRange(hsv, np.array([17, 85, 90], np.uint8),
                        np.array([43, 255, 255], np.uint8))
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for c in contours:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        if area < min_color_area or w < 55 or h < 55 or y+h > H*.90:
            continue
        if x <= 3 or x + w >= W - 3: # incomplete silhouette
            continue
        candidates.append((area,x,y,w,h))
    if not candidates:
        return None, None, None, 'no_full_yellow_ball_candidate'
    candidates.sort(reverse=True)
    # The largest high-saturation area away from the ground is usually the ball.
    _, bx, by, bw, bh = candidates[0]
    diameter = max(bw, bh)
    pad = int(diameter*.65+45)
    x0,y0 = max(0,bx-pad),max(0,by-pad)
    x1,y1 = min(W,bx+bw+pad),min(H,by+bh+pad)
    crop = image[y0:y1,x0:x1]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7,7), 1.6)
    minimum = max(35,int(diameter*.31))
    maximum = int(diameter*.94)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2,
                              minDist=max(75,int(diameter*.55)),
                              param1=130, param2=34,
                              minRadius=minimum,maxRadius=maximum)
    if circles is None:
        return None, (x0,y0,x1-x0,y1-y0), None, 'no_circle_seed'
    target = np.array([bx+bw/2-x0, by+bh/2-y0])
    circle_candidates = []
    for cx,cy,r in circles[0]:
        dist = np.linalg.norm(np.array([cx,cy])-target)
        if dist > max(80,diameter*.65) or 2*r < diameter*.75:
            continue
        circle_candidates.append((dist/diameter + .2*abs(2*r-diameter)/diameter,cx,cy,r))
    if not circle_candidates:
        return None, (x0,y0,x1-x0,y1-y0), None, 'circle_seed_far_from_color'
    _, cx, cy, rad = min(circle_candidates)
    # Candidate ellipse contour points from the outer silhouette only.
    edges = cv2.Canny(gray, 50, 135)
    ys, xs = np.where(edges > 0)
    distances = np.hypot(xs-cx,ys-cy)
    shell = np.abs(distances-rad) < max(12,rad*.18)
    p = np.column_stack((xs[shell]+x0,ys[shell]+y0)).astype(np.float32)
    if len(p) < 75:
        return None,(x0,y0,x1-x0,y1-y0),None,'insufficient_canny_silhouette'
    if len(p) > 2800:
        p = p[np.linspace(0,len(p)-1,2800).astype(int)]
    result = fit_ellipse_ransac(p,iterations=190,threshold=edge_threshold,
                                min_axis=max(50,rad*1.0),seed=2026,
                                min_coverage=.53)
    if result is None:
        return None,(x0,y0,x1-x0,y1-y0),None,'ellipse_ransac_failed'
    ellipse, mask, rmse, coverage = result
    (ex,ey),(a,b),angle = ellipse
    if (np.linalg.norm(np.array([ex-x0,ey-y0])-np.array([cx,cy])) > rad*.32 or
        min(a,b)/max(a,b) < .70 or
        not (.69*rad < (a+b)/4 < 1.33*rad)):
        return None,(x0,y0,x1-x0,y1-y0),None,'ellipse_inconsistent_with_sphere'
    quality = {'edge_inliers':int(mask.sum()),'ellipse_rmse_px':round(rmse,4),
               'angular_coverage':round(coverage,3),
               'candidate_score':round(float(mask.sum())/max(1,len(p)),3),
               'hough_radius_px':float(rad)}
    return ellipse,(x0,y0,x1-x0,y1-y0),quality,'ok'
