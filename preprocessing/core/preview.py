"""Optional montage visualisation used by the pipeline."""

import os

import cv2
import numpy as np

from core.pnp import ROI, gaze_to_2d  # noqa: F401  (kept for callers)


def make_face_tile(patch, lm_warped, gaze2d):
    tile = patch.copy()
    sx = 224.0 / ROI[0]
    sy = 224.0 / ROI[1]
    for (x, y) in lm_warped:
        cv2.circle(tile, (int(x * sx), int(y * sy)), 2, (0, 255, 0), -1)
    (h, w) = tile.shape[:2]
    length = np.min([h, w]) / 2.0
    pos = (int(w / 2.0), int(h / 2.0))
    dx = -length * np.sin(gaze2d[1]) * np.cos(gaze2d[0])
    dy = -length * np.sin(gaze2d[0])
    cv2.arrowedLine(tile, tuple(np.round(pos).astype(int)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)),
                    (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.2)
    return tile


def make_orig_tile(orig, lm68, sub6, reproj):
    lm_min = lm68.min(axis=0).astype(int)
    lm_max = lm68.max(axis=0).astype(int)
    m = 120
    x1 = max(0, lm_min[0] - m)
    y1 = max(0, lm_min[1] - m)
    x2 = min(orig.shape[1], lm_max[0] + m)
    y2 = min(orig.shape[0], lm_max[1] + m)
    crop = orig[y1:y2, x1:x2].copy()
    for (x, y) in lm68:
        cv2.circle(crop, (int(x) - x1, int(y) - y1), 2, (0, 0, 255), -1)
    for (x, y) in sub6:
        cv2.circle(crop, (int(x) - x1, int(y) - y1), 4, (0, 255, 0), 2)
    for (x, y) in reproj:
        cv2.drawMarker(crop, (int(x) - x1, int(y) - y1), (255, 0, 0),
                       cv2.MARKER_STAR, 9, 1, cv2.LINE_AA)
    crop_h = 224
    scale = crop_h / crop.shape[0]
    return cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), crop_h),
                      interpolation=cv2.INTER_AREA)


def save_montage(out_dir, subj, rows):
    flat = [np.hstack(r) if isinstance(r, tuple) else r for r in rows]
    width = max(r.shape[1] for r in flat)
    padded = []
    for r in flat:
        if r.shape[1] < width:
            r = cv2.copyMakeBorder(r, 0, 0, 0, width - r.shape[1],
                                   cv2.BORDER_CONSTANT, value=(128, 128, 128))
        padded.append(r)
    montage = np.vstack(padded)
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{subj}_viz.jpg")
    cv2.imwrite(path, montage)
    return path
