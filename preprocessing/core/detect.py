"""MTCNN face boxes + dlib 68 landmarks, operating on in-memory images."""

import cv2
import dlib
import numpy as np
from imutils import face_utils


def detect_faces(images, mtcnn, conf):
    """images: list of BGR arrays. Returns a list of (4,) boxes or None."""
    rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in images]
    try:
        boxes, probs = mtcnn.detect(rgb)
    except Exception:
        # a bad batch: fall back to one image at a time
        boxes, probs = [], []
        for im in rgb:
            b, p = mtcnn.detect(im)
            boxes.append(b)
            probs.append(p)
    if boxes is None:
        return [None]*len(images)
    return [_pick_box(b, p, conf) for b, p in zip(boxes, probs)]


def _pick_box(boxes, probs, conf):
    if boxes is None:
        return None
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)[:len(b)]
    keep = np.isfinite(p) & (p >= conf) & (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    b = b[keep]
    if len(b) == 0:
        return None
    area = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return b[int(np.argmax(area))]


def face_landmarks(image, box, predictor, min_face):
    """Return (pts68 float32, "ok") or (None, reason)."""
    if box is None:
        return None, "no_face"
    x1, y1, x2, y2 = box
    if min(x2 - x1, y2 - y1) < min_face:
        return None, "face_too_small"
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
    pts = face_utils.shape_to_np(predictor(rgb, rect))
    if pts.shape[0] != 68:
        return None, "bad_shape"
    return pts.astype(np.float32), "ok"
