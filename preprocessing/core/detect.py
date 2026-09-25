"""Generic face detection + 68 landmark detection (dataset agnostic).

Two modes, same as the original MPII script:
  gpu : facenet MTCNN on CUDA for face boxes + dlib for 68 landmarks
  cnn : dlib CNN face detector + dlib 68 landmarks (CPU, multiprocessing)

Both return a dict {image_path: (pts68 | None, reason)}.
"""

import cv2
import dlib
import numpy as np
from imutils import face_utils

_DETECTOR = None
_PREDICTOR = None
_UPSAMPLE = 1
_MIN_FACE = 40


def to_box_arr(x):
    a = np.asarray(x, dtype=np.float64)
    if a.size == 0 or a.size % 4 != 0:
        return np.zeros((0, 4), dtype=np.float64)
    return a.reshape(-1, 4)


def to_prob_arr(x):
    return np.asarray(x, dtype=np.float64).reshape(-1)


def pick_largest_box(boxes, probs):
    b = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    p = np.asarray(probs, dtype=np.float64).reshape(-1)
    if b.size == 0:
        return None
    ok = np.isfinite(p) & (p > 0) & (b[:, 2] > b[:, 0]) & (b[:, 3] > b[:, 1])
    area = np.zeros(len(b))
    area[ok] = (b[ok, 2] - b[ok, 0]) * (b[ok, 3] - b[ok, 1])
    i = int(np.argmax(area))
    if area[i] <= 0:
        return None
    return b[i]


def worker_init(mode, upsample, min_face, detector_path, predictor_path):
    global _DETECTOR, _PREDICTOR, _UPSAMPLE, _MIN_FACE
    _UPSAMPLE = upsample
    _MIN_FACE = min_face
    if mode == "cnn":
        cv2.setNumThreads(1)
        _DETECTOR = dlib.cnn_face_detection_model_v1(detector_path)
    _PREDICTOR = dlib.shape_predictor(predictor_path)


def _cnn_worker(fpath):
    img = cv2.imread(fpath)
    if img is None:
        return fpath, None, "read_failed"
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces = _DETECTOR(rgb, _UPSAMPLE)
    if len(faces) == 0:
        return fpath, None, "no_face"
    best, best_area = None, 0
    for f in faces:
        r = f.rect
        area = r.width() * r.height()
        if area > best_area:
            best, best_area = r, area
    if best is None or min(best.width(), best.height()) < _MIN_FACE:
        return fpath, None, "face_too_small"
    pts = face_utils.shape_to_np(_PREDICTOR(rgb, best))
    if pts.shape[0] != 68:
        return fpath, None, "bad_shape"
    return fpath, pts.astype(np.float32), "ok"


def _lm_worker(task):
    fpath, box = task
    img = cv2.imread(fpath)
    if img is None:
        return fpath, None, "read_failed"
    if box is None:
        return fpath, None, "no_face"
    x1, y1, x2, y2 = box
    if min(x2 - x1, y2 - y1) < _MIN_FACE:
        return fpath, None, "face_too_small"
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
    pts = face_utils.shape_to_np(_PREDICTOR(rgb, rect))
    if pts.shape[0] != 68:
        return fpath, None, "bad_shape"
    return fpath, pts.astype(np.float32), "ok"


def run_gpu(files, mtcnn, pool, gpu_batch, gpu_conf, print_freq):
    results = {}
    n = len(files)
    for start in range(0, n, gpu_batch):
        batch_paths = files[start:start + gpu_batch]
        imgs, valid = [], []
        for p in batch_paths:
            im = cv2.imread(p)
            if im is None:
                results[p] = (None, "read_failed")
            else:
                imgs.append(im)
                valid.append(p)

        boxes_per_img, probs_per_img = None
        if imgs:
            rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
            try:
                out = mtcnn.detect(rgb)
                if not isinstance(out, (list, tuple)) or len(out) < 2:
                    raise RuntimeError("unexpected mtcnn.detect output")
                boxes_per_img, probs_per_img = out[0], out[1]
            except Exception:
                boxes_per_img = None
            if boxes_per_img is None:
                boxes_per_img, probs_per_img = [], []
                for im in imgs:
                    res = mtcnn.detect(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    boxes_per_img.append(to_box_arr(res[0]))
                    probs_per_img.append(to_prob_arr(res[1]))

        tasks = []
        for i, p in enumerate(valid):
            if boxes_per_img is None:
                box = None
            else:
                b = to_box_arr(boxes_per_img[i])
                pr = to_prob_arr(probs_per_img[i])[:len(b)]
                keep = np.isfinite(pr) & (pr >= gpu_conf)
                box = pick_largest_box(b[keep], pr[keep])
            tasks.append((p, box))

        for fpath, pts, status in pool.imap_unordered(_lm_worker, tasks, chunksize=4):
            results[fpath] = (pts, status)

        done = start + len(batch_paths)
        if done % print_freq == 0 or done == n:
            print(f"  {done}/{n}", flush=True)
    return results


def run_cnn(files, pool, print_freq):
    results = {}
    for n, (fpath, pts, status) in enumerate(
            pool.imap_unordered(_cnn_worker, files, chunksize=8)):
        results[fpath] = (pts, status)
        if (n + 1) % print_freq == 0 or (n + 1) == len(files):
            print(f"  {n + 1}/{len(files)}", flush=True)
    return results
