"""人脸检测 + 68 关键点 (dataset agnostic).

MTCNN (CUDA) 出人脸框，dlib 出 68 关键点；
detect() 返回 dict {image_path: (pts68 | None, reason)}。
"""

import cv2
import dlib
import numpy as np
from imutils import face_utils

_PREDICTOR = None
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


def worker_init(predictor_path, min_face):
    global _PREDICTOR, _MIN_FACE
    _PREDICTOR = dlib.shape_predictor(predictor_path)
    _MIN_FACE = min_face


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


def detect(files, mtcnn, pool, gpu_batch, gpu_conf, print_freq):
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

        boxes_per_img, probs_per_img = [], []
        if imgs:
            try:
                rgb = [cv2.cvtColor(im, cv2.COLOR_BGR2RGB) for im in imgs]
                out = mtcnn.detect(rgb)
                boxes_per_img, probs_per_img = out[0], out[1]
            except Exception:
                # batch失败就逐张来
                for im in imgs:
                    res = mtcnn.detect(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    boxes_per_img.append(to_box_arr(res[0]))
                    probs_per_img.append(to_prob_arr(res[1]))

        tasks = []
        for i, p in enumerate(valid):
            b = to_box_arr(boxes_per_img[i])
            pr = to_prob_arr(probs_per_img[i])[:len(b)]
            keep = np.isfinite(pr) & (pr >= gpu_conf)
            tasks.append((p, pick_largest_box(b[keep], pr[keep])))

        for fpath, pts, status in pool.imap_unordered(_lm_worker, tasks, chunksize=4):
            results[fpath] = (pts, status)

        done = start + len(batch_paths)
        if done % print_freq == 0 or done == n:
            print(f"  {done}/{n}", flush=True)
    return results
