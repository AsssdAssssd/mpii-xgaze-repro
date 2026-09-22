import argparse
import csv
import glob
import multiprocessing as mp
import os
import sys
import time

import cv2
import dlib
import numpy as np
from imutils import face_utils

BASE = os.path.dirname(os.path.abspath(__file__))
MODULES_DIR = os.path.join(BASE, "..", "modules")
NN_DETECTOR = os.path.join(MODULES_DIR, "mmod_human_face_detector.dat")
PREDICTOR = os.path.join(MODULES_DIR, "shape_predictor_68_face_landmarks.dat")

_DETECTOR = None
_PREDICTOR = None
_UPSAMPLE = 1
_MIN_FACE = 40


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mpii_root", required=True)
    p.add_argument("--out_dir", default="landmarks")
    p.add_argument("--detector", choices=["gpu", "cnn"], default="gpu",
                   help="gpu: MTCNN on CUDA + dlib landmarks; cnn: dlib CNN detector")
    p.add_argument("--upsample", type=int, default=1,
                   help="detector upsample (cnn mode only)")
    p.add_argument("--subjects", default="", help="comma list, e.g. p00,p01")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--min_face", type=int, default=40)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--gpu_batch", type=int, default=32)
    p.add_argument("--gpu_conf", type=float, default=0.5)
    p.add_argument("--print_freq", type=int, default=50)
    p.add_argument("--overwrite", action="store_true",
                   help="reprocess subjects whose CSV already exists")
    return p.parse_args()


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


def detect_one(image_bgr, detector, predictor, upsample, min_face):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faces = detector(rgb, upsample)
    if len(faces) == 0:
        return None, "no_face"
    best = None
    best_area = 0
    for f in faces:
        r = f.rect
        area = r.width() * r.height()
        if area > best_area:
            best = r
            best_area = area
    if best is None or min(best.width(), best.height()) < min_face:
        return None, "face_too_small"
    shape = predictor(rgb, best)
    pts = face_utils.shape_to_np(shape)
    if pts.shape[0] != 68:
        return None, "bad_shape"
    return pts.astype(np.float32), "ok"


def _init_worker(mode, upsample, min_face):
    global _DETECTOR, _PREDICTOR, _UPSAMPLE, _MIN_FACE
    _UPSAMPLE = upsample
    _MIN_FACE = min_face
    if mode == "cnn":
        cv2.setNumThreads(1)
        _DETECTOR = dlib.cnn_face_detection_model_v1(NN_DETECTOR)
    _PREDICTOR = dlib.shape_predictor(PREDICTOR)


def _detect_worker(fpath):
    img = cv2.imread(fpath)
    if img is None:
        return fpath, None, "read_failed"
    pts, status = detect_one(img, _DETECTOR, _PREDICTOR, _UPSAMPLE, _MIN_FACE)
    return fpath, pts, status


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
    shape = _PREDICTOR(rgb, rect)
    pts = face_utils.shape_to_np(shape)
    if pts.shape[0] != 68:
        return fpath, None, "bad_shape"
    return fpath, pts.astype(np.float32), "ok"


def normalize_batch_det(out, n):
    boxes_all, probs_all = out[0], out[1]
    per = []
    for i in range(n):
        b = np.asarray(boxes_all[i], dtype=np.float64).reshape(-1, 4)
        p = np.asarray(probs_all[i], dtype=np.float64).reshape(-1)
        per.append((b, p))
    return per


def write_subject(out_dir, subj, mpii_root, files, results):
    csv_path = os.path.join(out_dir, f"{subj}.csv")
    fail_path = os.path.join(out_dir, f"{subj}_failed.txt")
    header = ["path"] + [f"x{i:02d}" for i in range(68)] + [f"y{i:02d}" for i in range(68)]
    ok = 0
    bad = 0
    with open(csv_path, "w", newline="") as fcsv, open(fail_path, "w") as ffail:
        writer = csv.writer(fcsv)
        writer.writerow(header)
        for fpath in files:
            rel = os.path.relpath(fpath, mpii_root).replace("\\", "/")
            pts, status = results.get(fpath, (None, "missing"))
            if pts is None:
                ffail.write(rel + "\t" + status + "\n")
                bad += 1
            else:
                row = [rel] + [f"{v:.2f}" for v in pts[:, 0]] + [f"{v:.2f}" for v in pts[:, 1]]
                writer.writerow(row)
                ok += 1
    return ok, bad, csv_path


def run_gpu(files, results, mtcnn, pool, args):
    n = len(files)
    t0 = time.time()
    for start in range(0, n, args.gpu_batch):
        batch_paths = files[start:start + args.gpu_batch]

        imgs = []
        valid_paths = []
        for p in batch_paths:
            im = cv2.imread(p)
            if im is None:
                results[p] = (None, "read_failed")
            else:
                imgs.append(im)
                valid_paths.append(p)

        boxes_per_img = None
        probs_per_img = None
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
                boxes_per_img = []
                probs_per_img = []
                for im in imgs:
                    res = mtcnn.detect(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
                    boxes_per_img.append(to_box_arr(res[0]))
                    probs_per_img.append(to_prob_arr(res[1]))

        tasks = []
        for i, p in enumerate(valid_paths):
            if boxes_per_img is None:
                box = None
            else:
                b = to_box_arr(boxes_per_img[i])
                pr = to_prob_arr(probs_per_img[i])
                pr = pr[: len(b)]
                ok = np.isfinite(pr) & (pr >= args.gpu_conf)
                box = pick_largest_box(b[ok], pr[ok])
            tasks.append((p, box))

        for item in pool.imap_unordered(_lm_worker, tasks, chunksize=4):
            results[item[0]] = (item[1], item[2])

        processed = start + len(batch_paths)
        if processed % args.print_freq == 0 or processed == n:
            el = time.time() - t0
            speed = processed / el if el > 0 else 0.0
            print(f"  {processed}/{n}  {speed:.1f} img/s  {el:.0f}s", flush=True)


def run_cpu(files, results, pool, args):
    for n, (fpath, pts, status) in enumerate(
        pool.imap_unordered(_detect_worker, files, chunksize=8)
    ):
        results[fpath] = (pts, status)
        if (n + 1) % args.print_freq == 0 or (n + 1) == len(files):
            print(f"  {n + 1}/{len(files)}", flush=True)


def main():
    args = parse_args()
    if not os.path.isfile(PREDICTOR):
        print("missing predictor:", PREDICTOR, flush=True)
        sys.exit(1)

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if not subjects:
        subjects = ["p{:02d}".format(i) for i in range(15)]

    os.makedirs(args.out_dir, exist_ok=True)

    workers = args.workers
    if workers <= 0:
        workers = os.cpu_count() or 1

    print(f"detector={args.detector}  workers={workers}", flush=True)

    mtcnn = None
    if args.detector == "gpu":
        import torch
        from facenet_pytorch import MTCNN
        if not torch.cuda.is_available():
            print("CUDA not available, use --detector cnn instead", flush=True)
            sys.exit(1)
        mtcnn = MTCNN(device="cuda", select_largest=False, post_process=False)

    total_ok = 0
    total_bad = 0

    with mp.Pool(
        workers,
        initializer=_init_worker,
        initargs=(args.detector, args.upsample, args.min_face),
    ) as pool:
        for subj in subjects:
            subj_dir = os.path.join(args.mpii_root, subj)
            if not os.path.isdir(subj_dir):
                print("skip missing subject:", subj_dir, flush=True)
                continue

            files = sorted(glob.glob(os.path.join(subj_dir, "day*", "*.jpg")))
            if args.limit:
                files = files[: args.limit]
            if not files:
                print("no images for subject:", subj, flush=True)
                continue

            csv_path = os.path.join(args.out_dir, f"{subj}.csv")
            if os.path.exists(csv_path) and not args.overwrite:
                print(f"[{subj}] already done ({csv_path}), skipping (--overwrite to redo)", flush=True)
                continue

            t0 = time.time()
            print(f"[{subj}] {len(files)} images ...", flush=True)
            results = {}

            if args.detector == "gpu":
                run_gpu(files, results, mtcnn, pool, args)
            else:
                run_cpu(files, results, pool, args)

            ok, bad, csv_path = write_subject(args.out_dir, subj, args.mpii_root, files, results)
            el = time.time() - t0
            print(f"{subj}: ok={ok} bad={bad}  {el:.0f}s -> {csv_path}", flush=True)
            total_ok += ok
            total_bad += bad

    print(f"\ndone. total ok={total_ok} bad={total_bad}", flush=True)


if __name__ == "__main__":
    main()
