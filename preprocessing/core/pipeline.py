"""Generic detect -> PnP -> h5 pipeline, driven by a dataset adapter.

A dataset adapter only has to provide subjects() and samples(subj); see
preprocessing/dataset_struct/mpii.py for the expected sample dict:

  {
    "key":        "p00/day01/0005"        # used for output naming
    "image":      "/abs/path.jpg"
    "camera":     (3,3) matrix
    "distortion": (n,1)
    "gaze_dir":   (3,) target3d - person3d, camera coordinates
    "landmarks":  optional (68,2) if the dataset ships them
  }
"""

import multiprocessing as mp
import os
import time

import cv2
import numpy as np

from core import detect as detect_mod
from core import loader, pnp
from core import preview as viz
from core.writers import H5Writer, write_failed, write_landmarks_csv


def _load_face_model(cfg):
    path = loader.resolve_path(cfg["paths"]["face_model"])
    full = np.loadtxt(path).astype(np.float64)
    return pnp.select_face_points(full), full


def _make_detector(cfg):
    opt, flt, rt = cfg["options"], cfg["filter"], cfg["runtime"]
    mode = opt.get("detector", "gpu")
    workers = rt.get("workers", 8) or 1
    initargs = (mode, opt.get("upsample", 1), flt.get("min_face", 40),
                loader.resolve_path(cfg["paths"]["face_detector"]),
                loader.resolve_path(cfg["paths"]["landmark_predictor"]))
    pool = mp.Pool(workers, initializer=detect_mod.worker_init, initargs=initargs)

    mtcnn = None
    if mode == "gpu":
        import torch
        from facenet_pytorch import MTCNN
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available, set options.detector: cnn")
        mtcnn = MTCNN(device="cuda", select_largest=False, post_process=False)
    return mode, pool, mtcnn


def _detect(mode, files, pool, mtcnn, cfg):
    rt, flt = cfg["runtime"], cfg["filter"]
    if mode == "gpu":
        return detect_mod.run_gpu(files, mtcnn, pool,
                                  rt.get("gpu_batch", 32),
                                  flt.get("gpu_conf", 0.5),
                                  rt.get("print_freq", 50))
    return detect_mod.run_cnn(files, pool, rt.get("print_freq", 50))


def run(cfg, dataset):
    in_root, dirs = loader.output_layout(cfg)
    opt, flt, rt = cfg["options"], cfg["filter"], cfg["runtime"]

    need_mesh = bool(opt.get("rotated_mesh_vis", False))
    viz_n = opt.get("viz_per_subject", 0) or 0
    need_viz = viz_n > 0
    keys = ["landmarks", "normalized_dataset", "failed"]
    if need_mesh:
        keys.append("rotated_mesh_vis")
    if need_viz:
        keys.append("viz")
    loader.ensure_dirs(dirs, keys=keys)

    face_model, face_model_full = _load_face_model(cfg)

    limit = flt.get("limit", 0) or 0
    filters_on = flt.get("enable", True)
    max_reproj = flt.get("max_reproj", 0) or 0
    skip_undistort = flt.get("skip_undistort", False)
    overwrite = opt.get("overwrite", False)
    save_lm = opt.get("save_landmarks", True)

    mode, pool, mtcnn = _make_detector(cfg)
    print(f"dataset={dataset.name}  detector={mode}  out={dirs['root']}", flush=True)

    try:
        for subj in dataset.subjects():
            samples = dataset.samples(subj)
            if limit:
                samples = samples[:limit]
            if not samples:
                print(f"[{subj}] no samples, skip", flush=True)
                continue

            h5_path = os.path.join(dirs["normalized_dataset"], f"{subj}.h5")
            lm_path = os.path.join(dirs["landmarks"], f"{subj}.csv")
            if os.path.exists(h5_path) and not overwrite:
                print(f"[{subj}] already done ({h5_path}), skip", flush=True)
                continue

            t0 = time.time()
            print(f"[{subj}] {len(samples)} samples ...", flush=True)

            if all(s.get("landmarks") is not None for s in samples):
                lm = {s["key"]: (np.asarray(s["landmarks"], dtype=np.float32), "ok")
                      for s in samples}
            else:
                res = _detect(mode, [s["image"] for s in samples], pool, mtcnn, cfg)
                lm = {s["key"]: res.get(s["image"], (None, "missing")) for s in samples}

            if save_lm:
                write_landmarks_csv(lm_path,
                                    [(s["key"], lm[s["key"]][0], lm[s["key"]][1])
                                     for s in samples])

            writer = H5Writer(h5_path)
            rejected = []
            rows = []
            for s in samples:
                pts, reason = lm[s["key"]]
                if pts is None:
                    rejected.append((s["key"], reason))
                    continue

                camera = s["camera"]
                distortion = np.zeros((5, 1)) if skip_undistort else s["distortion"]
                sub6 = pts[pnp.LM68_USE].astype(np.float32).reshape(6, 1, 2)
                rvec, tvec, reproj, err = pnp.estimate_head_pose(
                    sub6, face_model, camera, distortion)
                if rvec is None:
                    rejected.append((s["key"], "pnp_failed"))
                    continue
                if filters_on and max_reproj and err > max_reproj:
                    rejected.append((s["key"], "reproj_too_big"))
                    continue

                if need_mesh:
                    mesh_path = os.path.join(
                        dirs["rotated_mesh_vis"], os.path.splitext(s["key"])[0] + ".txt")
                    os.makedirs(os.path.dirname(mesh_path), exist_ok=True)
                    np.savetxt(mesh_path, pnp.rotated_mesh(rvec, tvec, face_model_full))

                img = cv2.imread(s["image"])
                warped, hr_norm, _, R, lm_warped = pnp.normalize_face(
                    img, face_model, pts, rvec, tvec, camera)
                gdir = R @ np.asarray(s["gaze_dir"], dtype=np.float64).reshape(3)
                gaze2d = pnp.gaze_to_2d(gdir)
                head2d = pnp.head_to_2d(hr_norm)
                writer.add(warped, gaze2d, head2d, R, lm_warped)

                if need_viz and writer.n <= viz_n:
                    o = viz.make_orig_tile(img, pts, sub6.reshape(6, 2), reproj)
                    p = viz.make_face_tile(warped, lm_warped, gaze2d)
                    rows.append((o, p))

            writer.close()
            if need_viz and rows:
                viz.save_montage(dirs["viz"], subj, rows)
            write_failed(os.path.join(dirs["failed"], f"{subj}_rejected.txt"), rejected)

            print(f"[{subj}] kept={writer.n}/{len(samples)} "
                  f"rejected={len(rejected)}  {time.time() - t0:.0f}s -> {h5_path}",
                  flush=True)
    finally:
        pool.close()
        pool.join()
