"""Shared detect -> PnP -> h5 pipeline.

Any dataset adapter (see preprocessing/dataset_struct/) yields sample dicts:

  {
    "key":        "p00/day01/0005"      # used for output naming
    "image":      (H,W,3) BGR uint8 array, or None if unreadable
    "camera":     (3,3) matrix
    "distortion": (n,1)
    "gaze_dir":   (3,) target3d - person3d, camera coordinates
  }

"""

import itertools
import time
from pathlib import Path
import cv2

import dlib
import numpy as np
import torch
from facenet_pytorch import MTCNN

from core import detect as detect_mod
from core import pnp
from core import preview as viz
from core.writers import H5Writer, write_failed, write_landmarks_csv

OUTPUT_SUBDIRS = (
    "landmarks",           # <subj>.csv
    "normalized_dataset",  # <subj>.h5
    "failed",              # <subj>_rejected.txt
    "rotated_mesh_vis",    # <key>.txt, optional
    "viz",                 # <subj>_viz.jpg, optional
)


def run(cfg, dataset):
    paths, flt, opt, rt = cfg["paths"], cfg["filter"], cfg["options"], cfg["runtime"]

    out_root = paths.get("output_root")
    if out_root:
        out_root = Path(out_root).resolve()
    else:
        in_root = Path(paths["input_root"]).resolve()
        out_root = in_root.parent/(in_root.name.upper() + "_normalized")
    dirs = {name: out_root/name for name in OUTPUT_SUBDIRS}
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)#这不是有问题吗，，有些不是有开关吗

    ###get config
    face_model_full = np.loadtxt(Path(paths["face_model"]).resolve()).astype(np.float64)
    face_model = face_model_full[pnp.FM50_USE]

    filters_on = flt.get("enable", True)
    max_reproj = flt.get("max_reproj", 0) or 0
    skip_undistort = flt.get("skip_undistort", False)
    min_face = flt.get("min_face", 40)
    conf = flt.get("conf", 0.5)

    overwrite = opt.get("overwrite", False)
    save_lm = opt.get("save_landmarks", True)
    need_mesh = bool(opt.get("rotated_mesh_vis", False))
    viz_n = opt.get("viz_per_subject", 0) or 0
    need_viz = viz_n > 0

    gpu_batch = rt.get("gpu_batch", 32)
    print_freq = rt.get("print_freq", 50)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    mtcnn = MTCNN(device="cuda", select_largest=False, post_process=False)
    predictor = dlib.shape_predictor(str(Path(paths["landmark_predictor"]).expanduser()))
    print(f"dataset={dataset.name}  out={out_root}", flush=True)

    for subj in dataset.subjects():
        samples = dataset.samples(subj)
        samples = iter(samples)
        first = next(samples, None)
        if first is None:
            print(f"[{subj}] no samples, skip", flush=True)
            continue

        h5_path = dirs["normalized_dataset"]/f"{subj}.h5"
        if h5_path.exists() and not overwrite:
            print(f"[{subj}] already done ({h5_path}), skip", flush=True)
            continue

        t0 = time.time()
        writer = H5Writer(h5_path)
        lm_rows, rejected, viz_rows = [], [], []
        n_seen, last_print = 0, 0

        def consume(batch):
            nonlocal n_seen, last_print
            n_seen += len(batch)

            for s in batch:
                if s["image"] is None:
                    rejected.append((s["key"], "read_failed"))

            to_detect = [s for s in batch
                         if s["image"] is not None and s.get("landmarks") is None]
            boxes = detect_mod.detect_faces(
                [s["image"] for s in to_detect], mtcnn, conf) if to_detect else []
            box_of = {id(s): b for s, b in zip(to_detect, boxes)}

            for s in batch:
                if s["image"] is None:
                    lm_rows.append((s["key"], None, "read_failed"))
                    continue
                pts = s.get("landmarks")
                reason = "ok"
                if pts is None:
                    pts, reason = detect_mod.face_landmarks(
                        s["image"], box_of[id(s)], predictor, min_face)
                pts = None if pts is None else np.asarray(pts, dtype=np.float32)
                lm_rows.append((s["key"], pts, reason))
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
                    mesh_path = dirs["rotated_mesh_vis"]/Path(s["key"]).with_suffix(".txt")
                    mesh_path.parent.mkdir(parents=True, exist_ok=True)
                    np.savetxt(mesh_path, pnp.rotated_mesh(rvec, tvec, face_model_full))

                img = s["image"]
                warped, hr_norm, _, R, lm_warped = pnp.normalize_face(
                    img, face_model, pts, rvec, tvec, camera)
                gdir = R @ np.asarray(s["gaze_dir"], dtype=np.float64).reshape(3)
                n = gdir / np.linalg.norm(gdir)
                gaze2d = np.array([np.arcsin(-n[1]), np.arctan2(-n[0], -n[2])])
                        
                M = cv2.Rodrigues(hr_norm.reshape(1, 3))[0]
                Zv = M[:, 2]
                head2d = np.array([np.arcsin(Zv[1]), np.arctan2(Zv[0], Zv[2])])

                writer.add(warped, gaze2d, head2d, R, lm_warped)

                if need_viz and writer.n <= viz_n:
                    o = viz.make_orig_tile(img, pts, sub6.reshape(6, 2), reproj)
                    p = viz.make_face_tile(warped, lm_warped, gaze2d)
                    viz_rows.append((o, p))

            if print_freq and n_seen - last_print >= print_freq:
                print(f"  {n_seen}", flush=True)
                last_print = n_seen

        print(f"[{subj}] processing ...", flush=True)
        batch = [first]
        for s in samples:
            batch.append(s)
            if len(batch) >= gpu_batch:
                consume(batch)
                batch = []
        consume(batch)
        if save_lm:
            write_landmarks_csv(dirs["landmarks"]/f"{subj}.csv", lm_rows)
        if need_viz and viz_rows:
            viz.save_montage(dirs["viz"], subj, viz_rows)
        write_failed(dirs["failed"]/f"{subj}_rejected.txt", rejected)
        writer.close()

        print(f"[{subj}] kept={writer.n} rejected={len(rejected)} "
              f"{time.time() - t0:.0f}s -> {h5_path}", flush=True)
