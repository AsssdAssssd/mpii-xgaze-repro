import argparse
import csv as csvmod
import multiprocessing as mp
import os
import sys
import time

import cv2
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
FACE_MODEL_TXT = os.path.join(BASE, "face_model", "simple_face_model.txt")
SIX_FACE_MODEL_TXT = os.path.join(BASE, "face_model", "xgaze_face_model.txt")
LM68_USE = [36, 39, 42, 45, 31, 35]
FM50_USE = [20, 23, 26, 29, 15, 19]
FOCAL_NORM = 960
DIST_NORM = 600
ROI = (224, 224)

ID=0
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mpii_root", required=True)
    p.add_argument("--landmarks_dir", default="landmarks")
    p.add_argument("--out_dir", default="pnp_h5")
    p.add_argument("--mesh_dir", default="meshtxt")
    p.add_argument("--subjects", default="", help="comma list, e.g. p00,p01")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--max_reproj", type=float, default=8.0)
    p.add_argument("--skip_undistort", action="store_true")
    p.add_argument("--viz_per_subject", type=int, default=4)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument(
        "--nose_z", default="",
        help="comma list of per-subject nose z (index 0..12), e.g. "
             "0.9,0.8,...")
    return p.parse_args()


def estimate_head_pose(landmarks6, face_model, camera, distortion):
    ret, rvec, tvec = cv2.solvePnP(
        face_model, landmarks6, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
    if not ret:
        return None, None, None, None
    ret, rvec, tvec = cv2.solvePnP(
        face_model, landmarks6, camera, distortion, rvec, tvec, True)
    if not ret:
        return None, None, None, None
    proj, _ = cv2.projectPoints(face_model, rvec, tvec, camera, distortion)
    reproj = proj.reshape(6, 2)
    err = float(np.mean(np.linalg.norm(
        reproj - landmarks6.reshape(6, 2), axis=1)))

    return rvec, tvec, reproj, err


def export_rotated_face(rvec, tvec, face_model_full, path):
    R, _ = cv2.Rodrigues(rvec)
    rotated = (R @ face_model_full.T + tvec.reshape(3, 1)).T
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savetxt(path, rotated)


def normalize_data_face(img, face_model, landmarks68, hr, ht, cam):
    ht = ht.reshape((3, 1))
    hR = cv2.Rodrigues(hr)[0]
    Fc = np.dot(hR, face_model.T) + ht
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_center = np.mean(
        np.concatenate((two_eye_center, nose_center), axis=1),
        axis=1).reshape((3, 1))

    distance = np.linalg.norm(face_center)
    z_scale = DIST_NORM / distance
    cam_norm = np.array([
        [FOCAL_NORM, 0, ROI[0] / 2],
        [0, FOCAL_NORM, ROI[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(cam)))
    img_warped = cv2.warpPerspective(img, W, ROI)

    hR_norm = np.dot(R, hR)
    hr_norm = cv2.Rodrigues(hR_norm)[0].reshape(3)

    det_point = landmarks68.reshape(-1, 1, 2)
    lm_warped = cv2.perspectiveTransform(det_point, W).reshape(-1, 2)

    return img_warped, hr_norm, face_center, R, lm_warped


def gaze_to_2d(gvec):
    n = gvec / np.linalg.norm(gvec)
    return np.array([np.arcsin(-n[1]), np.arctan2(-n[0], -n[2])])


def head_to_2d(hr_norm):
    M = cv2.Rodrigues(hr_norm.reshape(1, 3))[0]
    Zv = M[:, 2]
    return np.array([np.arcsin(Zv[1]), np.arctan2(Zv[0], Zv[2])])


def draw_gaze(image_in, pitchyaw, thickness=2, color=(0, 0, 255)):
    (h, w) = image_in.shape[:2]
    length = np.min([h, w]) / 2.0
    pos = (int(w / 2.0), int(h / 2.0))
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv2.arrowedLine(image_in, tuple(np.round(pos).astype(int)),
                    tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)),
                    color, thickness, cv2.LINE_AA, tipLength=0.2)
    return image_in


def load_camera(mpii_root, subj):
    from scipy.io import loadmat
    path = os.path.join(mpii_root, subj, "Calibration", "Camera.mat")
    m = loadmat(path)
    camera = np.array(m["cameraMatrix"], dtype=np.float64).reshape(3, 3)
    distortion = np.array(m["distCoeffs"], dtype=np.float64).reshape(-1, 1)
    return camera, distortion


def make_face_tile(patch, lm_warped, gaze2d):
    tile = patch.copy()
    sx = 224.0 / ROI[0]
    sy = 224.0 / ROI[1]
    for (x, y) in lm_warped:
        cv2.circle(tile, (int(x * sx), int(y * sy)), 2, (0, 255, 0), -1)
    return draw_gaze(tile, gaze2d)


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
    crop = cv2.resize(crop, (max(1, int(crop.shape[1] * scale)), crop_h),
                      interpolation=cv2.INTER_AREA)
    return crop


def save_viz(out_dir, subj, rows):
    flat = []
    for r in rows:
        if isinstance(r, tuple):
            r = np.hstack(r)
        flat.append(r)
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


def process_subject(payload):
    subj = payload["subj"]
    rels = payload["rels"]
    lms = payload["lms"]
    fcs = payload["fcs"]
    gts = payload["gts"]
    camera = payload["camera"]
    distortion = np.zeros((5, 1), dtype=np.float64) \
        if payload["skip_undistort"] else payload["distortion"]
    out_dir = payload["out_dir"]
    viz_dir = payload["viz_dir"]
    max_reproj = payload["max_reproj"]
    viz_count = payload["viz_per_subject"]

    face_model = payload["face_model"]
    face_model_full = payload["face_model_full"]
    face_pts = face_model.reshape(6, 1, 3).astype(np.float64)
    mesh_dir = payload["mesh_dir"]

    h5_path = os.path.join(out_dir, f"{subj}.h5")
    f = None
    ds = {}

    rows = []
    kept = 0
    idx = 0
    rejected = {"read_failed": [], "pnp_failed": [], "reproj_too_big": []}
    errs = []
    t0 = time.time()

    for i, rel in enumerate(rels):
        lm68 = lms[i]
        fc = fcs[i]
        gt = gts[i]
        img = cv2.imread(os.path.join(payload["mpii_root"], rel))
        if img is None:
            rejected["read_failed"].append(rel)
            continue
        sub6 = lm68[LM68_USE].astype(np.float32).reshape(6, 1, 2)
        rvec, tvec, reproj, err = estimate_head_pose(
            sub6, face_pts, camera, distortion)
        if rvec is None:
            rejected["pnp_failed"].append(rel)
            continue
        errs.append(err)
        if err > max_reproj:
            rejected["reproj_too_big"].append(rel)
            continue

        mesh_path = os.path.join(mesh_dir, os.path.splitext(rel)[0] + ".txt")
        export_rotated_face(rvec, tvec, face_model_full, mesh_path)

        warped, hr_norm, face_center, R, lm_warped = normalize_data_face(
            img, face_model, lm68.astype(np.float32), rvec, tvec, camera)
        gdir = np.dot(R, (gt - fc).reshape(3, 1)).reshape(3)
        gaze2d = gaze_to_2d(gdir)
        head2d = head_to_2d(hr_norm)

        if f is None:
            import h5py
            os.makedirs(out_dir, exist_ok=True)
            f = h5py.File(h5_path, "w", libver="latest")
            ds["face_patch"] = f.create_dataset(
                "face_patch", shape=(0, ROI[0], ROI[1], 3),
                maxshape=(None, ROI[0], ROI[1], 3),
                chunks=(1, ROI[0], ROI[1], 3),
                compression="lzf", dtype=np.uint8)
            ds["face_gaze"] = f.create_dataset(
                "face_gaze", shape=(0, 2), maxshape=(None, 2), dtype=np.float32)
            ds["face_head_pose"] = f.create_dataset(
                "face_head_pose", shape=(0, 2), maxshape=(None, 2),
                dtype=np.float32)
            ds["face_mat_norm"] = f.create_dataset(
                "face_mat_norm", shape=(0, 3, 3), maxshape=(None, 3, 3),
                dtype=np.float32)
            ds["facial_landmarks"] = f.create_dataset(
                "facial_landmarks", shape=(0, 68, 2), maxshape=(None, 68, 2),
                dtype=np.float32)

        for name in ("face_patch", "face_gaze", "face_head_pose",
                     "face_mat_norm", "facial_landmarks"):
            ds[name].resize(idx + 1, axis=0)
        ds["face_patch"][idx] = warped
        ds["face_gaze"][idx] = gaze2d.astype(np.float32)
        ds["face_head_pose"][idx] = head2d.astype(np.float32)
        ds["face_mat_norm"][idx] = R.astype(np.float32)
        ds["facial_landmarks"][idx] = lm_warped.astype(np.float32)
        idx += 1
        kept += 1

        if kept <= viz_count:
            o = make_orig_tile(img, lm68, sub6.reshape(6, 2), reproj)
            p = make_face_tile(warped, lm_warped, gaze2d)
            rows.append((o, p))

    if f is not None:
        f.flush()
        f.swmr_mode = True
        f.close()

    viz_path = save_viz(viz_dir, subj, rows) if rows else None

    rej_lines = []
    for k, v in rejected.items():
        for rel in v:
            rej_lines.append(f"{rel}\t{k}")
    if rej_lines:
        with open(os.path.join(out_dir, f"{subj}_rejected.txt"), "w") as fh:
            fh.write("\n".join(rej_lines) + "\n")

    e = np.asarray(errs, dtype=np.float64)
    return {
        "subj": subj,
        "n": len(rels),
        "kept": kept,
        "rej_read": len(rejected["read_failed"]),
        "rej_pnp": len(rejected["pnp_failed"]),
        "rej_reproj": len(rejected["reproj_too_big"]),
        "mean": float(e.mean()) if len(e) else 0.0,
        "p95": float(np.percentile(e, 95)) if len(e) else 0.0,
        "h5": h5_path if f is not None else None,
        "viz": viz_path,
        "sec": time.time() - t0,
    }


def subj_index(subj):
    import re
    m = re.search(r"\d+", subj)
    return int(m.group()) if m else -1


def apply_nose_z(face_model, subj, nose_z):
    m = face_model.copy()
    idx = subj_index(subj)
    if 0 <= idx < len(nose_z):
        v = float(nose_z[idx])
        m[4, 2] = v
        m[5, 2] = -v
    return m


def build_subject_payload(args, subj, face_model, payload_face_full):
    csv_path = os.path.join(args.landmarks_dir, f"{subj}.csv")
    anno_path = os.path.join(args.mpii_root, subj, f"{subj}.txt")
    if not os.path.isfile(csv_path):
        print("missing", csv_path, flush=True)
        return None
    if not os.path.isfile(anno_path):
        print("missing", anno_path, flush=True)
        return None

    camera, distortion = load_camera(args.mpii_root, subj)

    nose_z = args.nose_z.split(",") if args.nose_z else []
    face_model = apply_nose_z(face_model, subj, nose_z)
    payload_face_full = apply_nose_z(payload_face_full, subj, nose_z)

    anno = {}
    with open(anno_path) as fh:
        for line in fh:
            t = line.split()
            anno[t[0]] = (
                np.array(t[21:24], dtype=np.float64),
                np.array(t[24:27], dtype=np.float64),
            )

    rels = []
    lms = []
    fcs = []
    gts = []
    with open(csv_path) as fh:
        reader = csvmod.reader(fh)
        next(reader)
        for row in reader:
            rel = row[0]
            key = rel[len(subj) + 1:] if rel.startswith(subj + "/") else rel
            if key not in anno:
                continue
            if args.limit and len(rels) >= args.limit:
                break
            fc, gt = anno[key]
            xs = np.array(row[1:69], dtype=np.float64)
            ys = np.array(row[69:137], dtype=np.float64)
            rels.append(rel)
            lms.append(np.stack([xs, ys], axis=1))
            fcs.append(fc)
            gts.append(gt)

    if not rels:
        return None

    return {
        "subj": subj,
        "face_model": face_model,
        "face_model_full": payload_face_full,
        "mpii_root": args.mpii_root,
        "out_dir": args.out_dir,
        "mesh_dir": args.mesh_dir,
        "viz_dir": os.path.join(args.out_dir, "viz"),
        "rels": rels,
        "lms": np.asarray(lms),
        "fcs": np.asarray(fcs),
        "gts": np.asarray(gts),
        "camera": camera,
        "distortion": distortion,
        "max_reproj": args.max_reproj,
        "skip_undistort": args.skip_undistort,
        "viz_per_subject": args.viz_per_subject,
    }


def main():
    args = parse_args()
    # face_model_all = np.loadtxt(FACE_MODEL_TXT).astype(np.float64)
    face_model = [[-43.673908 ,-1.839432 ,2.381032],[-17.790553, 1.839432 ,-2.381032],[17.806574, 1.905477, -2.486947],[43.657887 ,-1.905477, 2.486947],[-13.017980 ,35.639090 ,0.137085],[13.862097 ,35.612960, -0.137085]]
    # face_model= np.loadtxt(SIX_FACE_MODEL_TXT).astype(np.float64)

    # ratio= [0.930, 0.1, 0.863, 0.970, 0.887,
    #           0.893, 0.788, 0.977, 0.940, 0.874,
    #           0.770, 0.729, 0.721, 0.814, 0.747]
    face_model[4][1]=(face_model[2][0]-face_model[1][0])/ratio[ID]
    face_model[5][1]=face_model[4][1]
    print(face_model[4][1])
    face_model=np.array(face_model)
    face_model_all=face_model

    ratio=[]

    subjects = [s.strip() for s in args.subjects.split(",") if s.strip()]
    if not subjects:
        subjects = ["p{:02d}".format(i) for i in range(15)]

    os.makedirs(args.out_dir, exist_ok=True)

    payloads = []
    for subj in subjects:
        p = build_subject_payload(args, subj, face_model, face_model_all)
        if p is not None:
            payloads.append(p)

    if not payloads:
        print("nothing to do", flush=True)
        return

    workers = max(1, min(args.workers, len(payloads)))
    t0 = time.time()
    results = []
    if workers <= 1:
        for p in payloads:
            r = process_subject(p)
            results.append(r)
            print(f"[{r['subj']}] n={r['n']} kept={r['kept']} "
                  f"read_fail={r['rej_read']} pnp_fail={r['rej_pnp']} "
                  f"reproj_big={r['rej_reproj']} "
                  f"reproj={r['mean']:.2f}/{r['p95']:.2f}px "
                  f"{r['sec']:.0f}s -> {r['h5']}", flush=True)
    else:
        with mp.Pool(workers) as pool:
            for r in pool.imap_unordered(process_subject, payloads):
                print(f"[{r['subj']}] n={r['n']} kept={r['kept']} "
                      f"read_fail={r['rej_read']} pnp_fail={r['rej_pnp']} "
                      f"reproj_big={r['rej_reproj']} "
                      f"reproj={r['mean']:.2f}/{r['p95']:.2f}px "
                      f"{r['sec']:.0f}s -> {r['h5']}", flush=True)
                results.append(r)

    total_kept = sum(r["kept"] for r in results)
    print(f"\ndone. subjects={len(results)} total_kept={total_kept} "
          f"time={time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
