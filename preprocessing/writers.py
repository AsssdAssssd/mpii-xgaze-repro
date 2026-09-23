"""Generic output writers (dataset agnostic).

All datasets end up in the same h5 schema:
  face_patch (N,224,224,3) uint8
  face_gaze (N,2) float32
  face_head_pose (N,2) float32
  face_mat_norm (N,3,3) float32
  facial_landmarks (N,68,2) float32
"""

import csv
import os

import h5py
import numpy as np

from pnp import ROI

LANDMARK_HEADER = ["path"] + [f"x{i:02d}" for i in range(68)] + \
    [f"y{i:02d}" for i in range(68)]


def write_landmarks_csv(path, records):
    """records: list of (rel_path, pts68 | None, reason). Returns (ok, bad)."""
    ok = bad = 0
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(LANDMARK_HEADER)
        for rel, pts, reason in records:
            if pts is None:
                bad += 1
                continue
            writer.writerow([rel] + [f"{v:.2f}" for v in pts[:, 0]] +
                            [f"{v:.2f}" for v in pts[:, 1]])
            ok += 1
    return ok, bad


def write_failed(path, records):
    """records: list of (rel_path, reason)."""
    lines = [f"{rel}\t{reason}" for rel, reason in records]
    if not lines:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


class H5Writer:
    def __init__(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = h5py.File(path, "w", libver="latest")
        d = self.f
        self.patch = d.create_dataset(
            "face_patch", shape=(0, ROI[0], ROI[1], 3),
            maxshape=(None, ROI[0], ROI[1], 3),
            chunks=(1, ROI[0], ROI[1], 3), compression="lzf", dtype=np.uint8)
        self.gaze = d.create_dataset(
            "face_gaze", shape=(0, 2), maxshape=(None, 2), dtype=np.float32)
        self.head = d.create_dataset(
            "face_head_pose", shape=(0, 2), maxshape=(None, 2), dtype=np.float32)
        self.mat = d.create_dataset(
            "face_mat_norm", shape=(0, 3, 3), maxshape=(None, 3, 3), dtype=np.float32)
        self.lm = d.create_dataset(
            "facial_landmarks", shape=(0, 68, 2), maxshape=(None, 68, 2),
            dtype=np.float32)
        self.n = 0

    def add(self, patch, gaze2d, head2d, mat_norm, lm):
        for ds in (self.patch, self.gaze, self.head, self.mat, self.lm):
            ds.resize(self.n + 1, axis=0)
        self.patch[self.n] = patch
        self.gaze[self.n] = np.asarray(gaze2d, dtype=np.float32)
        self.head[self.n] = np.asarray(head2d, dtype=np.float32)
        self.mat[self.n] = np.asarray(mat_norm, dtype=np.float32)
        self.lm[self.n] = np.asarray(lm, dtype=np.float32)
        self.n += 1

    def close(self):
        self.f.flush()
        self.f.swmr_mode = True
        self.f.close()
