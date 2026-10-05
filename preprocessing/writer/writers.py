"""Generic output writers 
MPII datasets 
  subject: p00/p01
  data:day01
  index:0001
  face_patch (N,224,224,3) uint8
  face_gaze (N,2) float32
  face_head_pose (N,2) float32
  face_mat_norm (N,3,3) float32
  facial_landmarks (N,68,2) float32
"""

from pathlib import Path

import h5py
import numpy as np

from core.pnp import ROI


def write_failed(path, records):
    """records: list of (rel_path, reason)."""
    lines = [f"{rel}\t{reason}" for rel, reason in records]
    if not lines:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


class H5Writer:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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
        self.written_num = 0

    def add(self, patch, gaze2d, head2d, mat_norm, lm, kwargs):
        for ds in (self.patch, self.gaze, self.head, self.mat, self.lm):
            ds.resize(self.written_num + 1, axis=0)
        self.patch[self.written_num] = patch
        self.gaze[self.written_num] = np.asarray(gaze2d, dtype=np.float32)
        self.head[self.written_num] = np.asarray(head2d, dtype=np.float32)
        self.mat[self.written_num] = np.asarray(mat_norm, dtype=np.float32)
        self.lm[self.written_num] = np.asarray(lm, dtype=np.float32)
        self.written_num += 1

    def close(self):
        self.f.flush()
        self.f.swmr_mode = True
        self.f.close()

    def __len__(self):
        return self.written_num


class MPIIWriter(H5Writer):
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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
        self.subject = d.create_dataset(
                    "subject", shape=(0,), maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"))
        self.data = d.create_dataset(
                    "data", shape=(0,), maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"))
        self.index = d.create_dataset(
                    "index", shape=(0,), maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"))
        self.written_num = 0

    def add(self, patch, gaze2d, head2d, mat_norm, lm, kwargs):
        for ds in (self.patch, self.gaze, self.head, self.mat, self.lm,self.subject,self.data,self.index):
            ds.resize(self.written_num + 1, axis=0)
        self.patch[self.written_num] = patch
        self.gaze[self.written_num] = np.asarray(gaze2d, dtype=np.float32)
        self.head[self.written_num] = np.asarray(head2d, dtype=np.float32)
        self.mat[self.written_num] = np.asarray(mat_norm, dtype=np.float32)
        self.lm[self.written_num] = np.asarray(lm, dtype=np.float32)
        self.subject[self.written_num]=kwargs["subject"]
        self.data[self.written_num]=kwargs["data"]
        self.index[self.written_num]=kwargs["index"]
        
        self.written_num += 1


class EveWriter(H5Writer):
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
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
        self.subject = d.create_dataset(
                    "subject", shape=(0,), maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"))
        self.step_dir = d.create_dataset(
                    "step_dir", shape=(0,), maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"))
        self.camera = d.create_dataset(
                    "camera", shape=(0,), maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"))
        self.index = d.create_dataset(
                    "index", shape=(0,), maxshape=(None,),
                    dtype=h5py.string_dtype(encoding="utf-8"))
        self.written_num = 0

    def add(self, patch, gaze2d, head2d, mat_norm, lm, kwargs):#和struct对应
        for ds in (self.patch, self.gaze, self.head, self.mat, self.lm,self.subject,self.step_dir,self.camera,self.index):
            ds.resize(self.written_num + 1, axis=0)
        self.patch[self.written_num] = patch
        self.gaze[self.written_num] = np.asarray(gaze2d, dtype=np.float32)
        self.head[self.written_num] = np.asarray(head2d, dtype=np.float32)
        self.mat[self.written_num] = np.asarray(mat_norm, dtype=np.float32)
        self.lm[self.written_num] = np.asarray(lm, dtype=np.float32)
        self.subject[self.written_num]=kwargs["subject"]
        self.step_dir[self.written_num]=kwargs["step_dir"]
        self.camera[self.written_num]=kwargs["camera"]
        self.index[self.written_num]=kwargs["index"]
        
        self.written_num += 1