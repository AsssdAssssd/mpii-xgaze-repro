import random
from pathlib import Path

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .general_data_loader import trans

CAMERA_FPS = {"basler": 60, "webcam_l": 30, "webcam_c": 30, "webcam_r": 30}
CAMERAS = tuple(CAMERA_FPS)


class EveDataset(IterableDataset):
    def __init__(self, datasetpath, split=("train",), target_fps=5, image_size=224,
                 transform=trans, shuffle=True, seed=0, is_load_label=True):
        self.root = Path(datasetpath)
        self.target_fps = target_fps
        self.image_size = image_size
        self.transform = transform
        self.shuffle = shuffle
        self.seed = seed
        self.is_load_label = is_load_label
        self.units = self._scan(split)
        if not self.units:
            raise FileNotFoundError(
                f"no EVE units under {self.root} for split={split}")

    def _scan(self, split):
        units = []
        for title in split:
            for subj in sorted(p for p in self.root.glob(f"{title}*") if p.is_dir()):
                for step in sorted(subj.glob("step*")):
                    for cam in CAMERAS:
                        h5_path = step / f"{cam}.h5"
                        mp4_path = step / f"{cam}_face.mp4"
                        if h5_path.is_file() and mp4_path.is_file():
                            units.append((h5_path, mp4_path))
        return units

    def _stride(self, mp4_path):
        cam = Path(mp4_path).stem.removesuffix("_face")
        return max(1, round(CAMERA_FPS[cam] / self.target_fps))

    @staticmethod
    def _num_frames(h5_path):
        with h5py.File(h5_path, "r") as f:
            for key in ("face_g_tobii/data", "facial_landmarks/data"):
                if key in f:
                    return f[key].shape[0]
        return 0

    def __len__(self):
        total = 0
        for h5_path, mp4_path in self.units:
            total += self._num_frames(h5_path) // self._stride(mp4_path)
        return total

    def __iter__(self):
        worker = get_worker_info()
        worker_id, num_workers = (0, 1) if worker is None else (worker.id, worker.num_workers)

        units = list(self.units)
        if self.shuffle:
            random.Random(self.seed).shuffle(units)
        units = units[worker_id::num_workers]

        for local_idx, (h5_path, mp4_path) in enumerate(units):
            rng = random.Random((self.seed) * 1_000_003 + worker_id * 10_007 + local_idx)
            yield from self._iter_unit(h5_path, mp4_path, rng)

    @staticmethod
    def _read_h5(h5_path):
        with h5py.File(h5_path, "r") as f:
            gaze = np.asarray(f["face_g_tobii/data"][:], dtype=np.float32)
            valid = np.asarray(f["face_g_tobii/validity"][:], dtype=bool)
        return gaze, valid

    def get_labels_keyed(self):
        labels = {}
        for h5_path, _ in self.units:
            with h5py.File(h5_path, "r") as f:
                labels[str(h5_path.relative_to(self.root))] = np.asarray(
                    f["face_g_tobii/data"][:], dtype=np.float64)
        return labels

    def _to_tensor(self, frame_bgr):
        if frame_bgr.shape[0] != self.image_size or frame_bgr.shape[1] != self.image_size:
            frame_bgr = cv2.resize(
                frame_bgr, (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR)
        frame_rgb = frame_bgr[:, :, ::-1]
        if self.transform is not None:
            return self.transform(frame_rgb)
        img = torch.from_numpy(np.ascontiguousarray(frame_rgb))
        return img.permute(2, 0, 1).float() / 255.0

    def _iter_unit(self, h5_path, mp4_path, rng):
        stride = self._stride(mp4_path)
        gaze = valid = n = None
        if self.is_load_label:
            gaze, valid = self._read_h5(h5_path)
            n = len(gaze)

        cap = cv2.VideoCapture(str(mp4_path))
        if not cap.isOpened():
            return

        samples = []
        try:
            i = 0
            while n is None or i < n:
                if not cap.grab():
                    break
                if i % stride == 0 and (valid is None or valid[i]):
                    ok, frame = cap.retrieve()
                    if ok and frame is not None:
                        samples.append((i, frame))
                i += 1
        finally:
            cap.release()

        if self.shuffle:
            rng.shuffle(samples)#视频内打乱

        for i, frame in samples:
            img = self._to_tensor(frame)
            if self.is_load_label:
                yield img, torch.from_numpy(gaze[i])
            else:
                yield img


def get_eve_train_loader(data_dir, batch_size, num_workers=4, is_shuffle=True):
    dataset = EveDataset(data_dir, split=("train",), shuffle=is_shuffle)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers,
                      pin_memory=True, drop_last=True)


def get_eve_val_loader(data_dir, batch_size, num_workers=4, is_shuffle=True):
    dataset = EveDataset(data_dir, split=("val",), shuffle=is_shuffle)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)


def get_eve_test_loader(data_dir, batch_size, num_workers=4, is_shuffle=True):
    dataset = EveDataset(data_dir, split=("test",), shuffle=is_shuffle,
                         is_load_label=False)
    return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers)

