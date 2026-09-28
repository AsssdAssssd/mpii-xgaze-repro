"""EVE dataset adapter.

Layout (input_root points either at EVE_dataset or EVE_dataset/eve_dataset):
  <input_root>/<subj>/<step>/basler.h5       per-frame annotations (N,)
  <input_root>/<subj>/<step>/basler.mp4      face camera frames (1920x1080, 60fps)
  <input_root>/<subj>/<step>/<cam>.h5/.mp4   cam in basler, webcam_l/c/r

Subjects are train01..train39, val01..val05, test01..test10.

Per-frame fields used here (all from basler.h5, camera coordinates in mm):
  camera_matrix          (3,3)   intrinsics of the basler camera
  camera_transformation  (4,4)   screen-plane (mm, z=0) -> camera
  millimeters_per_pixel  (2,)    screen px -> mm
  face_PoG_tobii/data    (N,2)   gaze point on screen, screen pixels
  left_o/right_o/data    (N,3)   eye origins in camera coords

EVE ships no distortion model, so a zero distortion vector is reported.

Landmarks are intentionally NOT taken from basler.h5: the adapter only hands
the raw frame to the shared pipeline so detection/PnP stays identical across
datasets (and can be swapped later). EVE's own 68-point annotations stay
unused here.

samples() decodes the mp4 frames on the fly so the shared pipeline keeps its
"image is an array" contract. gaze_dir is produced in camera coordinates as
(point_of_gaze_3d - eye_origin), i.e. the same convention as MPII's
(target3d - face_center3d), pointing back towards the screen/camera (-z).
"""

from pathlib import Path

import cv2
import h5py
import numpy as np

DEFAULT_CAMERA = "basler"


class EVE:
    name = "eve"

    def __init__(self, cfg):
        root = Path(cfg["paths"]["input_root"]).expanduser()
        if not self._looks_like_root(root):
            nested = root / "eve_dataset"
            if self._looks_like_root(nested):
                root = nested
        self.root = root

        subjects = [s.strip() for s in str(cfg["runtime"].get("subjects") or "").split(",")
                    if s.strip()]
        self.subjects_list = subjects or self._discover_subjects()

        self.camera = str(cfg["runtime"].get("camera", DEFAULT_CAMERA))
        self.frame_stride = max(1, int(cfg["runtime"].get("frame_stride", 1) or 1))
        self.max_frames_per_step = int(cfg["runtime"].get("max_frames_per_step", 0) or 0)

    @staticmethod
    def _looks_like_root(path):
        return path.is_dir() and any(
            (path/f"{split}01").is_dir() for split in ("train", "val", "test"))

    def _discover_subjects(self):
        subjects = []
        for split in ("train", "val", "test"):
            subjects += sorted(p.name for p in self.root.glob(f"{split}*") if p.is_dir())
        return subjects

    def subjects(self):
        return self.subjects_list

    def samples(self, subj):
        subj_dir = self.root/subj
        if not subj_dir.is_dir():
            return

        for step_dir in sorted(subj_dir.glob("step*")):
            h5_path = step_dir/f"{self.camera}.h5"
            video_path = step_dir/f"{self.camera}.mp4"
            if not h5_path.is_file() or not video_path.is_file():
                continue

            with h5py.File(h5_path, "r") as h5:
                camera = np.array(h5["camera_matrix"][:], dtype=np.float64)
                screen_to_cam = np.array(h5["camera_transformation"][:], dtype=np.float64)
                mm_per_px = np.array(h5["millimeters_per_pixel"][:], dtype=np.float64)
                n = int(h5["face_PoG_tobii/data"].shape[0])

                if self.max_frames_per_step:
                    n = min(n, self.max_frames_per_step)

                pog = self._read(h5, "face_PoG_tobii")
                left_o = self._read(h5, "left_o")
                right_o = self._read(h5, "right_o")
                face_o = self._read(h5, "face_o")

            cap = cv2.VideoCapture(str(video_path))
            try:
                for i in range(n):
                    ok, frame = cap.read()
                    if not ok:
                        break
                    if i % self.frame_stride:
                        continue

                    gaze_dir = self._gaze_dir(i, pog, left_o, right_o, face_o,
                                              screen_to_cam, mm_per_px)
                    if gaze_dir is None:
                        continue

                    yield {
                        "key": f"{subj}/{step_dir.name}/{i:07d}",
                        "image": frame,
                        "camera": camera,
                        "distortion": np.zeros((5, 1), dtype=np.float64),
                        "gaze_dir": gaze_dir,
                    }
            finally:
                cap.release()

    @staticmethod
    def _read(h5, name):
        if name not in h5:
            return None
        data = np.array(h5[f"{name}/data"][:], dtype=np.float64)
        validity = (np.array(h5[f"{name}/validity"][:], dtype=bool)
                    if f"{name}/validity" in h5 else np.ones(len(data), dtype=bool))
        return data, validity

    @staticmethod
    def _gaze_dir(i, pog, left_o, right_o, face_o, screen_to_cam, mm_per_px):
        if pog is None or not pog[1][i]:
            return None

        if left_o is not None and right_o is not None:
            origin = 0.5*(left_o[0][i] + right_o[0][i])
        elif face_o is not None:
            origin = face_o[0][i]
        else:
            return None

        mm = np.array([pog[0][i, 0]*mm_per_px[0],
                       pog[0][i, 1]*mm_per_px[1],
                       0.0, 1.0])
        pog_cam = (screen_to_cam @ mm)[:3]
        return pog_cam - origin
