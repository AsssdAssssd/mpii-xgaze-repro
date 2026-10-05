from pathlib import Path

import cv2
import h5py
import numpy as np

CAMERAS = ("basler", "webcam_l", "webcam_c", "webcam_r")

class EVE:
    name = "eve"

    def __init__(self, cfg):
        self.root = Path(cfg["paths"]["input_root"]).resolve()#有test/train的那个目录
        self.hz=cfg["options"]["target_hz"]
        subjects = [s.strip() for s in str(cfg["runtime"].get("subjects") or "").split(",")
                    if s.strip()]
        full_subjects = []
        for title in ("train", "val"):#test没有标注，暂时都不用
            full_subjects += sorted(p.name for p in self.root.glob(f"{title}*") if p.is_dir())
        if not full_subjects:
            raise FileNotFoundError(f"No target subjects found under {self.root}")
        if subjects:
            for s in subjects:
                if s not in full_subjects:
                    raise ValueError(f"Subject {s} not found under {self.root}")
        else:
            subjects = full_subjects
        self.subjects_list = subjects

    def subjects(self):
        return self.subjects_list


    def samples(self, subj):
        subj_dir = self.root/subj
        if not subj_dir.is_dir():
            return

        for step_dir in sorted(subj_dir.glob("step*")):
            for cam in CAMERAS:
                h5_path = step_dir/f"{cam}.h5"
                video_path = step_dir/f"{cam}.mp4"
                if not h5_path.is_file() or not video_path.is_file():
                    continue
                for sample in self._single_camera_samples(subj, step_dir, cam, h5_path, video_path,cam=="basler"):
                    yield sample

    def _single_camera_samples(self, subj, step_dir, cam, h5_path, video_path,is_basler=False):
        with h5py.File(h5_path, "r") as h5:
            camera = np.array(h5["camera_matrix"][:], dtype=np.float64)
            screen_to_cam = np.array(h5["camera_transformation"][:], dtype=np.float64)
            mm_per_px = np.array(h5["millimeters_per_pixel"][:], dtype=np.float64)
            n = int(h5["face_PoG_tobii/data"].shape[0])

            pog = self._read(h5, "face_PoG_tobii")
            face_o = self._read(h5, "face_o")

        cap = cv2.VideoCapture(str(video_path))
        step=int(60/self.hz) if is_basler else int(30/self.hz)
        try:
            for i in range(n):
                if not cap.grab():
                    break
                if i%step or i==0:
                    continue
                
                ok, frame = cap.retrieve()
                if not ok:
                    continue

                gaze_dir = self._gaze_dir(i, pog, face_o,
                                          screen_to_cam, mm_per_px)#这里单位mm
                if gaze_dir is None:
                    continue

                yield {
                    "key": f"{subj}/{step_dir.name}/{cam}/{i:07d}",
                    "image": frame,
                    "camera": camera,
                    "distortion": np.zeros((5, 1), dtype=np.float64),
                    "gaze_dir": gaze_dir,
                    "kwargs":{
                        "subject":subj,
                        "step_dir":step_dir.name,
                        "camera":cam,
                        "index":f"frame_{i:07d}"
                    }
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
    def _gaze_dir(i, pog, face_o, screen_to_cam, mm_per_px):
        if pog is None or not pog[1][i]:
            return None
        if face_o is not None:
            origin = face_o[0][i]
        else:
            return None
        mm = np.array([pog[0][i, 0]*mm_per_px[0],pog[0][i, 1]*mm_per_px[1],0.0, 1.0])
        pog_cam = (screen_to_cam @ mm)[:3]
        return pog_cam - origin#后续验证单位一致
