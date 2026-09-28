"""MPIIFaceGaze adapter.

Layout:
  <input_root>/<subj>/day*/*.jpg
  <input_root>/<subj>/<subj>.txt                annotation
  <input_root>/<subj>/Calibration/Camera.mat    camera

Annotation columns (whitespace split): [21:24] = face center 3D,
[24:27] = gaze target 3D, both in camera coordinates.

samples() reads each jpg into a BGR array so the shared pipeline never has to
know where the pixels came from (files here, decoded frames/arrays elsewhere).
"""

from pathlib import Path

import cv2
import numpy as np


def load_camera(subj_dir):
    from scipy.io import loadmat
    m = loadmat(Path(subj_dir)/"Calibration"/"Camera.mat")
    camera = np.array(m["cameraMatrix"], dtype=np.float64).reshape(3, 3)
    distortion = np.array(m["distCoeffs"], dtype=np.float64).reshape(-1, 1)
    return camera, distortion


class MPIIFaceGaze:
    name = "mpii"

    def __init__(self, cfg):
        self.root = Path(cfg["paths"]["input_root"]).expanduser()
        subjects = [s.strip() for s in str(cfg["runtime"].get("subjects") or "").split(",")
                    if s.strip()]
        self.subjects_list = subjects or [f"p{i:02d}" for i in range(15)]

    def subjects(self):
        return self.subjects_list

    def samples(self, subj):
        subj_dir = self.root/subj
        anno_path = subj_dir/f"{subj}.txt"
        if not subj_dir.is_dir() or not anno_path.is_file():
            return

        camera, distortion = load_camera(subj_dir)

        anno = {}
        with open(anno_path) as f:
            for line in f:
                t = line.split()
                anno[t[0]] = (np.array(t[21:24], dtype=np.float64),
                              np.array(t[24:27], dtype=np.float64))

        for fpath in sorted(subj_dir.glob("day*/*.jpg")):
            short = fpath.relative_to(subj_dir).as_posix()
            if short not in anno:
                continue
            fc, gt = anno[short]
            yield {
                "key": fpath.relative_to(self.root).as_posix(),
                "image": cv2.imread(str(fpath), cv2.IMREAD_COLOR),
                "camera": camera,
                "distortion": distortion,
                "gaze_dir": gt - fc,
            }
