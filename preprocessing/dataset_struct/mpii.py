"""MPIIFaceGaze adapter.

Layout:
  <input_root>/<subj>/day*/*.jpg
  <input_root>/<subj>/<subj>.txt                annotation
  <input_root>/<subj>/Calibration/Camera.mat    camera

Annotation columns (whitespace split): [21:24] = face center 3D,
[24:27] = gaze target 3D, both in camera coordinates.
"""

import glob
import os

import numpy as np

from core import loader


def load_camera(subj_dir):
    from scipy.io import loadmat
    m = loadmat(os.path.join(subj_dir, "Calibration", "Camera.mat"))
    camera = np.array(m["cameraMatrix"], dtype=np.float64).reshape(3, 3)
    distortion = np.array(m["distCoeffs"], dtype=np.float64).reshape(-1, 1)
    return camera, distortion


class MPIIFaceGaze:
    name = "mpii"

    def __init__(self, cfg):
        self.root = loader.resolve_path(cfg["paths"]["input_root"])
        self.subjects_list = loader.subjects_from(cfg)

    def subjects(self):
        return self.subjects_list

    def samples(self, subj):
        subj_dir = os.path.join(self.root, subj)
        anno_path = os.path.join(subj_dir, f"{subj}.txt")
        if not os.path.isdir(subj_dir) or not os.path.isfile(anno_path):
            return []

        camera, distortion = load_camera(subj_dir)

        anno = {}
        with open(anno_path) as f:
            for line in f:
                t = line.split()
                anno[t[0]] = (np.array(t[21:24], dtype=np.float64),
                              np.array(t[24:27], dtype=np.float64))

        samples = []
        for fpath in sorted(glob.glob(os.path.join(subj_dir, "day*", "*.jpg"))):
            short = os.path.relpath(fpath, subj_dir).replace("\\", "/")
            if short not in anno:
                continue
            fc, gt = anno[short]
            samples.append({
                "key": os.path.relpath(fpath, self.root).replace("\\", "/"),
                "image": fpath,
                "camera": camera,
                "distortion": distortion,
                "gaze_dir": gt - fc,
            })
        return samples
