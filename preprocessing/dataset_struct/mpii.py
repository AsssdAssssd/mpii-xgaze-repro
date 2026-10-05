"""MPIIFaceGaze adapter.
"""

from pathlib import Path

import cv2
import numpy as np
from scipy.io import loadmat

class MPIIFaceGaze:
    name = "mpii"

    def __init__(self, cfg):
        self.root = Path(cfg["paths"]["input_root"]).resolve()
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
        #camera metrix
        m = loadmat(Path(subj_dir)/"Calibration"/"Camera.mat")
        camera = np.array(m["cameraMatrix"], dtype=np.float64).reshape(3, 3)
        distortion = np.array(m["distCoeffs"], dtype=np.float64).reshape(-1, 1)
        #get gc/gt
        anno = {}
        with open(anno_path) as f:
            for line in f:
                t = line.split()
                anno[t[0]] = (np.array(t[21:24], dtype=np.float64),
                              np.array(t[24:27], dtype=np.float64))#name[fc,gt]

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
                "kwargs":{
                    "subject":subj,
                    "data":Path(short).parent.name,
                    "index":Path(short).name
                }
            }
