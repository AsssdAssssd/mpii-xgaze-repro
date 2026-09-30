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

import time
from pathlib import Path
import cv2

import numpy as np

from core.detect import Detecter
from core import pnp
from core import preview as viz
from core.writers import H5Writer, write_failed, write_landmarks_csv

OUTPUT_SUBDIRS = (
    "normalized_dataset",  # <subj>.h5
    "failed",              # <subj>_rejected.txt
    "landmarks",           # <subj>.csv
    "rotated_mesh_vis",    # <key>.txt, optional
    "viz",                 # <subj>_viz.jpg, optional
)

class Pipeline:
    def __init__(self,cfg,dataset):
        self.paths, self.flt, self.opt, self.rt = cfg["paths"], cfg["filter"], cfg["options"], cfg["runtime"]
        self.dataset=dataset
        out_root = self.paths.get("output_root")
        if out_root:
            out_root = Path(out_root).resolve()
        else:
            in_root = Path(self.paths["input_root"]).resolve()
            out_root = in_root.parent/(in_root.name.upper() + "_normalized")
    
        ###get config
        self.face_model_full = np.loadtxt(Path(self.paths["face_model"]).resolve()).astype(np.float64)
        self.face_model = self.face_model_full[pnp.FM50_USE]
    
        self.filters_on = self.flt.get("enable", True)
        self.max_reproj = self.flt.get("max_reproj", 0) or 0
        self.skip_undistort = self.flt.get("skip_undistort", False)
        self.min_face = self.flt.get("min_face", 40)
        self.conf = self.flt.get("conf", 0.5)
    
        self.overwrite = self.opt.get("overwrite", False)
        self.save_lm = self.opt.get("save_landmarks", True)
        self.need_mesh = bool(self.opt.get("rotated_mesh_vis", False))
        self.viz_n = self.opt.get("viz_per_subject", 0) or 0
        self.need_viz = self.viz_n > 0
    
        self.gpu_batch = self.rt.get("gpu_batch", 32)
        self.print_freq = self.rt.get("print_freq", 50)
    
        #create output dirs
        self.dirs = {name: out_root/name for name in OUTPUT_SUBDIRS}
        
        self.dirs["failed"].mkdir(parents=True, exist_ok=True)
        self.dirs["normalized_dataset"].mkdir(parents=True, exist_ok=True)
    
        if self.save_lm:
            self.dirs["landmarks"].mkdir(parents=True, exist_ok=True)
        if self.need_mesh:
            self.dirs["rotated_mesh_vis"].mkdir(parents=True, exist_ok=True)
        if self.need_viz:
            self.dirs["viz"].mkdir(parents=True, exist_ok=True)
        print(f"dataset={dataset.name}  out={out_root}", flush=True)

        #detecter
        self.detecter=Detecter(self.paths["landmark_predictor"])
        
    def run(self):
        for subj in self.dataset.subjects():
            #check repeat
            h5_path = self.dirs["normalized_dataset"]/f"{subj}.h5"
            if h5_path.exists() and not self.overwrite:
                print(f"[{subj}] already done ({h5_path}), skip", flush=True)
                continue

            t0 = time.time()

            samples = self.dataset.samples(subj)#generator
            first = next(samples, None)#not none
            if first is None:
                print(f"[{subj}] no samples, skip", flush=True)
                continue

            self.writer = H5Writer(h5_path)
            self.lm_rows, self.rejected, self.viz_rows = [], [], []

            #process one step with batch
            print(f"[{subj}] processing ...", flush=True)
            batch = [first]
            for index,s in enumerate(samples):
                #check
                if s["image"] is None or s["camera"] is None or s["gaze_dir"] is None or s["distortion"] is None or s["key"] is None:
                    self.rejected.append((s["key"], "info_lacked"))
                    continue

                batch.append(s)
                if len(batch) >= self.gpu_batch:
                    self.process_batch(batch,index)
                    batch = []
            self.process_batch(batch,len(batch))

            #save
            if self.save_lm:
                write_landmarks_csv(self.dirs["landmarks"]/f"{subj}.csv", self.lm_rows)
            if self.need_viz and self.viz_rows:
                viz.save_montage(self.dirs["viz"], subj, self.viz_rows)
            write_failed(self.dirs["failed"]/f"{subj}_rejected.txt", self.rejected)
            self.writer.close()
            #progress
            print(f"[{subj}] kept={self.writer.n} rejected={len(self.rejected)} "
                f"{time.time() - t0:.0f}s -> {h5_path}", flush=True)



    def process_batch(self,batch,index):
        if not batch:return

        boxes = self.detecter.detect_faces( [s["image"] for s in batch])

        for s,box in zip(batch, boxes):
            #detecter
            pts, reason = self.detecter.face_landmarks(s["image"], box, self.min_face)
            if pts is None:
                self.rejected.append((s["key"], reason))
                continue
            #landmarks
            self.lm_rows.append((s["key"], pts, reason))

            #pnp
            camera = s["camera"]
            distortion = np.zeros((5, 1)) if self.skip_undistort else s["distortion"]
            source = pts[pnp.LM68_USE].astype(np.float32).reshape(6, 1, 2)
            rvec, tvec, reproj, err = pnp.estimate_head_pose(source, self.face_model, camera, distortion)

            if rvec is None:
                self.rejected.append((s["key"], "pnp_failed"))
                continue
            if self.filters_on and self.max_reproj and err > self.max_reproj:
                self.rejected.append((s["key"], "reproj_too_big"))
                continue
            #viz mesh save
            if self.need_mesh:
                mesh_path = self.dirs["rotated_mesh_vis"]/Path(s["key"]).with_suffix(".txt")
                mesh_path.parent.mkdir(parents=True, exist_ok=True)
                R, _ = cv2.Rodrigues(rvec)
                np.savetxt(mesh_path,(R @ np.asarray(self.face_model_full).T + tvec.reshape(3, 1)).T)

            #normailzed 
            warped, hr_norm, R, lm_warped = pnp.normalize_face( s["image"], self.face_model, pts, rvec, tvec, camera)
            #transform
            gdir = R @ np.asarray(s["gaze_dir"], dtype=np.float64).reshape(3)
            n = gdir / np.linalg.norm(gdir)
            gaze2d = np.array([np.arcsin(-n[1]), np.arctan2(-n[0], -n[2])])
                    
            M = cv2.Rodrigues(hr_norm.reshape(1, 3))[0]
            Zv = M[:, 2]
            head2d = np.array([np.arcsin(Zv[1]), np.arctan2(Zv[0], Zv[2])])
            #write
            self.writer.add(warped, gaze2d, head2d, R, lm_warped)

            #viz
            if self.need_viz and self.writer.n <= self.viz_n:
                o = viz.make_orig_tile(s["image"], pts, source.reshape(6, 2), reproj)
                p = viz.make_face_tile(warped, lm_warped, gaze2d)
                self.viz_rows.append((o, p))

        if self.print_freq and index % self.print_freq == 0:
            print(f"  {index}", flush=True) 
