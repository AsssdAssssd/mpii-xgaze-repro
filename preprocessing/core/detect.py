import cv2
import dlib
import numpy as np
from imutils import face_utils
import torch
from facenet_pytorch import MTCNN


class Detecter:
    def __init__(self,predictor_path, device="cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available")
        self.mtcnn = MTCNN(device="cuda", select_largest=True, post_process=False)
        self.predictor = dlib.shape_predictor(str(predictor_path))

    def detect_faces(self, images):
        """images: list of BGR arrays. Returns a list of (4,) boxes or None."""
        out = [None] * len(images)
        groups = {}
        for i, im in enumerate(images):
            groups.setdefault(im.shape[:2], []).append(i) #按照shape分类index
        for idxs in groups.values():
            rgb = [cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB) for i in idxs] #重组
            boxes, _ = self.mtcnn.detect(rgb)
            for i, b in zip(idxs, boxes):
                if b is not None and len(b) > 0:
                    out[i] = np.asarray(b)[0, :4].astype(np.int32)
        return out


    def face_landmarks(self, image, box, min_face):
        """Return (pts68 float32, "ok") or (None, reason)."""
        if box is None:
            return None, "no_face"
        x1, y1, x2, y2 = box
        if min(x2 - x1, y2 - y1) < min_face:
            return None, "face_too_small"
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        rect = dlib.rectangle(int(x1), int(y1), int(x2), int(y2))
        pts = face_utils.shape_to_np(self.predictor(rgb, rect))
        if pts.shape[0] != 68:
            return None, "bad_shape"
        return pts.astype(np.float32), "ok"
