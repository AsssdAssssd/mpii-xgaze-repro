"""Generic head-pose / normalization math (dataset agnostic).

Everything here works on camera coordinates only. A dataset adapter is
responsible for producing:
  - 68 landmarks (2D, image pixels)
  - camera matrix + distortion
  - gaze direction in camera coordinates, i.e. (target3d - person3d)
"""

import cv2
import numpy as np

# 6 PnP points selected from the 50-point 3D face model
LM68_USE = [36, 39, 42, 45, 31, 35]
FM50_USE = [20, 23, 26, 29, 15, 19]

FOCAL_NORM = 960
DIST_NORM = 600
ROI = (224, 224)


def select_face_points(face_model_full):
    return np.asarray(face_model_full, dtype=np.float64)[FM50_USE]


def estimate_head_pose(landmarks6, face_model6, camera, distortion):
    """Return (rvec, tvec, reproj(6,2), mean_reproj_err) or (None,)*4."""
    ret, rvec, tvec = cv2.solvePnP(
        face_model6, landmarks6, camera, distortion, flags=cv2.SOLVEPNP_EPNP)
    if not ret:
        return None, None, None, None
    ret, rvec, tvec = cv2.solvePnP(
        face_model6, landmarks6, camera, distortion, rvec, tvec, True)
    if not ret:
        return None, None, None, None
    proj, _ = cv2.projectPoints(face_model6, rvec, tvec, camera, distortion)
    reproj = proj.reshape(6, 2)
    err = float(np.mean(np.linalg.norm(
        reproj - landmarks6.reshape(6, 2), axis=1)))
    return rvec, tvec, reproj, err


def rotated_mesh(rvec, tvec, face_model_full):
    R, _ = cv2.Rodrigues(rvec)
    return (R @ np.asarray(face_model_full).T + tvec.reshape(3, 1)).T


def normalize_face(image, face_model6, landmarks68, rvec, tvec, camera):
    """Return (warped_patch, hr_norm, R, lm_warped)."""
    ht = tvec.reshape((3, 1))
    hR = cv2.Rodrigues(rvec)[0]
    Fc = np.dot(hR, np.asarray(face_model6).T) + ht
    two_eye_center = np.mean(Fc[:, 0:4], axis=1).reshape((3, 1))
    nose_center = np.mean(Fc[:, 4:6], axis=1).reshape((3, 1))
    face_center = np.mean(
        np.concatenate((two_eye_center, nose_center), axis=1),
        axis=1).reshape((3, 1))

    distance = np.linalg.norm(face_center)
    z_scale = DIST_NORM / distance
    cam_norm = np.array([
        [FOCAL_NORM, 0, ROI[0] / 2],
        [0, FOCAL_NORM, ROI[1] / 2],
        [0, 0, 1.0],
    ])
    S = np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, z_scale],
    ])

    hRx = hR[:, 0]
    forward = (face_center / distance).reshape(3)
    down = np.cross(forward, hRx)
    down /= np.linalg.norm(down)
    right = np.cross(down, forward)
    right /= np.linalg.norm(right)
    R = np.c_[right, down, forward].T

    W = np.dot(np.dot(cam_norm, S), np.dot(R, np.linalg.inv(camera)))
    img_warped = cv2.warpPerspective(image, W, ROI)

    hR_norm = np.dot(R, hR)
    hr_norm = cv2.Rodrigues(hR_norm)[0].reshape(3)

    det_point = landmarks68.reshape(-1, 1, 2)
    lm_warped = cv2.perspectiveTransform(det_point, W).reshape(-1, 2)

    return img_warped, hr_norm, face_center, R, lm_warped


def gaze_to_2d(gvec):
    n = gvec / np.linalg.norm(gvec)
    return np.array([np.arcsin(-n[1]), np.arctan2(-n[0], -n[2])])


def head_to_2d(hr_norm):
    M = cv2.Rodrigues(hr_norm.reshape(1, 3))[0]
    Zv = M[:, 2]
    return np.array([np.arcsin(Zv[1]), np.arctan2(Zv[0], Zv[2])])
