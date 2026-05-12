# feature_engineering/head_pose.py
import cv2
import numpy as np

# 3D model points referensi wajah (generic face model)
MODEL_POINTS = np.array([
    (0.0,    0.0,    0.0),    # hidung
    (0.0,   -330.0, -65.0),   # dagu
    (-225.0,  170.0, -135.0), # sudut mata kiri
    (225.0,   170.0, -135.0), # sudut mata kanan
    (-150.0, -150.0, -125.0), # sudut mulut kiri
    (150.0,  -150.0, -125.0), # sudut mulut kanan
], dtype=np.float64)

# Index landmark yang sesuai model points di atas
POSE_LANDMARKS = [4, 152, 33, 263, 61, 291]

def estimate_head_pose(landmarks, frame_shape):
    """
    Output: (pitch, yaw, roll) dalam derajat
    - pitch: anggukan (+ = menunduk → sinyal fatigue)
    - yaw  : geleng kepala
    - roll : miring kepala
    """
    h, w = frame_shape[:2]

    image_points = np.array([
        landmarks[idx][:2] for idx in POSE_LANDMARKS
    ], dtype=np.float64)

    focal_length = w
    center = (w / 2, h / 2)
    camera_matrix = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1]
    ], dtype=np.float64)

    dist_coeffs = np.zeros((4, 1))  # asumsi no lens distortion

    success, rotation_vec, _ = cv2.solvePnP(
        MODEL_POINTS, image_points,
        camera_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return None

    rotation_mat, _ = cv2.Rodrigues(rotation_vec)
    angles, *_ = cv2.RQDecomp3x3(rotation_mat)

    pitch, yaw, roll = angles
    return pitch, yaw, roll