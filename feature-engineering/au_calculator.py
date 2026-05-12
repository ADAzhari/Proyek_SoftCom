# feature_engineering/au_calculator.py
import numpy as np

# Index landmark MediaPipe yang relevan
# Referensi: https://mediapipe.dev/images/mobile/face_mesh_full_res.jpg
LANDMARKS = {
    # Mata kiri
    "left_eye_top":    159,
    "left_eye_bottom": 145,
    "left_eye_left":   33,
    "left_eye_right":  133,

    # Mata kanan
    "right_eye_top":    386,
    "right_eye_bottom": 374,
    "right_eye_left":   362,
    "right_eye_right":  263,

    # Alis kiri
    "left_brow_inner":  107,
    "left_brow_outer":  70,

    # Alis kanan
    "right_brow_inner": 336,
    "right_brow_outer": 300,

    # Hidung (referensi jarak)
    "nose_tip": 4,
}

def eye_aspect_ratio(landmarks, side="left"):
    """
    EAR = (vertical_1 + vertical_2) / (2 * horizontal)
    Semakin kecil EAR → mata semakin menutup → AU43/AU45
    """
    if side == "left":
        top    = landmarks[LANDMARKS["left_eye_top"]][:2]
        bottom = landmarks[LANDMARKS["left_eye_bottom"]][:2]
        left   = landmarks[LANDMARKS["left_eye_left"]][:2]
        right  = landmarks[LANDMARKS["left_eye_right"]][:2]
    else:
        top    = landmarks[LANDMARKS["right_eye_top"]][:2]
        bottom = landmarks[LANDMARKS["right_eye_bottom"]][:2]
        left   = landmarks[LANDMARKS["right_eye_left"]][:2]
        right  = landmarks[LANDMARKS["right_eye_right"]][:2]

    vertical   = np.linalg.norm(top - bottom)
    horizontal = np.linalg.norm(left - right)

    ear = vertical / (horizontal + 1e-6)  # hindari division by zero
    return ear

def brow_raise(landmarks, face_height):
    """
    Jarak alis ke mata, dinormalisasi dengan tinggi wajah
    Nilai tinggi → alis terangkat → AU1/AU2 (tanda kaget/stres)
    Nilai rendah → alis turun → AU4 (tanda mengantuk/lelah)
    """
    left_brow  = landmarks[LANDMARKS["left_brow_inner"]][:2]
    left_eye   = landmarks[LANDMARKS["left_eye_top"]][:2]
    right_brow = landmarks[LANDMARKS["right_brow_inner"]][:2]
    right_eye  = landmarks[LANDMARKS["right_eye_top"]][:2]

    left_dist  = np.linalg.norm(left_brow - left_eye)
    right_dist = np.linalg.norm(right_brow - right_eye)

    avg_dist = (left_dist + right_dist) / 2
    return avg_dist / (face_height + 1e-6)  # normalisasi

def compute_au_vector(landmarks):
    """
    Output: dict berisi semua nilai AU yang akan jadi input ANFIS
    """
    # hitung face height sebagai normalization reference
    top_face    = landmarks[10][:2]
    bottom_face = landmarks[152][:2]
    face_height = np.linalg.norm(top_face - bottom_face)

    left_ear  = eye_aspect_ratio(landmarks, "left")
    right_ear = eye_aspect_ratio(landmarks, "right")
    avg_ear   = (left_ear + right_ear) / 2

    brow      = brow_raise(landmarks, face_height)

    return {
        "EAR":       avg_ear,   # proxy AU43/AU45 — utama
        "EAR_left":  left_ear,  # asimetri mata (tanda fatigue)
        "EAR_right": right_ear,
        "BROW":      brow,      # proxy AU1/AU4
    }