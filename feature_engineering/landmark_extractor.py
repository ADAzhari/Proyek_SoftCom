# feature_engineering/landmark_extractor.py
import cv2
import numpy as np

# import cara baru yang works di 0.10.x
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh

class LandmarkExtractor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        h, w, _ = frame.shape
        landmarks = results.multi_face_landmarks[0].landmark

        points = np.array([
            (lm.x * w, lm.y * h, lm.z * w)
            for lm in landmarks
        ], dtype=np.float32)

        return points