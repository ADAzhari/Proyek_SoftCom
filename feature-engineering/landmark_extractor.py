# feature_engineering/landmark_extractor.py
import cv2
import mediapipe as mp
import numpy as np

class LandmarkExtractor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,  # lebih presisi di area mata
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract(self, frame):
        """
        Input  : frame BGR dari OpenCV
        Output : array (478, 3) koordinat landmark, atau None
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        h, w, _ = frame.shape
        landmarks = results.multi_face_landmarks[0].landmark

        # konversi ke pixel coordinates
        points = np.array([
            (lm.x * w, lm.y * h, lm.z * w)
            for lm in landmarks
        ], dtype=np.float32)

        return points