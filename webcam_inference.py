import sys
from pathlib import Path
import cv2
import numpy as np
import pickle
from collections import deque
import time

# Pastikan sys.path benar
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from feature_engineering.landmark_extractor import LandmarkExtractor
from feature_engineering.au_calculator import compute_au_vector, PERCLOSCalculator
from feature_engineering.head_pose import estimate_head_pose

def main():
    # ─── 1. Load Model & Scaler ────────────────────────────────────────────────
    model_path = ROOT / "models" / "anfis_model.pkl"
    scaler_path = ROOT / "models" / "scaler.pkl"
    
    if not model_path.exists() or not scaler_path.exists():
        print("[ERROR] Model atau Scaler tidak ditemukan. Jalankan train_local.py dulu!")
        return

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)

    print("✅ Model dan Scaler berhasil dimuat.")

    # ─── 2. Inisialisasi Modul ──────────────────────────────────────────────────
    extractor = LandmarkExtractor()
    perclos_calc = PERCLOSCalculator(window_seconds=10, fps=30, ear_threshold=0.22)
    
    # Buffer untuk menghaluskan output ANFIS (Temporal Smoothing)
    # Rata-rata dari 30 frame terakhir (sekitar 1 detik)
    anfis_history = deque(maxlen=30)

    # Buka Webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Kamera tidak dapat diakses.")
        return

    print("\n✅ Webcam aktif. Tekan 'q' untuk keluar.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame secara horizontal agar seperti cermin
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        # ─── 3. Ekstraksi Fitur ─────────────────────────────────────────────────
        landmarks = extractor.extract(frame)
        
        status_text = "Status: TIDAK TERDETEKSI"
        color = (200, 200, 200)

        if landmarks is not None:
            # Hitung Action Units
            au = compute_au_vector(landmarks)
            
            # Estimasi Head Pose
            pose = estimate_head_pose(landmarks, frame.shape)
            pitch = pose[0] if pose is not None else 0.0

            # Susun vektor input persis seperti saat training
            # ["EAR", "EAR_asym", "MAR", "BROW", "PITCH"]
            ear = au["EAR"]
            ear_asym = abs(au["EAR_left"] - au["EAR_right"])
            mar = au["MAR"]
            brow = au["BROW"]
            
            features = np.array([[ear, ear_asym, mar, brow, pitch]])

            # Scale fitur
            features_scaled = scaler.transform(features)[0]

            # ─── 4. Prediksi ANFIS & PERCLOS ─────────────────────────────────────
            # Update PERCLOS
            perclos_calc.update(ear)
            perclos_val = perclos_calc.compute()

            # Prediksi ANFIS per-frame
            raw_score = model.forward(features_scaled)
            anfis_history.append(raw_score)
            
            # Skor kantuk dihaluskan (moving average)
            smooth_score = sum(anfis_history) / len(anfis_history)

            # ─── 5. Logika Peringatan (Alert Logic) ──────────────────────────────
            # Jika skor ANFIS > 0.5 ATAU PERCLOS > 0.3 (30% mata tertutup dlm 10 dtk)
            if smooth_score > 0.55 or perclos_val > 0.30:
                status_text = "Status: MENGANTUK!"
                color = (0, 0, 255) # Merah
                # Tampilkan peringatan besar di tengah layar
                cv2.putText(frame, "!!! AWAS MENGANTUK !!!", (w//2 - 200, h//2), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            else:
                status_text = "Status: NORMAL"
                color = (0, 255, 0) # Hijau

            # ─── 6. Visualisasi Data di Layar ───────────────────────────────────
            # Background teks
            cv2.rectangle(frame, (10, 10), (320, 180), (0, 0, 0), -1)
            
            y_pos = 40
            cv2.putText(frame, status_text, (20, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"ANFIS Score : {smooth_score:.2f}", (20, y_pos+30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"PERCLOS     : {perclos_val:.2f}", (20, y_pos+55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.putText(frame, f"EAR         : {ear:.3f}", (20, y_pos+80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, f"MAR (Yawn)  : {mar:.3f}", (20, y_pos+105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)
            cv2.putText(frame, f"PITCH       : {pitch:.1f} deg", (20, y_pos+130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200,200,200), 1)

            # Gambar landmark di wajah (opsional, bisa dinonaktifkan jika berat)
            for point in landmarks:
                x, y = int(point[0]), int(point[1])
                cv2.circle(frame, (x, y), 1, (255, 255, 0), -1)

        else:
            # Kosongkan history jika wajah hilang
            if len(anfis_history) > 0:
                anfis_history.popleft()
            cv2.putText(frame, status_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Tampilkan
        cv2.imshow('FaceGuard AI - Live Inference', frame)

        # Tekan 'q' atau klik tombol 'X' (close) pada window untuk keluar
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or cv2.getWindowProperty('FaceGuard AI - Live Inference', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
