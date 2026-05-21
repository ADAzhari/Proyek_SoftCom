# Proyek_SoftCom: FaceGuard AI
## Deteksi Kantuk Pengemudi dengan ANFIS

**Oleh:** Eustachius Rivan V.N (230016), Achmad Dzaki A. (230034), Michael Jordan A.S (230060)

---

## Deskripsi Proyek

FaceGuard AI adalah sistem deteksi kantuk pengemudi menggunakan **ANFIS (Adaptive Neuro-Fuzzy Inference System)**. Sistem ini mengekstrak fitur wajah dari kamera menggunakan **MediaPipe Face Mesh**, kemudian memproses fitur tersebut melalui ANFIS untuk mengklasifikasikan kondisi pengemudi: **Normal** atau **Mengantuk (Drowsy)**.

### Pipeline

```
Gambar Wajah → MediaPipe (468 landmark) → Ekstraksi Fitur → MinMaxScaler → ANFIS → Klasifikasi (Normal/Drowsy)
```

---

## Struktur Proyek

```
Proyek_SoftCom/
├── anfis/
│   ├── anfis_model.py       # Arsitektur ANFIS 5-layer (Sugeno)
│   ├── membership.py        # Gaussian Membership Function + gradient
│   └── train.py             # Training dengan pure Gradient Descent
├── feature_engineering/
│   ├── landmark_extractor.py # Wrapper MediaPipe Face Mesh
│   ├── au_calculator.py      # Hitung EAR, MAR, BROW dari landmark
│   └── head_pose.py          # Estimasi pose kepala (pitch/yaw/roll)
├── data/
│   ├── raw/                  # Dataset DDD (dari NTHU-DDD)
│   └── processed/            # Fitur hasil ekstraksi + visualisasi
├── models/                   # Model ANFIS + scaler tersimpan
├── notebooks/
│   └── training.ipynb        # Notebook Colab (versi awal)
├── train_local.py            # Script training utama (lokal)
├── requirements.txt
└── README.md
```

---

## Fitur yang Diekstrak

| Fitur | Deskripsi | Indikator |
|-------|-----------|-----------|
| **EAR** | Eye Aspect Ratio — rasio vertikal/horizontal mata | Mata menutup → EAR rendah |
| **EAR_asym** | Selisih EAR kiri vs kanan | Asimetri mata → tanda kelelahan |
| **MAR** | Mouth Aspect Ratio — rasio bukaan mulut | Menguap → MAR tinggi |
| **BROW** | Jarak alis ke mata (dinormalisasi) | Alis turun → mengantuk |
| **PITCH** | Sudut anggukan kepala | Menunduk → tanda kantuk |

---

## Arsitektur ANFIS

- **Input**: 5 fitur (sudah di-MinMaxScale ke [0,1])
- **Membership Function**: 2 Gaussian MF per variabel ("Rendah", "Tinggi")
- **Rules**: 2⁵ = 32 aturan fuzzy (product T-norm)
- **Consequent**: Linear Sugeno (p·x + q per rule)
- **Output**: Skor 0–1, threshold 0.5 untuk klasifikasi
- **Training**: Pure gradient descent untuk premise (MF) dan consequent

---

## Cara Menggunakan

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Dataset
Dataset: [Driver Drowsiness Dataset (DDD)](https://www.kaggle.com/datasets/ismailnasri20/driver-drowsiness-dataset-ddd)
```bash
kaggle datasets download -d ismailnasri20/driver-drowsiness-dataset-ddd --unzip -p data/raw
```
Atau download manual dan unzip ke `data/raw/`.

### 3. Jalankan Training
```bash
python train_local.py
```
Tutup gambar-gambar yang muncul jika dirasa tidak ada problem.

## Hasil Evaluasi Terbaru (Setelah Strict Frame Selection)

Setelah menerapkan **Strict Frame Selection** (menyaring dataset agar hanya berisi frame di mana mata benar-benar terbuka untuk kelas Normal, dan mata tertutup/menguap untuk kelas Drowsy), performa model meningkat drastis.

### Metrik Evaluasi

| Metrik | Nilai |
|--------|-------|
| **Accuracy** | **96.3%** |
| **Precision (Drowsy)** | ~99% |
| **Recall (Drowsy)** | ~91% |

### Confusion Matrix

|  | Prediksi Normal | Prediksi Drowsy |
|--|-----------------|-----------------|
| **Aktual Normal** | 492 | 2 |
| **Aktual Drowsy** | 27 | 279 |

### Observasi

- **Pemisahan Kelas yang Jelas:** Dengan membuang frame "Drowsy" yang matanya terbuka lebar, ANFIS kini dapat dengan mudah membedakan sinyal fisik kantuk. Hanya 2 frame Normal yang salah diprediksi.
- **ANFIS Sangat Akurat sebagai Detektor Sinyal:** Model ini kini berfungsi sempurna sebagai pendeteksi event (mata tertutup / menguap).

---

## Solusi Sistem Terintegrasi

Mengingat batasan bahwa kantuk adalah fenomena temporal, kami telah mengimplementasikan solusi yang menggabungkan model akurasi tinggi ini dengan agregasi waktu nyata (real-time).

### Live Webcam Inference (`webcam_inference.py`)
Script ini dirancang untuk mendeteksi kantuk secara langsung melalui kamera web, dengan fitur:
1. **Temporal Smoothing (Rolling Average):** Skor ANFIS dihaluskan menggunakan rata-rata bergerak dari 30 frame terakhir (~1 detik). Ini mencegah peringatan palsu akibat kedipan mata normal.
2. **PERCLOS (Percentage of Eye Closure):** Dihitung secara real-time berdasarkan 10 detik terakhir. Jika mata tertutup lebih dari 30% dari waktu tersebut, sistem akan langsung memicu peringatan.
3. **Peringatan Ganda:** Status berubah menjadi "MENGANTUK!" jika *Smooth ANFIS Score* > 0.55 ATAU *PERCLOS* > 0.30.

---

## Rencana Perbaikan ke Depan

1. **Eksperimen dengan Dataset Tambahan:** Diskusi dengan dosen mengenai DISFA/DISFA+ untuk kalibrasi intensitas Action Unit.
2. **Perbaiki landmark Mulut (MAR):** Saat ini menggunakan bibir bagian dalam (inner lip). Bisa diganti ke bibir luar (outer lip) agar rasio pelebaran saat menguap lebih signifikan.
3. **Optimasi Hyperparameter:** Menyesuaikan *learning rate* dan jumlah fungsi keanggotaan (MF) untuk meminimalkan *false negative* (27 sampel) yang tersisa.

---

## Catatan Teknis

- Dataset DDD adalah kumpulan frame gambar yang diekstrak dari **NTHU-DDD (National Tsing Hua University Drowsy Driving Dataset)**
- MediaPipe Face Mesh menghasilkan **468 landmark 3D** per wajah
- ANFIS menggunakan **Sugeno-type** inference dengan output linear
- Training menggunakan **pure gradient descent** (premise + consequent) untuk 32 rules fuzzy.
- Semua fitur di-**MinMaxScale** ke rentang [0,1] sebelum masuk ke ANFIS.
