"""
train_local.py
=============
Versi lokal dari notebooks/training.ipynb.
Menggantikan semua bagian Colab-specific (Google Drive mount,
Kaggle download, git clone, dll.) dengan path lokal.

Cara pakai:
    python train_local.py

Asumsi struktur direktori:
    Proyek_SoftCom/
    ├── data/
    │   ├── raw/
    │   │   └── Driver Drowsiness Dataset (DDD)/
    │   │       ├── Drowsy/       ← gambar *.png / *.jpg
    │   │       └── Non Drowsy/   ← gambar *.png / *.jpg
    │   └── processed/
    ├── models/
    ├── anfis/
    ├── feature_engineering/
    └── train_local.py   ← file ini
"""

import sys
import os
from pathlib import Path

# ─── 0. Pastikan root proyek ada di sys.path ──────────────────────────────────
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ─── 1. Buat folder yang diperlukan ──────────────────────────────────────────
for d in ["models", "data/raw", "data/processed"]:
    (ROOT / d).mkdir(parents=True, exist_ok=True)

# ─── 2. Import library ───────────────────────────────────────────────────────
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from feature_engineering.landmark_extractor import LandmarkExtractor
from feature_engineering.au_calculator import compute_au_vector, PERCLOSCalculator
from feature_engineering.head_pose import estimate_head_pose

# ─── 3. Tentukan path dataset ────────────────────────────────────────────────
DATA_DIR = ROOT / "data" / "raw" / "Driver Drowsiness Dataset (DDD)"

if not DATA_DIR.exists():
    print(
        f"\n[ERROR] Dataset tidak ditemukan di:\n  {DATA_DIR}\n"
        "Silakan download dataset terlebih dahulu:\n"
        "  kaggle datasets download -d ismailnasri20/driver-drowsiness-dataset-ddd "
        f"--unzip -p {ROOT / 'data' / 'raw'}\n"
    )
    sys.exit(1)

# Tampilkan isi dataset
classes = [d for d in DATA_DIR.iterdir() if d.is_dir()]
print("\n=== Struktur Dataset ===")
for cls in classes:
    n = len(list(cls.glob("*.jpg")) + list(cls.glob("*.png")))
    print(f"  {cls.name}: {n} gambar")

# ─── 4. Visualisasi sampel gambar ─────────────────────────────────────────────
print("\nMenampilkan sampel gambar...")
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
for i, cls in enumerate(classes[:2]):
    images = list(cls.glob("*.png"))[:4] or list(cls.glob("*.jpg"))[:4]
    for j, img_path in enumerate(images):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        axes[i][j].imshow(img)
        axes[i][j].set_title(cls.name)
        axes[i][j].axis("off")
plt.tight_layout()
plt.savefig(str(ROOT / "data" / "processed" / "sample_images.png"))
plt.show()

# ─── 5. Ekstraksi fitur dari gambar ──────────────────────────────────────────
# Label: Drowsy → 1, Non Drowsy → 0
LABEL_MAP = {"Drowsy": 1, "Non Drowsy": 0}

# Batasi jumlah gambar per kelas agar tidak terlalu lama di lokal.
# Set ke None untuk memproses semua gambar.
MAX_PER_CLASS = 1000

extractor = LandmarkExtractor()
perclos   = PERCLOSCalculator(window_seconds=60, fps=30, ear_threshold=0.2)
records   = []

print("\nMengekstrak fitur dari gambar...")
for cls in classes:
    label = LABEL_MAP.get(cls.name)
    if label is None:
        print(f"  [SKIP] Kelas tidak dikenal: {cls.name}")
        continue

    img_paths = list(cls.glob("*.png")) + list(cls.glob("*.jpg"))
    if MAX_PER_CLASS is not None:
        img_paths = img_paths[:MAX_PER_CLASS]

    print(f"\n  [{cls.name}] — memproses {len(img_paths)} gambar...")
    for img_path in tqdm(img_paths, ncols=80):
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        landmarks = extractor.extract(img)
        if landmarks is None:
            continue

        au   = compute_au_vector(landmarks)
        pose = estimate_head_pose(landmarks, img.shape)
        pitch = pose[0] if pose is not None else 0.0

        perclos.update(au["EAR"])
        perclos_val = perclos.compute()

        records.append(
            {
                "EAR":       au["EAR"],
                "EAR_asym":  abs(au["EAR_left"] - au["EAR_right"]),
                "PERCLOS":   perclos_val,
                "BROW":      au["BROW"],
                "PITCH":     pitch,
                "label":     label,
            }
        )

df = pd.DataFrame(records)
print(f"\nTotal sampel berhasil diekstrak: {len(df)}")
print(f"Distribusi label:\n{df['label'].value_counts()}")
print(df.head())

# ─── 6. Buat fatigue score dari label binary ──────────────────────────────────
def label_to_fatigue_score(label):
    """
    Konversi label binary ke fatigue score kontinu.
    Normal (0)  → skor ~35 (rentang 10–55)
    Drowsy (1)  → skor ~80 (rentang 65–100)
    """
    if label == 0:
        return np.clip(np.random.normal(35, 10), 10, 55)
    else:
        return np.clip(np.random.normal(80, 10), 65, 100)

np.random.seed(42)
df["fatigue_score"] = df["label"].apply(label_to_fatigue_score)

# Visualisasi distribusi
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
df[df["label"] == 0]["fatigue_score"].hist(bins=20, color="green", alpha=0.7)
df[df["label"] == 1]["fatigue_score"].hist(bins=20, color="red", alpha=0.7)
plt.legend(["Normal", "Drowsy"])
plt.title("Distribusi Fatigue Score")
plt.xlabel("Fatigue Score")

plt.subplot(1, 2, 2)
df.boxplot(column="EAR", by="label")
plt.title("EAR per Label")
plt.suptitle("")
plt.tight_layout()
plt.savefig(str(ROOT / "data" / "processed" / "fatigue_distribution.png"))
plt.show()

print(df[["EAR", "EAR_asym", "PERCLOS", "BROW", "PITCH", "fatigue_score"]].describe())

# ─── 7. Simpan dataset ───────────────────────────────────────────────────────
processed_path = ROOT / "data" / "processed" / "features.csv"
df.to_csv(str(processed_path), index=False)
print(f"\nDataset tersimpan: {processed_path} ({len(df)} baris)")

# ─── 8. Split train / test ───────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

features = ["EAR", "EAR_asym", "PERCLOS", "BROW", "PITCH"]
target   = "fatigue_score"

X = df[features].values
y = df[target].values

scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=df["label"],
)

print(f"\nTrain: {X_train.shape[0]} sampel")
print(f"Test : {X_test.shape[0]} sampel")

# ─── 9. Training ANFIS ───────────────────────────────────────────────────────
from anfis.anfis_model import ANFIS
from anfis.train import ANFISTrainer

model   = ANFIS(n_mf=3)
trainer = ANFISTrainer(model, lr=0.01)

print("\nMulai training ANFIS...")
history = trainer.train(X_train, y_train, epochs=100, verbose=True)

# Plot learning curve
plt.figure(figsize=(8, 4))
plt.plot(history)
plt.title("ANFIS Training — RMSE per Epoch")
plt.xlabel("Epoch")
plt.ylabel("RMSE")
plt.grid(True)
plt.tight_layout()
plt.savefig(str(ROOT / "data" / "processed" / "training_curve.png"))
plt.show()

# ─── 10. Evaluasi ────────────────────────────────────────────────────────────
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

y_pred = model.predict(X_test)

mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print("=" * 35)
print(f"  RMSE : {rmse:.4f}")
print(f"  MAE  : {mae:.4f}")
print(f"  R²   : {r2:.4f}")
print("=" * 35)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.4, color="steelblue")
plt.plot([0, 100], [0, 100], "r--", label="Ideal")
plt.xlabel("Fatigue Score Aktual")
plt.ylabel("Fatigue Score Prediksi")
plt.title("Prediksi vs Aktual")
plt.legend()
plt.tight_layout()
plt.savefig(str(ROOT / "data" / "processed" / "prediction_vs_actual.png"))
plt.show()

# ─── 11. Export Model ────────────────────────────────────────────────────────
import pickle

trainer.save_model(str(ROOT / "models" / "anfis_model.pkl"))

with open(ROOT / "models" / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved → models/scaler.pkl")

print("\n✓ Selesai! Semua artefak tersimpan di folder models/ dan data/processed/")
