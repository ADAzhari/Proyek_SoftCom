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
from feature_engineering.au_calculator import compute_au_vector
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

# Jumlah gambar per kelas. Set ke None untuk semua.
MAX_PER_CLASS = 3000

extractor = LandmarkExtractor()
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

        records.append(
            {
                "EAR":       au["EAR"],
                "EAR_asym":  abs(au["EAR_left"] - au["EAR_right"]),
                "MAR":       au["MAR"],
                "BROW":      au["BROW"],
                "PITCH":     pitch,
                "label":     label,
            }
        )

df = pd.DataFrame(records)
print(f"\nTotal sampel berhasil diekstrak: {len(df)}")
print(f"Distribusi label:\n{df['label'].value_counts()}")
print(df.head())

# ─── 6. Visualisasi distribusi fitur ──────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for ax, feat in zip(axes, ["EAR", "EAR_asym", "MAR", "BROW", "PITCH"]):
    df.boxplot(column=feat, by="label", ax=ax)
    ax.set_title(feat)
    ax.set_xlabel("Label (0=Normal, 1=Drowsy)")
plt.suptitle("Distribusi Fitur per Label", fontsize=14)
plt.tight_layout()
plt.savefig(str(ROOT / "data" / "processed" / "feature_distribution.png"))
plt.show()

print(df[["EAR", "EAR_asym", "MAR", "BROW", "PITCH", "label"]].describe())

# ─── 7. Simpan dataset ───────────────────────────────────────────────────────
processed_path = ROOT / "data" / "processed" / "features.csv"
df.to_csv(str(processed_path), index=False)
print(f"\nDataset tersimpan: {processed_path} ({len(df)} baris)")

# ─── 8. Split train / test ───────────────────────────────────────────────────
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

features = ["EAR", "EAR_asym", "MAR", "BROW", "PITCH"]
target   = "label"

X = df[features].values
y = df[target].values.astype(np.float64)

scaler   = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print(f"\nTrain: {X_train.shape[0]} sampel")
print(f"Test : {X_test.shape[0]} sampel")

# ─── 8b. Baseline: sklearn LogisticRegression (sanity check) ─────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc_score

baseline = LogisticRegression(max_iter=1000)
baseline.fit(X_train, y_train)
baseline_acc = acc_score(y_test.astype(int), baseline.predict(X_test))
print(f"\n[Baseline] Logistic Regression accuracy: {baseline_acc:.4f}")
print("  (Jika baseline juga rendah, berarti fitur kurang diskriminatif)")

# ─── 9. Training ANFIS ───────────────────────────────────────────────────────
from anfis.anfis_model import ANFIS
from anfis.train import ANFISTrainer

model   = ANFIS(n_mf=2)
trainer = ANFISTrainer(model, lr=0.005)

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
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix,
                             classification_report)

y_pred_raw = model.predict(X_test)
y_pred_cls = (y_pred_raw >= 0.5).astype(int)
y_test_cls = y_test.astype(int)

acc  = accuracy_score(y_test_cls, y_pred_cls)
f1   = f1_score(y_test_cls, y_pred_cls)
prec = precision_score(y_test_cls, y_pred_cls)
rec  = recall_score(y_test_cls, y_pred_cls)

print("=" * 40)
print(f"  ANFIS Accuracy  : {acc:.4f}")
print(f"  ANFIS F1 Score  : {f1:.4f}")
print(f"  ANFIS Precision : {prec:.4f}")
print(f"  ANFIS Recall    : {rec:.4f}")
print(f"  Baseline (LR)   : {baseline_acc:.4f}")
print("=" * 40)
print("\nClassification Report:")
print(classification_report(y_test_cls, y_pred_cls,
                            target_names=["Normal", "Drowsy"]))

# Confusion Matrix
cm = confusion_matrix(y_test_cls, y_pred_cls)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Confusion Matrix
im = axes[0].imshow(cm, cmap="Blues")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i, j]), ha="center", va="center",
                     fontsize=16, color="white" if cm[i, j] > cm.max()/2 else "black")
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["Normal", "Drowsy"])
axes[0].set_yticklabels(["Normal", "Drowsy"])
axes[0].set_xlabel("Prediksi")
axes[0].set_ylabel("Aktual")
axes[0].set_title("Confusion Matrix")
fig.colorbar(im, ax=axes[0])

# Plot 2: Distribusi output ANFIS
axes[1].hist(y_pred_raw[y_test_cls == 0], bins=20, alpha=0.7, color="green", label="Normal")
axes[1].hist(y_pred_raw[y_test_cls == 1], bins=20, alpha=0.7, color="red", label="Drowsy")
axes[1].axvline(x=0.5, color="black", linestyle="--", label="Threshold 0.5")
axes[1].legend()
axes[1].set_xlabel("ANFIS Output")
axes[1].set_ylabel("Jumlah Sampel")
axes[1].set_title("Distribusi Output ANFIS")

plt.tight_layout()
plt.savefig(str(ROOT / "data" / "processed" / "evaluation.png"))
plt.show()

# ─── 11. Export Model ────────────────────────────────────────────────────────
import pickle

trainer.save_model(str(ROOT / "models" / "anfis_model.pkl"))

with open(ROOT / "models" / "scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved → models/scaler.pkl")

print("\n✓ Selesai! Semua artefak tersimpan di folder models/ dan data/processed/")
