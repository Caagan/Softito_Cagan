import numpy as np
import pandas as pd
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from PIL import Image
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  GORUNTU ON ISLEME — Temel Goruntu Isleme Teknikleri")
print("=" * 70)

print("\n[PROJE] Sklearn Digits — Goruntu On Isleme ve Siniflandirma")
print("-" * 50)

digits = load_digits()
X = digits.data
y = digits.target

print(f"    Veri boyutu: {X.shape}")
print(f"    Sinif sayisi: {len(np.unique(y))}")
print(f"    Ornek boyut: {digits.images[0].shape}")

images = digits.images

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle("Orijinal Digits Gorselleri", fontsize=14)
for i, ax in enumerate(axes.flat):
    ax.imshow(images[i], cmap="gray")
    ax.set_title(f"Etiket: {y[i]}", fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\DeepLearning\CNN\02-goruntu-on-isleme\01_original_digits.png", dpi=120)
plt.close()

print(f"\n[1] Piksel Duzeyinde Islem")
print("-" * 50)

fig, axes = plt.subplots(2, 4, figsize=(14, 6))
fig.suptitle("Goruntu On Isleme Teknikleri", fontsize=14)

axes[0, 0].imshow(images[0], cmap="gray")
axes[0, 0].set_title("Orijinal")

brightness = images[0] * 1.5
axes[0, 1].imshow(np.clip(brightness, 0, 16), cmap="gray")
axes[0, 1].set_title("Parlaklik (x1.5)")

threshold = (images[0] > 6).astype(float) * 16
axes[0, 2].imshow(threshold, cmap="gray")
axes[0, 2].set_title("Esikleme (>6)")

axes[0, 3].imshow(np.flipud(images[0]), cmap="gray")
axes[0, 3].set_title("Dikey Cevirme")

axes[1, 0].imshow(np.fliplr(images[0]), cmap="gray")
axes[1, 0].set_title("Yatay Cevirme")

rotated = np.rot90(images[0])
axes[1, 1].imshow(rotated, cmap="gray")
axes[1, 1].set_title("90 Derece Donme")

noise = images[0] + np.random.normal(0, 2, images[0].shape)
axes[1, 2].imshow(np.clip(noise, 0, 16), cmap="gray")
axes[1, 2].set_title("Gurultu Ekleme")

from scipy.ndimage import gaussian_filter
blurred = gaussian_filter(images[0], sigma=1.5)
axes[1, 3].imshow(blurred, cmap="gray")
axes[1, 3].set_title("Gaussian Bulaniklik")

for ax in axes.flat:
    ax.axis("off")

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\DeepLearning\CNN\02-goruntu-on-isleme\02_preprocessing.png", dpi=120)
plt.close()

print(f"    Piksel araligi: [{images[0].min()}, {images[0].max()}]")
print(f"    Ortalama parlaklik: {images[0].mean():.2f}")
print(f"    Standart sapma: {images[0].std():.2f}")

print(f"\n[2] Ozellik Cikarma ve Siniflandirma")
print("-" * 50)

X_flattened = X / 16.0
X_train, X_test, y_train, y_test = train_test_split(X_flattened, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=15, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
acc_dt = accuracy_score(y_test, y_pred_dt)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)

print(f"    Decision Tree  : {acc_dt:.4f}")
print(f"    Random Forest  : {acc_rf:.4f}")

print(f"\n    Random Forest Classification Report:")
print(classification_report(y_test, y_pred_rf, digits=4))

print(f"\n[3] Karakteristik Ozellik Analizi")
print("-" * 50)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Goruntu Ozellik Analizi", fontsize=14)

axes[0].bar(range(10), [np.sum(y == i) for i in range(10)], color="#3B8BD4")
axes[0].set_title("Sinif Dagilimi")
axes[0].set_xlabel("Rakam")
axes[0].set_ylabel("Adet")

pixel_importances = pd.Series(rf.feature_importances_.reshape(8, 8), index=range(8), columns=range(8))
pixel_importances.plot(kind="bar", ax=axes[1], colormap="viridis", legend=False)
axes[1].set_title("Piksel Onemleri (RF)")
axes[1].set_xlabel("Satir")

importances = pd.Series(rf.feature_importances_).sort_values(ascending=True)
importances.tail(15).plot(kind="barh", ax=axes[2], color="#E24B4A")
axes[2].set_title("En Onemli 15 Piksel")

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\DeepLearning\CNN\02-goruntu-on-isleme\03_feature_analysis.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — RF Dogruluk={acc_rf:.4f}")
print(f"{'='*70}")
