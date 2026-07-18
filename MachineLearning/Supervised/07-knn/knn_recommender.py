import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  KNN — Iris Çiçeği Sınıflandırması (K Komşu Arama)")
print("=" * 70)

print("\n[PROJE] Klassik Iris Veri Seti ile KNN Optimizasyonu")
print("-" * 50)

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["species"] = pd.Categorical.from_codes(data.target, data.target_names)

print(f"    Veri boyutu: {df.shape}")
print(f"    Sınıflar: {df['species'].value_counts().to_dict()}")

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

k_range = range(1, 31)
train_scores, test_scores = [], []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_scaled, y_train)
    train_scores.append(accuracy_score(y_train, knn.predict(X_train_scaled)))
    test_scores.append(accuracy_score(y_test, knn.predict(X_test_scaled)))

best_k = k_range[np.argmax(test_scores)]
best_score = max(test_scores)

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_scaled, y_train)
y_pred = final_knn.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n    En İyi K Değeri       : {best_k}")
print(f"    En İyi Test Doğruluğu: {acc:.4f}")
print(f"\n    Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("KNN — Iris Sınıflandırması", fontsize=14)

axes[0].plot(k_range, train_scores, "o-", label="Eğitim", color="#3B8BD4")
axes[0].plot(k_range, test_scores, "o-", label="Test", color="#E24B4A")
axes[0].axvline(best_k, color="green", linestyle="--", label=f"En İyi K={best_k}")
axes[0].set_xlabel("K Değeri")
axes[0].set_ylabel("Doğruluk")
axes[0].set_title("K Değeri vs Doğruluk")
axes[0].legend()
axes[0].grid(alpha=0.3)

from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_test_scaled)
colors = ["#E24B4A", "#3B8BD4", "#1D9E75"]
for i, species in enumerate(data.target_names):
    mask = y_test.values == species
    axes[1].scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[i], label=species, alpha=0.7, s=50)
axes[1].set_xlabel("PC1")
axes[1].set_ylabel("PC2")
axes[1].set_title("PCA ile 2D Görselleştirme")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Supervised\\07-knn\\01_iris_knn.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — En İyi K={best_k}, Doğruluk={acc:.4f}")
print(f"{'='*70}")
