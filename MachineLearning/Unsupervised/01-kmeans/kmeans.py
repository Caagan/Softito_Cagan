import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  K-MEANS — Wine Veri Seti ile Müşteri Segmentasyonu")
print("=" * 70)

print("\n[PROJE] Wine Dataset — Kümeleme ve Segmentasyon")
print("-" * 50)

data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print(f"    Veri boyutu: {df.shape}")
print(f"    Sınıflar: {df['target'].value_counts().to_dict()}")

X = df.drop("target", axis=1)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"\n    Elbow ve Silhouette Analizi:")
print(f"    {'K':>5}  {'Inertia':>15}  {'Silhouette':>12}")
print(f"    {'─'*35}")

K_range = range(2, 11)
inertias, silhouettes = [], []
for k in K_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_scaled, km.labels_)
    silhouettes.append(sil)
    print(f"    {k:>5}  {km.inertia_:>15,.0f}  {sil:>12.4f}")

best_k = K_range[np.argmax(silhouettes)]
print(f"\n    En İyi K (Silhouette): {best_k}")

final_km = KMeans(n_clusters=best_k, random_state=42, n_init=10)
df["kume"] = final_km.fit_predict(X_scaled)

print(f"\n    Küme Ortalamaları:")
print(df.groupby("kume")[data.feature_names[:6]].mean().round(2).to_string())

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("K-Means — Wine Segmentasyonu", fontsize=14)

colors = ["#E24B4A", "#3B8BD4", "#1D9E75"]
for k in range(best_k):
    mask = df["kume"] == k
    axes[0].scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[k % len(colors)], label=f"Küme {k}", alpha=0.5, s=20)
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].set_title(f"K={best_k} Kümeleme (PCA)")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(list(K_range), inertias, "o-", color="#3B8BD4", label="Inertia")
axes[1].axvline(best_k, color="red", linestyle="--", label=f"K={best_k}")
axes[1].set_xlabel("K")
axes[1].set_ylabel("Inertia")
axes[1].set_title("Elbow Grafiği")
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Unsupervised\\01-kmeans\\01_wine_kmeans.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — K={best_k}, Silhouette={max(silhouettes):.4f}")
print(f"{'='*70}")
