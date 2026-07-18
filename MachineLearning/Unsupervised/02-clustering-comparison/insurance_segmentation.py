import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  KLÜMELEME KARŞILAŞTIRMASI — 4 Algoritma, Wine Dataset")
print("=" * 70)

print("\n[PROJE] Wine Dataset — K-Means, Hiyerarşik, DBSCAN, GMM")
print("-" * 50)

data = load_wine()
X = pd.DataFrame(data.data, columns=data.feature_names)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print(f"    Veri boyutu: {X.shape}")

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_sil = silhouette_score(X_scaled, kmeans_labels)

hierarchical = AgglomerativeClustering(n_clusters=3)
hier_labels = hierarchical.fit_predict(X_scaled)
hier_sil = silhouette_score(X_scaled, hier_labels)

dbscan = DBSCAN(eps=2.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
n_clusters_db = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
dbscan_sil = silhouette_score(X_scaled, dbscan_labels) if n_clusters_db > 1 else -1

gmm = GaussianMixture(n_components=3, random_state=42)
gmm_labels = gmm.fit_predict(X_scaled)
gmm_sil = silhouette_score(X_scaled, gmm_labels)

print(f"\n    {'Algoritma':<20} {'Küme':<10} {'Silhouette':<12}")
print(f"    {'─'*42}")
print(f"    {'K-Means':<20} {len(set(kmeans_labels)):<10} {kmeans_sil:<12.4f}")
print(f"    {'Hiyerarşik':<20} {len(set(hier_labels)):<10} {hier_sil:<12.4f}")
print(f"    {'DBSCAN':<20} {n_clusters_db:<10} {dbscan_sil:<12.4f}")
print(f"    {'GMM':<20} {len(set(gmm_labels)):<10} {gmm_sil:<12.4f}")

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
fig.suptitle("Klümeleme Algoritmaları Karşılaştırması — Wine", fontsize=14)

all_labels = [("K-Means", kmeans_labels), ("Hiyerarşik", hier_labels),
              ("DBSCAN", dbscan_labels), ("GMM", gmm_labels)]

for ax, (name, labels) in zip(axes.flat, all_labels):
    unique_labels = set(labels)
    colors_map = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
    for label, color in zip(unique_labels, colors_map):
        if label == -1:
            ax.scatter(X_pca[labels == label, 0], X_pca[labels == label, 1], c="gray", marker="x", s=10, alpha=0.3, label="Gürültü")
        else:
            mask = labels == label
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=[color], label=f"Küme {label}", alpha=0.5, s=15)
    ax.set_title(f"{name} (Küme: {len(unique_labels) - (1 if -1 in unique_labels else 0)})")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Unsupervised\\02-clustering-comparison\\01_karsilastirma.png", dpi=120)
plt.close()

fig, ax = plt.subplots(figsize=(8, 5))
algortimalar = ["K-Means", "Hiyerarşik", "DBSCAN", "GMM"]
sil_scores = [kmeans_sil, hier_sil, dbscan_sil, gmm_sil]
colors = ["#3B8BD4", "#1D9E75", "#E24B4A", "#F5A623"]
bars = ax.bar(algortimalar, sil_scores, color=colors, width=0.5)
ax.bar_label(bars, fmt="%.4f", padding=4)
ax.set_ylim(0, 1)
ax.set_ylabel("Silhouette Skoru")
ax.set_title("Algoritma Karşılaştırması")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Unsupervised\\02-clustering-comparison\\02_silhouette.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI")
print(f"{'='*70}")
