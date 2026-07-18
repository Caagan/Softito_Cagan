import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  ONE-CLASS SVM — Normal Veriyle Anomali Tespiti")
print("=" * 70)

print("\n[PROJE] Sentetik Anomali Tespiti (One-Class Learning)")
print("-" * 50)

np.random.seed(42)
X_normal, _ = make_blobs(n_samples=500, centers=1, cluster_std=0.8, random_state=42)
X_outliers = np.random.uniform(low=X_normal.min(axis=0) - 2, high=X_normal.max(axis=0) + 2, size=(50, 2))

X_all = np.vstack([X_normal, X_outliers])
y_true = np.array([0] * len(X_normal) + [1] * len(X_outliers))

print(f"    Normal: {len(X_normal)}, Anomali: {len(X_outliers)}")

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_all)

oc_svm = OneClassSVM(kernel="rbf", gamma="auto", nu=0.05)
oc_svm.fit(X_scaled[:len(X_normal)])

y_pred_raw = oc_svm.predict(X_scaled)
y_pred = np.where(y_pred_raw == -1, 1, 0)

tp = ((y_pred == 1) & (y_true == 1)).sum()
fp = ((y_pred == 1) & (y_true == 0)).sum()
fn = ((y_pred == 0) & (y_true == 1)).sum()
tn = ((y_pred == 0) & (y_true == 0)).sum()

recall = tp / (tp + fn) if (tp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
roc_auc = roc_auc_score(y_true, oc_svm.decision_function(X_scaled) * -1)

print(f"\n    True Positive  : {tp}")
print(f"    False Positive : {fp}")
print(f"    False Negative : {fn}")
print(f"    True Negative  : {tn}")
print(f"    Recall         : {recall:.4f}")
print(f"    Precision      : {precision:.4f}")
print(f"    ROC-AUC        : {roc_auc:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("One-Class SVM — Anomali Tespiti", fontsize=14)

axes[0].scatter(X_all[y_true == 0, 0], X_all[y_true == 0, 1], c="#3B8BD4", label="Normal", alpha=0.5, s=20)
axes[0].scatter(X_all[y_true == 1, 0], X_all[y_true == 1, 1], c="#E24B4A", label="Anomali", alpha=0.7, s=30)
axes[0].set_title("Veri Dağılımı")
axes[0].legend()
axes[0].grid(alpha=0.3)

fpr, tpr, _ = roc_curve(y_true, oc_svm.decision_function(X_scaled) * -1)
axes[1].plot(fpr, tpr, color="#1D9E75", linewidth=2, label=f"AUC = {roc_auc:.3f}")
axes[1].plot([0, 1], [0, 1], "r--")
axes[1].set_title("ROC Eğrisi")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].scatter(X_all[y_pred == 0, 0], X_all[y_pred == 0, 1], c="#3B8BD4", label="Normal", alpha=0.5, s=20)
axes[2].scatter(X_all[y_pred == 1, 0], X_all[y_pred == 1, 1], c="#E24B4A", label="Tespit Edilen Anomali", alpha=0.7, s=30)
axes[2].set_title("One-Class SVM Tespit Sonucu")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Unsupervised\\04-one-class-svm\\01_one_class_svm.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — ROC-AUC={roc_auc:.4f}")
print(f"{'='*70}")
