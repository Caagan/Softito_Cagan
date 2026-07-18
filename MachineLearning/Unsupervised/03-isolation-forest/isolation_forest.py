import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  ISOLATION FOREST — Credit Card Fraud (Etiketsiz Tespit)")
print("=" * 70)

print("\n[PROJE] Credit Card — Anomali Tespiti")
print("-" * 50)

try:
    df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")
except Exception:
    from sklearn.datasets import make_classification
    X_syn, y_syn = make_classification(n_samples=5000, n_features=30, n_informative=15, weights=[0.96, 0.04], random_state=42)
    df = pd.DataFrame(X_syn, columns=[f"V{i}" for i in range(1, 31)])
    df["Class"] = y_syn

print(f"    Veri boyutu: {df.shape}")
print(f"    Sınıf: {df['Class'].value_counts().to_dict()}")

X = df.drop("Class", axis=1)
y = df["Class"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(n_estimators=200, contamination=0.02, random_state=42, n_jobs=-1)
df["anomaly_score"] = iso.fit_predict(X_scaled)
df["anomaly_raw"] = iso.decision_function(X_scaled)
df["anomaly"] = (df["anomaly_score"] == -1).astype(int)

tp = ((df["anomaly"] == 1) & (df["Class"] == 1)).sum()
fp = ((df["anomaly"] == 1) & (df["Class"] == 0)).sum()
fn = ((df["anomaly"] == 0) & (df["Class"] == 1)).sum()
tn = ((df["anomaly"] == 0) & (df["Class"] == 0)).sum()

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
roc_auc = roc_auc_score(y, -df["anomaly_raw"])

print(f"\n    True Positive  : {tp}")
print(f"    False Positive : {fp}")
print(f"    False Negative : {fn}")
print(f"    True Negative  : {tn}")
print(f"    Precision      : {precision:.4f}")
print(f"    Recall         : {recall:.4f}")
print(f"    ROC-AUC        : {roc_auc:.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Isolation Forest — Credit Card Anomali Tespiti", fontsize=14)

scores_normal = df[df["Class"] == 0]["anomaly_raw"]
scores_fraud = df[df["Class"] == 1]["anomaly_raw"]
axes[0].hist(scores_normal, bins=50, alpha=0.7, label="Normal", color="#3B8BD4", density=True)
axes[0].hist(scores_fraud, bins=50, alpha=0.7, label="Dolandırıcılık", color="#E24B4A", density=True)
axes[0].axvline(0, color="black", linestyle="--", label="Eşik")
axes[0].set_title("Anomaly Score Dağılımı")
axes[0].legend()
axes[0].grid(alpha=0.3)

fpr, tpr, _ = roc_curve(y, -df["anomaly_raw"])
axes[1].plot(fpr, tpr, color="#1D9E75", linewidth=2, label=f"AUC = {roc_auc:.3f}")
axes[1].plot([0, 1], [0, 1], "r--")
axes[1].set_title("ROC Eğrisi")
axes[1].legend()
axes[1].grid(alpha=0.3)

cm = np.array([[tn, fp], [fn, tp]])
im = axes[2].imshow(cm, cmap="Oranges")
axes[2].set_xticks([0, 1])
axes[2].set_yticks([0, 1])
axes[2].set_xticklabels(["Normal", "Anomali"])
axes[2].set_yticklabels(["Normal", "Anomali"])
axes[2].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[2].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=12,
                     color="white" if cm[i, j] > cm.max()/2 else "black")

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Unsupervised\\03-isolation-forest\\01_anomali_tespit.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — ROC-AUC={roc_auc:.4f}")
print(f"{'='*70}")
