import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, precision_recall_curve, average_precision_score, roc_auc_score, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  RANDOM FOREST — Credit Card Fraud Detection")
print("=" * 70)

print("\n[PROJE] Kredi Kartı Dolandırıcılık Tespiti (Imbalanced)")
print("-" * 50)

try:
    df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")
except Exception:
    from sklearn.datasets import make_classification
    X_syn, y_syn = make_classification(n_samples=5000, n_features=30, n_informative=15, weights=[0.97, 0.03], random_state=42)
    df = pd.DataFrame(X_syn, columns=[f"V{i}" for i in range(1, 31)])
    df["Class"] = y_syn

print(f"    Veri boyutu: {df.shape}")
print(f"    Sınıf: {df['Class'].value_counts().to_dict()}")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

rf = RandomForestClassifier(n_estimators=200, max_depth=12, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)

print(f"\n    Accuracy            : {acc:.4f}")
print(f"    ROC-AUC             : {roc_auc:.4f}")
print(f"    PR-AUC              : {pr_auc:.4f}")
print(f"\n    Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Dolandırıcılık"]))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Random Forest — Kredi Kartı Dolandırıcılık Tespiti", fontsize=14)

precision, recall, _ = precision_recall_curve(y_test, y_prob)
axes[0].plot(recall, precision, color="#3B8BD4", linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")
axes[0].set_xlabel("Recall")
axes[0].set_ylabel("Precision")
axes[0].set_title("Precision-Recall Eğrisi")
axes[0].legend()
axes[0].grid(alpha=0.3)

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()
importances.tail(10).plot(kind="barh", ax=axes[1], color="#E24B4A")
axes[1].set_title("En Önemli 10 Özellik")
axes[1].grid(alpha=0.3, axis="x")

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=["Normal", "Dolandırıcılık"], cmap="Oranges", ax=axes[2])
axes[2].set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Supervised\\04-random-forest\\01_dolandiricilik.png", dpi=120)
plt.close()

print(f"\n    Neden PR-AUC?")
print(f"    Sınıflar dengesiz (%97 Normal, %3 Dolandırıcılık)")
print(f"    Accuracy kandırıcı olabilir, PR-AUC daha güvenilir")

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — PR-AUC={pr_auc:.4f}, ROC-AUC={roc_auc:.4f}")
print(f"{'='*70}")
