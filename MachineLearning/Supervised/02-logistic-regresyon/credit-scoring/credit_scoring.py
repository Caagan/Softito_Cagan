import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  CREDIT SCORING — Kredi Skoru Tahmini")
print("=" * 70)

print("\n[PROJE] Credit Card Fraud — Kredi Onay/Red Siniflandirmasi")
print("-" * 50)

try:
    df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")
except Exception:
    from sklearn.datasets import make_classification
    X_syn, y_syn = make_classification(n_samples=5000, n_features=20, n_informative=12,
                                       weights=[0.92, 0.08], random_state=42)
    df = pd.DataFrame(X_syn, columns=[f"V{i}" for i in range(1, 21)])
    df["Class"] = y_syn

print(f"    Veri boyutu: {df.shape}")
print(f"    Sinif: {df['Class'].value_counts().to_dict()}")

X = df.drop("Class", axis=1)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
lr.fit(X_train_scaled, y_train)
y_pred = lr.predict(X_test_scaled)
y_prob = lr.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc = roc_auc_score(y_test, y_prob)
cv = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring="roc_auc").mean()

print(f"\n    Test Dogruluk : {acc:.4f}")
print(f"    ROC-AUC       : {roc:.4f}")
print(f"    5-Fold CV AUC : {cv:.4f}")
print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Credit Scoring — Logistic Regression", fontsize=14)

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, linewidth=2, label=f"AUC = {roc:.3f}")
axes[0].plot([0, 1], [0, 1], "r--")
axes[0].set_title("ROC Egrisi")
axes[0].legend()
axes[0].grid(alpha=0.3)

from sklearn.metrics import precision_recall_curve
prec, rec, _ = precision_recall_curve(y_test, y_prob)
axes[1].plot(rec, prec, linewidth=2, color="#E24B4A")
axes[1].set_xlabel("Recall")
axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Egrisi")
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\MachineLearning\Supervised\02-logistic-regresyon\credit-scoring\01_credit_scoring.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — ROC-AUC={roc:.4f}")
print(f"{'='*70}")
