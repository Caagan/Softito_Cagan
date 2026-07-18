import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  LOGISTIC REGRESYON — Kredi Kartı Dolandırıcılık Tespiti")
print("=" * 70)

print("\n[PROJE] Credit Card Fraud Detection Veri Seti")
print("-" * 50)

try:
    df = pd.read_csv("https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv")
    print("    Veri yüklendi: creditcard.csv")
except Exception:
    print("    İnternet bağlantısı yok, alternatif veri deneniyor...")
    from sklearn.datasets import make_classification
    X_syn, y_syn = make_classification(n_samples=5000, n_features=30, n_informative=15, n_redundant=5, weights=[0.95, 0.05], random_state=42)
    cols = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]
    df = pd.DataFrame(X_syn, columns=cols)
    df["Class"] = y_syn

print(f"    Veri boyutu: {df.shape}")
print(f"    Sınıf dağılımı:\n{df['Class'].value_counts()}")

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_prob = model.predict_proba(X_test_scaled)[:, 1]

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\n    Doğruluk             : {acc:.4f}")
print(f"    ROC-AUC              : {roc_auc:.4f}")
print(f"\n    Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=["Normal", "Dolandırıcılık"]))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Logistic Regresyon — Kredi Kartı Dolandırıcılık Tespiti", fontsize=14)

cm = confusion_matrix(y_test, y_pred)
im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
axes[0].set_xticklabels(["Normal", "Dolandırıcılık"])
axes[0].set_yticklabels(["Normal", "Dolandırıcılık"])
axes[0].set_xlabel("Tahmin")
axes[0].set_ylabel("Gerçek")
axes[0].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=14, color="white" if cm[i, j] > cm.max()/2 else "black")

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color="#3B8BD4", linewidth=2, label=f"AUC = {roc_auc:.3f}")
axes[1].plot([0, 1], [0, 1], "r--")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].set_title("ROC Eğrisi")
axes[1].legend()
axes[1].grid(alpha=0.3)

feature_importance = pd.Series(model.coef_[0], index=X.columns).sort_values()
feature_importance.tail(10).plot(kind="barh", ax=axes[2], color="#1D9E75")
axes[2].set_title("En Önemli 10 Özellik")
axes[2].grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Supervised\\02-logistic-regresyon\\01_kredi_dolandiricilik.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — ROC-AUC={roc_auc:.4f}")
print(f"{'='*70}")
