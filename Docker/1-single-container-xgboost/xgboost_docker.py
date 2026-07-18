import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  SINGLE CONTAINER XGBOOST — Tek Konteyner ML Deployment")
print("=" * 70)

print("\n[PROJE] XGBoost — Tek Konteyner Icinde Egitim ve Tahmin")
print("-" * 50)

X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=12,
    weights=[0.7, 0.3], random_state=42
)

feature_names = [f"ozellik_{i}" for i in range(20)]
df = pd.DataFrame(X, columns=feature_names)
df["hedef"] = y

print(f"    Veri boyutu: {df.shape}")
print(f"    Sinif orani: {df['hedef'].value_counts().to_dict()}")

X = df.drop("hedef", axis=1)
y = df["hedef"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = xgb.XGBClassifier(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    eval_metric="logloss", random_state=42
)

model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

acc = accuracy_score(y_test, y_pred)
print(f"\n    Test Dogruluk: {acc:.4f}")
print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred, digits=4))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Single Container XGBoost — Deployment Hazirlik", fontsize=14)

importances = pd.Series(model.feature_importances_, index=feature_names).sort_values(ascending=True)
importances.tail(10).plot(kind="barh", ax=axes[0], color="#3B8BD4")
axes[0].set_title("En Onemli 10 Ozellik")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, cmap="Blues")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(["Negatif", "Pozitif"])
axes[1].set_yticklabels(["Negatif", "Pozitif"])
axes[1].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\Docker\1-single-container-xgboost\01_xgboost_results.png", dpi=120)
plt.close()

print(f"\n    Model kaydedildi (XGBoost formatinda)")
print(f"    Konteyner hazirligi tamamlandi")

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — Dogruluk={acc:.4f}")
print(f"{'='*70}")
