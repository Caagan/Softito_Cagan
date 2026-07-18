import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  MODEL KARŞILAŞTIRMA — Aynı Veri, 4 Farklı Algoritma")
print("=" * 70)

print("\n[PROJE] Breast Cancer — 5-Fold Cross-Validation")
print("-" * 50)

data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target)

print(f"    Veri boyutu: {X.shape}")
print(f"    Sınıf: {dict(pd.Series(y).value_counts())}")

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric="logloss", random_state=42),
    "LightGBM": lgb.LGBMClassifier(n_estimators=100, verbose=-1, random_state=42),
}

results = {}
print(f"\n    5-Fold Cross-Validation Sonuçları:")
print(f"    {'Model':<25} {'Ort Accuracy':>14} {'Std':>8} {'Ort AUC':>12}")
print(f"    {'─'*60}")

for name, model in models.items():
    acc_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc")
    results[name] = {"accuracy": acc_scores.mean(), "accuracy_std": acc_scores.std(),
                     "auc": auc_scores.mean(), "auc_std": auc_scores.std()}
    print(f"    {name:<25} {acc_scores.mean():>13.4f} {acc_scores.std():>8.4f} {auc_scores.mean():>11.4f}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Karşılaştırması — 5-Fold Cross-Validation", fontsize=14)

names = list(results.keys())
accs = [results[n]["accuracy"] for n in names]
aucs = [results[n]["auc"] for n in names]
colors = ["#3B8BD4", "#1D9E75", "#E24B4A", "#F5A623"]

bars1 = axes[0].bar(names, accs, color=colors, width=0.5)
axes[0].bar_label(bars1, fmt="%.4f", padding=4)
axes[0].set_ylim(0.9, 1)
axes[0].set_ylabel("Accuracy")
axes[0].set_title("Accuracy Karşılaştırması")
axes[0].grid(axis="y", alpha=0.3)
axes[0].tick_params(axis="x", rotation=15)

bars2 = axes[1].bar(names, aucs, color=colors, width=0.5)
axes[1].bar_label(bars2, fmt="%.4f", padding=4)
axes[1].set_ylim(0.9, 1)
axes[1].set_ylabel("ROC-AUC")
axes[1].set_title("ROC-AUC Karşılaştırması")
axes[1].grid(axis="y", alpha=0.3)
axes[1].tick_params(axis="x", rotation=15)

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Supervised\\Model Karsilastirma\\01_model_karsilastirma.png", dpi=120)
plt.close()

best_model = max(results.items(), key=lambda x: x[1]["auc"])
print(f"\n    En İyi Model: {best_model[0]} (AUC={best_model[1]['auc']:.4f})")

print(f"\n{'='*70}")
print(f"  TAMAMLANDI")
print(f"{'='*70}")
