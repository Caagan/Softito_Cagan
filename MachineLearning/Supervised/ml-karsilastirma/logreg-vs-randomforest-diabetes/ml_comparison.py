import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  LOGISTIC REGRESSION vs RANDOM FOREST — Diabetes Siniflandirma")
print("=" * 70)

print("\n[PROJE] Diabetes Veri Seti — Model Karsilastirma")
print("-" * 50)

data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = (data.target > data.target.median()).astype(int)

print(f"    Veri boyutu: {df.shape}")
print(f"    Siniflar: {df['target'].value_counts().to_dict()}")

X = df.drop("target", axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)
y_pred_lr = lr.predict(X_test_scaled)
y_prob_lr = lr.predict_proba(X_test_scaled)[:, 1]
acc_lr = accuracy_score(y_test, y_pred_lr)
roc_lr = roc_auc_score(y_test, y_prob_lr)
cv_lr = cross_val_score(lr, X_train_scaled, y_train, cv=5, scoring="accuracy").mean()

rf = RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
acc_rf = accuracy_score(y_test, y_pred_rf)
roc_rf = roc_auc_score(y_test, y_prob_rf)
cv_rf = cross_val_score(rf, X_train, y_train, cv=5, scoring="accuracy").mean()

print(f"\n    {'Metrik':<25} {'LogReg':>12} {'RandForest':>12}")
print(f"    {'─'*50}")
print(f"    {'Test Dogruluk':<25} {acc_lr:>12.4f} {acc_rf:>12.4f}")
print(f"    {'ROC-AUC':<25} {roc_lr:>12.4f} {roc_rf:>12.4f}")
print(f"    {'5-Fold CV Dogruluk':<25} {cv_lr:>12.4f} {cv_rf:>12.4f}")

print(f"\n    Logistic Regression:")
print(classification_report(y_test, y_pred_lr, digits=4))
print(f"    Random Forest:")
print(classification_report(y_test, y_pred_rf, digits=4))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("LogReg vs RandomForest — Diabetes", fontsize=14)

fpr_lr, tpr_lr, _ = roc_curve(y_test, y_prob_lr)
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
axes[0].plot(fpr_lr, tpr_lr, label=f"LogReg (AUC={roc_lr:.3f})", linewidth=2)
axes[0].plot(fpr_rf, tpr_rf, label=f"RF (AUC={roc_rf:.3f})", linewidth=2)
axes[0].plot([0, 1], [0, 1], "r--")
axes[0].set_title("ROC Egrisi")
axes[0].legend()
axes[0].grid(alpha=0.3)

importances = pd.Series(rf.feature_importances_, index=data.feature_names).sort_values(ascending=True)
importances.plot(kind="barh", ax=axes[1], color="#1D9E75")
axes[1].set_title("Random Forest Ozellik Onem Siralamsi")

models = ["LogReg", "RandomForest"]
metrics = {"Dogruluk": [acc_lr, acc_rf], "ROC-AUC": [roc_lr, roc_rf], "CV": [cv_lr, cv_rf]}
x_pos = np.arange(len(models))
width = 0.25
for i, (metrik, degerler) in enumerate(metrics.items()):
    axes[2].bar(x_pos + i * width, degerler, width, label=metrik)
axes[2].set_xticks(x_pos + width)
axes[2].set_xticklabels(models)
axes[2].set_ylim(0.5, 1)
axes[2].set_title("Metrik Karsilastirmasi")
axes[2].legend()
axes[2].grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\MachineLearning\Supervised\ml-karsilastirma\logreg-vs-randomforest-diabetes\01_karsilastirma.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI")
print(f"{'='*70}")
