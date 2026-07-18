import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, roc_curve
import xgboost as xgb
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  XGBOOST vs LIGHTGBM — CTR Siniflandirma Karsilastirmasi")
print("=" * 70)

print("\n[PROJE] Sendetik CTR Verisi — Gradient Boosting Karsilastirmasi")
print("-" * 50)

X, y = make_classification(
    n_samples=10000, n_features=20, n_informative=12, n_redundant=3,
    weights=[0.92, 0.08], random_state=42
)
feature_names = [f"feat_{i}" for i in range(X.shape[1])]
df = pd.DataFrame(X, columns=feature_names)
df["clicked"] = y

print(f"    Veri boyutu: {df.shape}")
print(f"    Click orani: %{y.mean()*100:.1f}")

X_train, X_test, y_train, y_test = train_test_split(df.drop("clicked", axis=1), y, test_size=0.2, random_state=42, stratify=y)

xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42, eval_metric="auc", use_label_encoder=False
)
xgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
y_pred_xgb = xgb_model.predict(X_test)
y_prob_xgb = xgb_model.predict_proba(X_test)[:, 1]
acc_xgb = accuracy_score(y_test, y_pred_xgb)
roc_xgb = roc_auc_score(y_test, y_prob_xgb)
cv_xgb = cross_val_score(xgb_model, X_train, y_train, cv=5, scoring="roc_auc").mean()

lgb_model = lgb.LGBMClassifier(
    n_estimators=300, max_depth=6, learning_rate=0.1,
    scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
    random_state=42, verbose=-1
)
lgb_model.fit(X_train, y_train, eval_set=[(X_test, y_test)], callbacks=[lgb.early_stopping(50, verbose=False)])
y_pred_lgb = lgb_model.predict(X_test)
y_prob_lgb = lgb_model.predict_proba(X_test)[:, 1]
acc_lgb = accuracy_score(y_test, y_pred_lgb)
roc_lgb = roc_auc_score(y_test, y_prob_lgb)
cv_lgb = cross_val_score(lgb_model, X_train, y_train, cv=5, scoring="roc_auc").mean()

print(f"\n    {'Metrik':<25} {'XGBoost':>12} {'LightGBM':>12}")
print(f"    {'─'*50}")
print(f"    {'Test Dogruluk':<25} {acc_xgb:>12.4f} {acc_lgb:>12.4f}")
print(f"    {'ROC-AUC':<25} {roc_xgb:>12.4f} {roc_lgb:>12.4f}")
print(f"    {'5-Fold CV AUC':<25} {cv_xgb:>12.4f} {cv_lgb:>12.4f}")

print(f"\n    XGBoost:")
print(classification_report(y_test, y_pred_xgb, digits=4))
print(f"    LightGBM:")
print(classification_report(y_test, y_pred_lgb, digits=4))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("XGBoost vs LightGBM — CTR Siniflandirma", fontsize=14)

fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
fpr_lgb, tpr_lgb, _ = roc_curve(y_test, y_prob_lgb)
axes[0].plot(fpr_xgb, tpr_xgb, label=f"XGBoost (AUC={roc_xgb:.3f})", linewidth=2)
axes[0].plot(fpr_lgb, tpr_lgb, label=f"LightGBM (AUC={roc_lgb:.3f})", linewidth=2)
axes[0].plot([0, 1], [0, 1], "r--")
axes[0].set_title("ROC Egrisi")
axes[0].legend()
axes[0].grid(alpha=0.3)

xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_names).sort_values(ascending=True).tail(10)
lgb_imp = pd.Series(lgb_model.feature_importances_, index=feature_names).sort_values(ascending=True).tail(10)

axes[1].barh(xgb_imp.index, xgb_imp.values, color="#E24B4A")
axes[1].set_title("XGBoost — En Onemli 10 Ozellik")

axes[2].barh(lgb_imp.index, lgb_imp.values, color="#3B8BD4")
axes[2].set_title("LightGBM — En Onemli 10 Ozellik")

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\MachineLearning\Supervised\ml-karsilastirma\xgboost-vs-lightgbm\01_karsilastirma.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI")
print(f"{'='*70}")
