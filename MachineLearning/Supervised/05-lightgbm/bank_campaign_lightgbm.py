import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, ConfusionMatrixDisplay
import lightgbm as lgb
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  LIGHTGBM — Bank Marketing Campaign Tahmini")
print("=" * 70)

print("\n[PROJE] Banka Telefon Kampanyasına Katılım Tahmini")
print("-" * 50)

try:
    df = pd.read_csv("https://raw.githubusercontent.com/datasets-br/bank-marketing/main/data/bank.csv", sep=";")
    print("    Veri yüklendi: bank.csv")
except Exception:
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/mohitguptavpn/bank-marketing/master/bank.csv", sep=";")
    except Exception:
        print("    Alternatif veri deneniyor...")
        from sklearn.datasets import make_classification
        X_syn, y_syn = make_classification(n_samples=3000, n_features=15, n_informative=10, weights=[0.88, 0.12], random_state=42)
        df = pd.DataFrame(X_syn, columns=[f"f{i}" for i in range(15)])
        df["y"] = y_syn

print(f"    Veri boyutu: {df.shape}")
print(f"    Sütunlar: {list(df.columns)}")

for col in df.select_dtypes(include="object").columns:
    df[col] = pd.Categorical(df[col]).codes

y_col = "y" if "y" in df.columns else df.columns[-1]
X = df.drop(y_col, axis=1)
y = df[y_col]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

train_data = lgb.Dataset(X_train, label=y_train)
test_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

params = {
    "objective": "binary",
    "metric": "binary_logloss",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

model = lgb.train(params, train_data, num_boost_round=1000, valid_sets=[test_data],
                  callbacks=[lgb.early_stopping(stopping_rounds=50), lgb.log_evaluation(period=0)])

y_prob = model.predict(X_test)
y_pred = (y_prob > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

print(f"\n    En İyi İterasyon      : {model.best_iteration}")
print(f"    Doğruluk              : {acc:.4f}")
print(f"    ROC-AUC               : {roc_auc:.4f}")
print(f"\n    Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred))

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("LightGBM — Bank Marketing Tahmini", fontsize=14)

lgb.plot_importance(model, ax=axes[0], max_num_features=10, color="#3B8BD4")
axes[0].set_title("En Önemli 10 Özellik")

from sklearn.metrics import roc_curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[1].plot(fpr, tpr, color="#1D9E75", linewidth=2, label=f"AUC = {roc_auc:.3f}")
axes[1].plot([0, 1], [0, 1], "r--")
axes[1].set_title("ROC Eğrisi")
axes[1].legend()
axes[1].grid(alpha=0.3)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred, cmap="Greens", ax=axes[2])
axes[2].set_title("Confusion Matrix")

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Supervised\\05-lightgbm\\01_bank_kampanya.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — ROC-AUC={roc_auc:.4f}")
print(f"{'='*70}")
