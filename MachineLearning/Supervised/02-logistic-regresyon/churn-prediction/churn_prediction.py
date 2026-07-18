import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  CHURN PREDICTION — Musteri Kaybi Tahmini")
print("=" * 70)

print("\n[PROJE] Bank Marketing (UCI) — Churn Tahmini")
print("-" * 50)

try:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"
    import zipfile, io, urllib.request
    resp = urllib.request.urlopen(url)
    z = zipfile.ZipFile(io.BytesIO(resp.read()))
    df = pd.read_csv(z.open("bank-additional/bank-additional-full.csv"), sep=";")
except Exception:
    from sklearn.datasets import fetch_openml
    data = fetch_openml(name="bank-marketing", version=1, as_frame=True)
    df = data.frame

print(f"    Veri boyutu: {df.shape}")

y_col = "y" if "y" in df.columns else "target"
if df[y_col].dtype == object or df[y_col].dtype.name == "category":
    y = (df[y_col] == "yes").astype(int)
else:
    y = df[y_col].astype(int)

X = df.drop(y_col, axis=1)
cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    le_dict[col] = le

print(f"    Siniflar: {y.value_counts().to_dict()}")
print(f"    Churn orani: %{y.mean()*100:.1f}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, random_state=42)
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

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Churn Prediction — Bank Marketing (UCI)", fontsize=14)

fpr, tpr, _ = roc_curve(y_test, y_prob)
axes[0].plot(fpr, tpr, linewidth=2, label=f"AUC = {roc:.3f}")
axes[0].plot([0, 1], [0, 1], "r--")
axes[0].set_title("ROC Egrisi")
axes[0].legend()
axes[0].grid(alpha=0.3)

cm = confusion_matrix(y_test, y_pred)
im = axes[1].imshow(cm, cmap="Blues")
axes[1].set_xticks([0, 1])
axes[1].set_yticks([0, 1])
axes[1].set_xticklabels(["Hayir", "Evet"])
axes[1].set_yticklabels(["Hayir", "Evet"])
axes[1].set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        axes[1].text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max()/2 else "black")

importances = pd.Series(np.abs(lr.coef_[0]), index=X.columns).sort_values(ascending=True)
importances.tail(10).plot(kind="barh", ax=axes[2], color="#E24B4A")
axes[2].set_title("En Onemli 10 Ozellik")

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\MachineLearning\Supervised\02-logistic-regresyon\churn-prediction\01_churn_prediction.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — ROC-AUC={roc:.4f}")
print(f"{'='*70}")
