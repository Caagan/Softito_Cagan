import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  MOBILE PRICE — Fiyat Segmenti Tahmini")
print("=" * 70)

print("\n[PROJE] Wine Quality — Kalite Siniflandirmasi (UCI)")
print("-" * 50)

try:
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")
except Exception:
    url = "https://raw.githubusercontent.com/rashida048/Datasets/master/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

print(f"    Veri boyutu: {df.shape}")
print(f"    Kalite skorlari: {sorted(df['quality'].unique())}")

df["kalite_sinifi"] = pd.cut(df["quality"], bins=[0, 4, 6, 10], labels=["Dusuk", "Orta", "Yuksek"])
y = df["kalite_sinifi"].cat.codes
X = df.drop(["quality", "kalite_sinifi"], axis=1)

print(f"    Sinif dagilimi: {df['kalite_sinifi'].value_counts().to_dict()}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=8, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n    Test Dogruluk: {acc:.4f}")
print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred, digits=4, target_names=["Dusuk", "Orta", "Yuksek"]))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Wine Quality — Decision Tree", fontsize=14)

importances = pd.Series(dt.feature_importances_, index=X.columns).sort_values(ascending=True)
importances.plot(kind="barh", ax=axes[0], color="#3B8BD4")
axes[0].set_title("Ozellik Onemleri")

plot_tree(dt, feature_names=X.columns.tolist(), class_names=["Dusuk", "Orta", "Yuksek"],
          filled=True, rounded=True, ax=axes[1], fontsize=5, max_depth=3)
axes[1].set_title("Karar Agaci (max_depth=3 gosterimi)")

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\MachineLearning\Supervised\03-decision-tree\mobile-price-decision-tree\01_mobile_price.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — Dogruluk={acc:.4f}")
print(f"{'='*70}")
