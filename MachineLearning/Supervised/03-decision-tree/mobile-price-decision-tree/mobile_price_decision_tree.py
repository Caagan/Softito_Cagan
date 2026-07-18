import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  MOBILE PRICE — Fiyat Segmenti Tahmini")
print("=" * 70)

print("\n[PROJE] Cihoz Ozelliklerinden Fiyat Segmenti Siniflandirmasi")
print("-" * 50)

X, y = make_classification(
    n_samples=2000, n_features=12, n_informative=8, n_classes=4,
    n_clusters_per_class=1, random_state=42
)
feature_names = ["battery_power", "blue", "clock_speed", "dual_sim", "fc",
                 "four_g", "int_memory", "m_dep", "mobile_wt", "n_cores",
                 "pc", "px_height"]
df = pd.DataFrame(X, columns=feature_names)
df["price_range"] = y
price_labels = {0: "Dusuk", 1: "Orta-Dusuk", 2: "Orta-Yuksek", 3: "Yuksek"}
df["price_label"] = df["price_range"].map(price_labels)

print(f"    Veri boyutu: {df.shape}")
print(f"    Fiyat segmentleri: {df['price_label'].value_counts().to_dict()}")

X = df.drop(["price_range", "price_label"], axis=1)
y = df["price_range"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=8, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n    Test Dogruluk: {acc:.4f}")
print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred, digits=4, target_names=["Dusuk", "Orta-Dusuk", "Orta-Yuksek", "Yuksek"]))

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle("Mobile Price Segmenti — Decision Tree", fontsize=14)

importances = pd.Series(dt.feature_importances_, index=feature_names).sort_values(ascending=True)
importances.plot(kind="barh", ax=axes[0], color="#3B8BD4")
axes[0].set_title("Ozellik Onemleri")

from sklearn.tree import plot_tree
plot_tree(dt, feature_names=feature_names, class_names=["Dusuk", "Orta-Dusuk", "Orta-Yuksek", "Yuksek"],
          filled=True, rounded=True, ax=axes[1], fontsize=5, max_depth=3)
axes[1].set_title("Karar Agaci (max_depth=3 gosterimi)")

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\MachineLearning\Supervised\03-decision-tree\mobile-price-decision-tree\01_mobile_price.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — Dogruluk={acc:.4f}")
print(f"{'='*70}")
