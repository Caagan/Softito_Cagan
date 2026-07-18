import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  DECISION TREE — Klinik Karar Destek Sistemi")
print("=" * 70)

print("\n[PROJE] Breast Cancer — Tani Karar Agaci")
print("-" * 50)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df["target_name"] = df["target"].map({0: "Malign", 1: "Benign"})

print(f"    Veri boyutu: {df.shape}")
print(f"    Siniflar: {df['target_name'].value_counts().to_dict()}")

X = df.drop(["target", "target_name"], axis=1)
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

dt = DecisionTreeClassifier(max_depth=4, random_state=42)
dt.fit(X_train, y_train)
y_pred = dt.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"\n    Test Dogruluk: {acc:.4f}")
print(f"\n    Classification Report:")
print(classification_report(y_test, y_pred, digits=4, target_names=["Malign", "Benign"]))

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle("Decision Tree — Klinik Tani", fontsize=14)

plot_tree(dt, feature_names=data.feature_names, class_names=["Malign", "Benign"],
          filled=True, rounded=True, ax=axes[0], fontsize=6, max_depth=3)
axes[0].set_title("Karar Agaci (max_depth=4)")

importances = pd.Series(dt.feature_importances_, index=data.feature_names).sort_values(ascending=True)
importances[importances > 0].tail(10).plot(kind="barh", ax=axes[1], color="#1D9E75")
axes[1].set_title("En Onemli 10 Ozellik")

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\MachineLearning\Supervised\03-decision-tree\decision_tree_clinical\01_decision_tree.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — Dogruluk={acc:.4f}")
print(f"{'='*70}")
