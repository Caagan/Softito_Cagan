import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  DECISION TREE — Breast Cancer Wisconsin Sınıflandırması")
print("=" * 70)

print("\n[PROJE] Meme Kanseri Hücre Tanımlama")
print("-" * 50)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target
df["target"] = df["target"].map({0: "Malignant", 1: "Benign"})

print(f"    Veri boyutu: {df.shape}")
print(f"    Sınıflar: {df['target'].value_counts().to_dict()}")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
shallow.fit(X_train, y_train)
y_pred_shallow = shallow.predict(X_test)
acc_shallow = accuracy_score(y_test, y_pred_shallow)

deep = DecisionTreeClassifier(max_depth=None, min_samples_split=5, random_state=42)
deep.fit(X_train, y_train)
y_pred_deep = deep.predict(X_test)
acc_deep = accuracy_score(y_test, y_pred_deep)

print(f"\n    Sığ Ağaç (depth=3)    : {acc_shallow:.4f}")
print(f"    Derin Ağaç (sınırsız)  : {acc_deep:.4f}")
print(f"\n    Sınıflandırma Raporu (Sığ):")
print(classification_report(y_test, y_pred_shallow))

fig, axes = plt.subplots(1, 2, figsize=(18, 8))
fig.suptitle("Decision Tree — Breast Cancer Sınıflandırması", fontsize=14)

plot_tree(shallow, feature_names=X.columns, class_names=["Malignant", "Benign"],
          filled=True, rounded=True, ax=axes[0], fontsize=8)
axes[0].set_title(f"Sığ Ağaç (Doğruluk: {acc_shallow:.3f})")

importances = pd.Series(shallow.feature_importances_, index=X.columns).sort_values(ascending=False)
importances.head(10).plot(kind="barh", ax=axes[1], color="#E24B4A")
axes[1].set_title("En Önemli 10 Özellik")
axes[1].grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Supervised\\03-decision-tree\\01_breast_cancer.png", dpi=120)
plt.close()

depths = range(1, 15)
train_accs, test_accs = [], []
for d in depths:
    t = DecisionTreeClassifier(max_depth=d, random_state=42)
    t.fit(X_train, y_train)
    train_accs.append(accuracy_score(y_train, t.predict(X_train)))
    test_accs.append(accuracy_score(y_test, t.predict(X_test)))

plt.figure(figsize=(8, 5))
plt.plot(depths, train_accs, "o-", label="Eğitim", color="#3B8BD4")
plt.plot(depths, test_accs, "o-", label="Test", color="#E24B4A")
plt.xlabel("Derinlik")
plt.ylabel("Doğruluk")
plt.title("Derinlik vs Doğruluk (Overfitting Gösterimi)")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Supervised\\03-decision-tree\\02_derinlik_analiz.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — Sığ: {acc_shallow:.4f}, Derin: {acc_deep:.4f}")
print(f"{'='*70}")
