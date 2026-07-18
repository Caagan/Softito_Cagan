import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  SVM — Breast Cancer Wisconsin (Linear vs RBF Kernel)")
print("=" * 70)

print("\n[PROJE] Meme Kanseri Hücre Tanımlama — SVM Karşılaştırması")
print("-" * 50)

data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

print(f"    Veri boyutu: {df.shape}")
print(f"    Sınıflar: {dict(pd.Series(data.target_names).value_counts())}")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

linear_svm = SVC(kernel="linear", C=1.0, random_state=42)
linear_svm.fit(X_train_scaled, y_train)
y_pred_linear = linear_svm.predict(X_test_scaled)
acc_linear = accuracy_score(y_test, y_pred_linear)

rbf_svm = SVC(kernel="rbf", C=10, gamma="scale", random_state=42)
rbf_svm.fit(X_train_scaled, y_train)
y_pred_rbf = rbf_svm.predict(X_test_scaled)
acc_rbf = accuracy_score(y_test, y_pred_rbf)

print(f"\n    Linear SVM Doğruluk   : {acc_linear:.4f}")
print(f"    RBF SVM Doğruluk      : {acc_rbf:.4f}")
print(f"\n    Linear SVM Raporu:")
print(classification_report(y_test, y_pred_linear, target_names=data.target_names))

pca = PCA(n_components=2)
X_train_2d = pca.fit_transform(X_train_scaled)
svm_2d = SVC(kernel="linear", C=1.0, random_state=42)
svm_2d.fit(X_train_2d, y_train)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("SVM — Breast Cancer Sınıflandırması", fontsize=14)

xx, yy = np.meshgrid(
    np.linspace(X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1, 200),
    np.linspace(X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1, 200)
)
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
axes[0].contourf(xx, yy, Z, alpha=0.3, cmap="RdYlBu")
axes[0].scatter(X_train_2d[y_train == 0, 0], X_train_2d[y_train == 0, 1], c="#E24B4A", label="Malignant", alpha=0.5, s=15)
axes[0].scatter(X_train_2d[y_train == 1, 0], X_train_2d[y_train == 1, 1], c="#1D9E75", label="Benign", alpha=0.5, s=15)
axes[0].set_xlabel("PC1")
axes[0].set_ylabel("PC2")
axes[0].set_title("SVM Karar Sınırı (PCA ile 2D)")
axes[0].legend()
axes[0].grid(alpha=0.3)

ConfusionMatrixDisplay.from_predictions(y_test, y_pred_linear, display_labels=data.target_names, cmap="Greens", ax=axes[1])
axes[1].set_title("Linear SVM Confusion Matrix")

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Supervised\\06-svm\\01_svm_breast_cancer.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — Linear: {acc_linear:.4f}, RBF: {acc_rbf:.4f}")
print(f"{'='*70}")
