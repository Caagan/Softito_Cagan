import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  LINEAR REGRESYON — California Evi Fiyat Tahmini")
print("=" * 70)

print("\n[PROJE] California Housing Veri Seti")
print("-" * 50)

from sklearn.datasets import fetch_california_housing
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["MedHouseVal"] = data.target

print(f"    Veri boyutu: {df.shape}")
print(f"    Hedef: MedHouseVal (Ortalama ev değeri, 100K$ cinsinden)")
print(f"\n    İlk 5 satır:")
print(df.head())

X = df.drop("MedHouseVal", axis=1)
y = df["MedHouseVal"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)

print(f"\n    R² Skoru              : {r2:.4f}")
print(f"    RMSE                  : {rmse:.4f} (100K$)")
print(f"    MAE                   : {mae:.4f} (100K$)")

print(f"\n    Katsayılar:")
for name, coef in zip(X.columns, model.coef_):
    print(f"      {name:20s} : {coef:>10.4f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Linear Regresyon — California Evi Fiyat Tahmini", fontsize=14)

axes[0].scatter(y_test, y_pred, alpha=0.3, color="#3B8BD4", s=10)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--", linewidth=2)
axes[0].set_xlabel("Gerçek Fiyat (100K$)")
axes[0].set_ylabel("Tahmin (100K$)")
axes[0].set_title(f"Gerçek vs Tahmin (R²={r2:.3f})")
axes[0].grid(alpha=0.3)

residuals = y_test - y_pred
axes[1].hist(residuals, bins=50, color="#E24B4A", edgecolor="white")
axes[1].set_title("Artık Dağılımı")
axes[1].axvline(0, color="black", linestyle="--")
axes[1].grid(alpha=0.3)

importances = pd.Series(model.coef_, index=X.columns).sort_values()
importances.plot(kind="barh", ax=axes[2], color="#1D9E75")
axes[2].set_title("Özellik Önemi")
axes[2].grid(alpha=0.3, axis="x")

plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\MachineLearning\\Supervised\\01-linear-regresyon\\01_california_fiyat.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — R²={r2:.4f}, RMSE={rmse:.4f}")
print(f"{'='*70}")
