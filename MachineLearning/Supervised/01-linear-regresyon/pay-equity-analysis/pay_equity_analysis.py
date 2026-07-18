import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  PAY EQUITY ANALYSIS — Cinsiyete Dayali Ucret Farki Tespiti")
print("=" * 70)

print("\n[PROJE] California Housing — Emlak Fiyat Tahmini (Cinsiyet Etkisi)")
print("-" * 50)

housing = fetch_california_housing()
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df["price"] = housing.target

np.random.seed(42)
df["gender"] = np.random.choice(["Erkek", "Kadin"], size=len(df), p=[0.55, 0.45])
df.loc[df["gender"] == "Kadin", "price"] *= np.random.uniform(0.85, 1.0, size=(df["gender"] == "Kadin").sum())

print(f"    Veri boyutu: {df.shape}")
print(f"    Cinsiyet dagilimi: {df['gender'].value_counts().to_dict()}")

X = df.drop(["price", "gender"], axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\n    Model Performansi:")
print(f"    RMSE : {rmse:.4f}")
print(f"    MAE  : {mae:.4f}")
print(f"    R2   : {r2:.4f}")

df_test = df.iloc[y_test.index].copy()
df_test["predicted"] = y_pred

print(f"\n    Cinsiyet Bazli Fiyat Analizi:")
for gender in ["Erkek", "Kadin"]:
    alt = df_test[df_test["gender"] == gender]
    print(f"    {gender:<8}: gercek ort={alt['price'].mean():.3f}, tahmin ort={alt['predicted'].mean():.3f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Pay Equity Analysis — Cinsiyet ve Fiyat Iliskisi", fontsize=14)

for gender, renk in zip(["Erkek", "Kadin"], ["#3B8BD4", "#E24B4A"]):
    mask = df_test["gender"] == gender
    axes[0].scatter(df_test[mask]["price"], df_test[mask]["predicted"], alpha=0.3, s=10, label=gender, color=renk)
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "k--", linewidth=2)
axes[0].set_xlabel("Gercek Fiyat")
axes[0].set_ylabel("Tahmin Fiyat")
axes[0].set_title("Gercek vs Tahmin (Cinsiyet Bazinda)")
axes[0].legend()
axes[0].grid(alpha=0.3)

importances = pd.Series(model.coef_, index=housing.feature_names).sort_values()
importances.plot(kind="barh", ax=axes[1], color="#1D9E75")
axes[1].set_title("Ozellik Katsayilari (Linear Reg.)")

for gender, renk in zip(["Erkek", "Kadin"], ["#3B8BD4", "#E24B4A"]):
    alt = df_test[df_test["gender"] == gender]
    axes[2].hist(alt["price"], bins=30, alpha=0.5, label=gender, color=renk, density=True)
axes[2].set_title("Fiyat Dagilimi (Cinsiyet Bazinda)")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\MachineLearning\Supervised\01-linear-regresyon\pay-equity-analysis\01_pay_equity.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI — R2={r2:.4f}")
print(f"{'='*70}")
