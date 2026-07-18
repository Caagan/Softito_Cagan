import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  SUPER LIG GOAL TAHMINI — Regresyon Analizi")
print("=" * 70)

print("\n[PROJE] Futbol Istatistiklerinden Gol Sayisi Tahmini")
print("-" * 50)

np.random.seed(42)
n = 200
df = pd.DataFrame({
    "topla_sut": np.random.poisson(12, n),
    "isabetli_sut": np.random.poisson(5, n),
    "top": np.random.uniform(40, 65, n),
    "pas_basari": np.random.uniform(70, 92, n),
    "kaleyi_bulma_orani": np.random.uniform(0.25, 0.55, n),
    "serit_atisi": np.random.poisson(4, n),
    "korner": np.random.poisson(6, n),
})

df["gol"] = (
    0.3 * df["isabetli_sut"] +
    2.0 * df["kaleyi_bulma_orani"] +
    0.05 * df["top"] +
    0.02 * df["pas_basari"] +
    0.15 * df["serit_atisi"] +
    np.random.normal(0, 0.8, n)
).clip(0, None).astype(int)

print(f"    Veri boyutu: {df.shape}")
print(f"    Gol istatistikleri: ort={df['gol'].mean():.2f}, std={df['gol'].std():.2f}")

X = df.drop("gol", axis=1)
y = df["gol"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

models = {
    "LinearReg": LinearRegression(),
    "Ridge": Ridge(alpha=1.0),
    "Lasso": Lasso(alpha=0.1),
}

print(f"\n    {'Model':<15} {'RMSE':>8} {'R2':>8} {'CV-R2':>8}")
print(f"    {'─'*42}")

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    cv = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring="r2").mean()
    print(f"    {name:<15} {rmse:>8.4f} {r2:>8.4f} {cv:>8.4f}")

best_model = Ridge(alpha=1.0)
best_model.fit(X_train_scaled, y_train)
y_pred_best = best_model.predict(X_test_scaled)

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Super Lig Gol Tahmini — Regresyon Analizi", fontsize=14)

axes[0].scatter(y_test, y_pred_best, alpha=0.5, s=20, color="#3B8BD4")
axes[0].plot([y.min(), y.max()], [y.min(), y.max()], "r--", linewidth=2)
axes[0].set_xlabel("Gercek Gol")
axes[0].set_ylabel("Tahmin Gol")
axes[0].set_title("Gercek vs Tahmin (Ridge)")
axes[0].grid(alpha=0.3)

importances = pd.Series(best_model.coef_, index=X.columns).sort_values()
importances.plot(kind="barh", ax=axes[1], color="#1D9E75")
axes[1].set_title("Ozellik Onemleri (Ridge)")

residuals = y_test.values - y_pred_best
axes[2].scatter(y_pred_best, residuals, alpha=0.5, s=20, color="#E24B4A")
axes[2].axhline(0, color="black", linestyle="--")
axes[2].set_xlabel("Tahmin")
axes[2].set_ylabel("Artik")
axes[2].set_title("Artik Analizi")
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\MachineLearning\Supervised\01-linear-regresyon\superlig-goal-prediction\01_goal_prediction.png", dpi=120)
plt.close()

print(f"\n{'='*70}")
print(f"  TAMAMLANDI")
print(f"{'='*70}")
