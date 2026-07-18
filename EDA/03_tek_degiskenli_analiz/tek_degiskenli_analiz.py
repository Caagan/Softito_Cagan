import os
import warnings
import kagglehub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 50)
sns.set_style("whitegrid")

KAGGLE_DATASET = "argonnxx/teen-mental-health"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


def save_fig(filename):
    path = os.path.join(FIGURES_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  -> kaydedildi: {path}")


def load_and_clean():
    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    df = pd.read_csv(os.path.join(dataset_path, csv_files[0]))
    df = df.drop_duplicates()
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def main():
    print("=" * 70)
    print("  03 - TEK DEGISKENLI ANALIZ")
    print("=" * 70)

    df = load_and_clean()
    print(f"Veri hazir: {df.shape}")

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns

    if len(num_cols) > 0:
        n = len(num_cols)
        cols_per_row = 3
        rows = int(np.ceil(n / cols_per_row))
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(18, 5 * rows))
        axes = np.array(axes).reshape(-1)

        for i, col in enumerate(num_cols):
            sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#2ECC71")
            skew = df[col].skew()
            axes[i].set_title(f"{col} Dagilimi (Carpiklik: {skew:.2f})")

        for j in range(len(num_cols), len(axes)):
            axes[j].axis("off")

        save_fig("03_sayisal_degisken_dagilimlari.png")

        print("\nSayisal degisken detaylari:")
        for col in num_cols:
            v = df[col].dropna()
            print(f"  {col}: ort={v.mean():.2f} med={v.median():.2f} std={v.std():.2f} min={v.min():.2f} max={v.max():.2f}")

    if len(cat_cols) > 0:
        n = len(cat_cols)
        cols_per_row = 3
        rows = int(np.ceil(n / cols_per_row))
        fig, axes = plt.subplots(rows, cols_per_row, figsize=(18, 5 * rows))
        axes = np.array(axes).reshape(-1)

        renkler = ["#88CCEE", "#CC6677", "#DDCC77", "#117733", "#332288", "#AA4499"]

        for i, col in enumerate(cat_cols):
            degerler = df[col].value_counts().head(10)
            yuzde = (degerler / len(df) * 100).round(1)
            bars = axes[i].bar(degerler.index.astype(str), degerler.values, color=renkler[:len(degerler)])
            axes[i].set_title(f"{col} Dagilimi")
            axes[i].tick_params(axis="x", rotation=45)
            for bar, pct in zip(bars, yuzde):
                axes[i].text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                             f"%{pct}", ha="center", va="bottom", fontsize=9)

        for j in range(len(cat_cols), len(axes)):
            axes[j].axis("off")

        save_fig("03_kategorik_degisken_dagilimlari.png")


if __name__ == "__main__":
    main()
