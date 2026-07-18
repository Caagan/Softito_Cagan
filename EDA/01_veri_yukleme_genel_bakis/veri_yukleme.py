import os
import warnings
import kagglehub
import matplotlib.pyplot as plt
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


def load_dataset():
    dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith(".csv")]
    if not csv_files:
        raise FileNotFoundError("CSV dosyasi bulunamadi.")
    return pd.read_csv(os.path.join(dataset_path, csv_files[0]))


def main():
    print("=" * 70)
    print("  01 - VERI YUKLEME & GENEL BAKIS")
    print("=" * 70)

    df = load_dataset()
    print(f"\nVeri yuklendi: {df.shape[0]} satir, {df.shape[1]} sutun")

    print("\nIlk 5 satir:")
    print(df.head().to_string())

    print("\nSon 5 satir:")
    print(df.tail().to_string())

    print("\nRastgele 5 satir:")
    print(df.sample(5, random_state=42).to_string())

    dtype_df = pd.DataFrame({
        "sutun": df.columns,
        "tip": df.dtypes.values,
        "null": df.isnull().sum().values,
        "null_%": (df.isnull().sum().values / len(df) * 100).round(2),
        "unique": df.nunique().values,
    })
    print("\nSutun ozeti:")
    print(dtype_df.to_string(index=False))

    print("\nSayisal istatistikler:")
    print(df.describe().T.to_string())

    if df.select_dtypes(include=["object", "category"]).shape[1] > 0:
        print("\nKategorik ozet:")
        print(df.describe(include=["object", "category"]).T.to_string())

    plt.figure(figsize=(12, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap="viridis", yticklabels=False)
    plt.title("Eksik Deger Haritasi (Sari = Eksik)")
    save_fig("01_eksik_deger_haritasi.png")

    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=True)
    if len(missing) > 0:
        plt.figure(figsize=(10, 6))
        missing.plot(kind="barh", color="#E67E22")
        plt.title("Sutun Bazinda Eksik Deger Sayisi")
        plt.xlabel("Eksik Deger Sayisi")
        save_fig("01_sutun_eksik_deger_sayisi.png")
    else:
        print("\nEksik deger bulunmuyor.")


if __name__ == "__main__":
    main()
