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
    print("  05 - FEATURE ENGINEERING")
    print("=" * 70)

    df = load_and_clean()
    print(f"Veri hazir: {df.shape}")

    df_fe = df.copy()
    yeni_ozellikler = []

    if "age" in df_fe.columns:
        df_fe["age_group"] = pd.cut(
            df_fe["age"], bins=[12, 14, 16, 19],
            labels=["13-14", "15-16", "17-19"]
        )
        yeni_ozellikler.append("age_group")
        print("  age_group turetildi (13-14 / 15-16 / 17-19)")

    if "daily_social_media_hours" in df_fe.columns and "sleep_hours" in df_fe.columns:
        df_fe["screen_sleep_ratio"] = (
            df_fe["daily_social_media_hours"] / df_fe["sleep_hours"].replace(0, np.nan)
        ).round(2)
        yeni_ozellikler.append("screen_sleep_ratio")
        print("  screen_sleep_ratio turetildi (sosyal medya / uyku)")

    risk_kolonlari = [c for c in ["stress_level", "anxiety_level", "addiction_level"]
                       if c in df_fe.columns]
    if len(risk_kolonlari) >= 2:
        df_fe["risk_score"] = df_fe[risk_kolonlari].sum(axis=1)
        df_fe["risk_category"] = pd.cut(
            df_fe["risk_score"],
            bins=[-1, df_fe["risk_score"].quantile(0.33),
                  df_fe["risk_score"].quantile(0.66), df_fe["risk_score"].max()],
            labels=["Dusuk", "Orta", "Yuksek"]
        )
        yeni_ozellikler.extend(["risk_score", "risk_category"])
        print(f"  risk_score / risk_category turetildi ({' + '.join(risk_kolonlari)})")

    cat_cols = df_fe.select_dtypes(include=["object", "category"]).columns
    encode_col = next((c for c in cat_cols if c.lower() in ("gender", "cinsiyet")), None)
    if encode_col:
        df_fe[f"{encode_col}_encoded"] = df_fe[encode_col].astype("category").cat.codes
        yeni_ozellikler.append(f"{encode_col}_encoded")
        print(f"  {encode_col}_encoded turetildi (label encoding)")

    if "academic_performance" in df_fe.columns and "daily_social_media_hours" in df_fe.columns:
        df_fe["social_academic_ratio"] = (
            df_fe["daily_social_media_hours"] / df_fe["academic_performance"].replace(0, np.nan)
        ).round(2)
        yeni_ozellikler.append("social_academic_ratio")
        print("  social_academic_ratio turetildi (sosyal medya / akademik)")

    print(f"\nUretilen ozellikler: {yeni_ozellikler}")

    yeni_sayisal = [c for c in yeni_ozellikler
                     if c in df_fe.columns and df_fe[c].dtype in ("int64", "float64")]
    if yeni_sayisal:
        fig, axes = plt.subplots(1, len(yeni_sayisal), figsize=(7 * len(yeni_sayisal), 5))
        axes = np.array([axes]) if len(yeni_sayisal) == 1 else axes
        for i, col in enumerate(yeni_sayisal):
            sns.histplot(df_fe[col].dropna(), kde=True, ax=axes[i], color="#9B59B6")
            axes[i].set_title(f"Yeni Ozellik: {col}")
        save_fig("05_yeni_ozellik_dagilimlari.png")

    if "risk_category" in df_fe.columns:
        plt.figure(figsize=(8, 6))
        sns.countplot(data=df_fe, x="risk_category",
                       order=["Dusuk", "Orta", "Yuksek"], palette="Reds")
        plt.title("Risk Kategorisi Dagilimi")
        save_fig("05_risk_kategorisi_dagilimi.png")

    df_fe.to_csv("processed_data.csv", index=False)
    print("\nIslenmis veri kaydedildi: processed_data.csv")


if __name__ == "__main__":
    main()
