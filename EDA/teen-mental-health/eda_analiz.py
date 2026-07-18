import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("=" * 70)
print("  EDA: Sosyal Medya Kullanımı ve Genç Ruh Sağlığı Analizi")
print("=" * 70)

print("\n[ADIM 1] Veri Setini Yükleme")
print("-" * 50)

try:
    import kagglehub
    path = kagglehub.dataset_download("m0hit/teen-mental-health-dataset")
    import os
    csv_files = [f for f in os.listdir(path) if f.endswith(".csv")]
    df = pd.read_csv(os.path.join(path, csv_files[0]))
    print(f"    Kaggle'dan yüklendi: {csv_files[0]}")
except Exception:
    print("    Kaggle bağlantısı kurulamadı, alternatif veri seti deneniyor...")
    try:
        df = pd.read_csv("https://raw.githubusercontent.com/datasets-br/mental-health/main/teen-mental-health.csv")
    except Exception:
        print("    İnternet bağlantısı yok, sentetik veri üretiliyor...")
        np.random.seed(42)
        NUM = 2000
        df = pd.DataFrame({
            "Age": np.random.randint(13, 19, NUM),
            "Gender": np.random.choice(["Male", "Female"], NUM),
            "Daily_Usage_Hours": np.random.exponential(4, NUM).clip(0.5, 12).round(1),
            "Sleep_Hours": np.random.normal(6.5, 1.2, NUM).clip(3, 10).round(1),
            "Stress_Level": np.random.randint(1, 11, NUM),
            "Anxiety_Score": np.random.normal(50, 15, NUM).clip(0, 100).round(1),
            "Academic_Performance": np.random.normal(3.0, 0.6, NUM).clip(1, 4).round(2),
            "Social_Interactions": np.random.poisson(5, NUM).clip(0, 20),
            "Screen_Time_Hours": np.random.exponential(6, NUM).clip(1, 16).round(1),
            "Exercise_Hours": np.random.exponential(3, NUM).clip(0, 15).round(1),
        })

print(f"    Veri boyutu: {df.shape}")
print(f"\n    Sütunlar: {list(df.columns)}")

print(f"\n[ADIM 2] Veri Keşfi")
print("-" * 50)
print(df.head())
print(f"\n    Veri tipleri:")
print(df.dtypes)
print(f"\n    Eksik değerler:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print(f"\n    Temel istatistikler:")
print(df.describe().round(2))

print(f"\n[ADIM 3] Veri Temizleme")
print("-" * 50)

oncesi = df.shape[0]
df = df.dropna(subset=df.columns[:5])
df = df.drop_duplicates()
sonrasi = df.shape[0]
print(f"    Temizlik: {oncesi} → {sonrasi} satır ({oncesi - sonrasi} satır silindi)")

sayisal = df.select_dtypes(include=[np.number]).columns.tolist()
print(f"    Sayısal sütunlar: {sayisal}")

print(f"\n[ADIM 4] Tek Değişkenli Analiz")
print("-" * 50)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle("Tek Değişkenli Analiz — Sayısal Değişkenler", fontsize=14)
for i, col in enumerate(sayisal[:6]):
    ax = axes[i // 3, i % 3]
    sns.histplot(df[col], bins=30, kde=True, ax=ax, color="#3B8BD4")
    ax.set_title(col)
plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\EDA\\teen-mental-health\\01_tek_degiskenli.png", dpi=120)
plt.close()

print(f"\n[ADIM 5] Korelasyon Analizi")
print("-" * 50)
corr = df[sayisal].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, square=True)
plt.title("Korelasyon Isı Haritası")
plt.tight_layout()
plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\EDA\\teen-mental-health\\02_korelasyon.png", dpi=120)
plt.close()
print(corr.round(2).to_string())

print(f"\n[ADIM 6] Çift Değişkenli Analiz")
print("-" * 50)

if "Gender" in df.columns:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Cinsiyete Göre Dağılımlar", fontsize=14)
    for i, col in enumerate(sayisal[:3]):
        sns.boxplot(data=df, x="Gender", y=col, ax=axes[i], palette="Set2")
        axes[i].set_title(col)
    plt.tight_layout()
    plt.savefig("c:\\Users\\cagan\\Downloads\\softitoprojelerim\\proje-reposu\\EDA\\teen-mental-health\\03_cinsiyet_analiz.png", dpi=120)
    plt.close()

print(f"\n[ADIM 7] Sonuç ve Bulgular")
print("-" * 50)

for col in sayisal[:4]:
    print(f"    {col:30s} — Ort: {df[col].mean():.2f}, Std: {df[col].std():.2f}")

print(f"\n{'='*70}")
print(f"  TAMAMLANDI")
print(f"{'='*70}")
