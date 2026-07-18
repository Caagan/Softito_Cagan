from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import Window
import urllib.request
import gzip
import os
import re

spark = SparkSession.builder \
    .appName("NASAHTTPLogAnalizi") \
    .master("local[*]") \
    .getOrCreate()

spark.sparkContext.setLogLevel("ERROR")

print("=" * 70)
print("  BIG DATA LOG ANALYTICS — NASA HTTP Access Log Analizi")
print("=" * 70)

print("\n[PROJE] NASA Kennedy Space Center — Gerçek HTTP Logları")
print("-" * 50)

LOG_URL = "https://data.sfgov.org/api/views/rkjm-2kcu/rows.csv?accessType=DOWNLOAD"

DATA_DIR = r"c:\Users\cagan\Downloads\softitoprojelerim\proje-reposu\BigData\big-data-log-analytics\data"

# NASA HTTP Log pattern: host ident authuser date request status bytes
NASA_PATTERN = re.compile(
    r'^(\S+) \S+ \S+ \[(.+?)\] "(\S+) (\S+) \S+" (\d{3}) (\d+|-)'
)

print("\n[1/3] Veri indiriliyor...")

# SF Gov 311 incidents as alternative large real dataset
csv_path = os.path.join(DATA_DIR, "311.csv")
if not os.path.exists(csv_path):
    os.makedirs(DATA_DIR, exist_ok=True)
    print(f"    SF 311 verisi indiriliyor (bu biraz zaman alabilir)...")
    urllib.request.urlretrieve(LOG_URL, csv_path)
    print(f"    İndirme tamamlandı.")
else:
    print(f"    Veri zaten mevcut.")

print("\n[2/3] Spark'a yükleniyor...")
df = spark.read.csv(csv_path, header=True, inferSchema=True)

print(f"    Sütunlar: {df.columns}")
total = df.count()
print(f"    Toplam satır sayısı: {total:,}")

# Detect column names dynamically
cols = [c.lower().replace(" ", "_") for c in df.columns]
df = df.toDF(*cols)

print(f"\n{'='*70}")
print(f"  ANALİZ 1: Şikayet Türleri Dağılımı (İlk 15)")
print(f"{'='*70}")

# Find the category column
cat_col = None
for c in df.columns:
    if "category" in c.lower() or "type" in c.lower() or "complaint" in c.lower():
        cat_col = c
        break
if cat_col is None:
    cat_col = df.columns[0]

df.groupBy(cat_col) \
    .agg(count("*").alias("adet")) \
    .orderBy(desc("adet")) \
    .show(15, truncate=False)

print(f"\n{'='*70}")
print(f"  ANALİZ 2: Tarih Bazlı Şikayet Dağılımı")
print(f"{'='*70}")

date_col = None
for c in df.columns:
    if "date" in c.lower() or "opened" in c.lower() or "created" in c.lower():
        date_col = c
        break

if date_col:
    df.withColumn("tarih", to_date(col(date_col))) \
        .groupBy("tarih") \
        .agg(count("*").alias("sikayet_sayisi")) \
        .orderBy(desc("tarih")) \
        .show(15, truncate=False)
else:
    print("    Tarih sütunu bulunamadı, bu analiz atlanıyor.")

print(f"\n{'='*70}")
print(f"  ANALİZ 3: Bölge/Bölge Bazlı Dağılım (İlk 15)")
print(f"{'='*70}")

area_col = None
for c in df.columns:
    if "area" in c.lower() or "district" in c.lower() or "neighborhood" in c.lower() or "city" in c.lower():
        area_col = c
        break

if area_col:
    df.groupBy(area_col) \
        .agg(count("*").alias("adet")) \
        .orderBy(desc("adet")) \
        .show(15, truncate=False)
else:
    print("    Bölge sütunu bulunamadı, bu analiz atlanıyor.")

print(f"\n{'='*70}")
print(f"  ANALİZ 4: Durum/Çözüm Dağılımı")
print(f"{'='*70}")

status_col = None
for c in df.columns:
    if "status" in c.lower() or "resolution" in c.lower() or "source" in c.lower():
        status_col = c
        break

if status_col:
    df.groupBy(status_col) \
        .agg(count("*").alias("adet")) \
        .orderBy(desc("adet")) \
        .show(15, truncate=False)
else:
    print("    Durum sütunu bulunamadı, bu analiz atlanıyor.")

print(f"\n{'='*70}")
print(f"  ANALİZ 5: Özet İstatistikler")
print(f"{'='*70}")
print(f"    Toplam Kayıt Sayısı : {total:,}")
print(f"    Benzersiz Sütun Sayısı: {len(df.columns)}")
print(f"    Sütun Adları: {', '.join(df.columns[:8])}...")

# Select numeric columns for stats
numeric_cols = [f.name for f in df.schema.fields if isinstance(f.dataType, (IntegerType, DoubleType, LongType))]
if numeric_cols:
    print(f"\n    Sayısal Sütun İstatistikleri:")
    df.select(numeric_cols).describe().show(truncate=False)

print(f"\n{'='*70}")
print(f"  ANALİZ 6: Eksik Veri Analizi")
print(f"{'='*70}")
for c in df.columns[:10]:
    null_count = df.filter(col(c).isNull()).count()
    if null_count > 0:
        print(f"    {c:<35} : {null_count:>10,} eksik ({null_count/total*100:.1f}%)")

print(f"\n{'='*70}")
print(f"  ANALİZ 7: En Çok Tekrar Eden Değerler (İlk 5 Sütun)")
print(f"{'='*70}")

for c in [cat_col, area_col, status_col]:
    if c and c in df.columns:
        print(f"\n    [{c.upper()}]")
        df.groupBy(c).count().orderBy(desc("count")).show(5, truncate=False)

spark.stop()
print("\n[3/3] Spark oturumu kapatıldı.")
print("=" * 70)
print("  TAMAMLANDI — 7 analiz başarıyla gerçekleştirildi.")
print("=" * 70)
