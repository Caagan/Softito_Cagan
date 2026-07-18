import time
import pandas as pd
from sklearn.ensemble import IsolationForest
import psycopg2

print("ML Servisi: Veritabanının ayağa kalkması bekleniyor...")
time.sleep(5)

try:
    
    veri = pd.read_csv('creditcard.csv') 
    print("[BAŞARILI] Kaggle Kredi Kartı veri seti okundu.")
except Exception as e:
    print(f"[HATA] Veri seti okunamadı: {e}")
    exit(1)

# Creditcard.csv içindeki en popüler iki sayısal sütun: 'Amount' (Tutar) ve 'V1'
sutunlar = ['Amount', 'V1'] 
veri = veri.dropna(subset=sutunlar)

# ML Model Eğitimi (Dolandırıcılık / Anomali Tespiti)
print("Model eğitiliyor, bu işlem verinin boyutuna göre biraz sürebilir...")
model = IsolationForest(contamination=0.01, random_state=42) # Kontaminasyonu %1 yaptık (çünkü fraud oranı düşüktür)
veri['anomali_sonucu'] = model.fit_predict(veri[sutunlar])

print("\n--- ML MODEL SONUÇLARI (İLK 5 SATIR) ---")
print(veri[['Amount', 'V1', 'anomali_sonucu']].head())

# Veritabanına Bağlantı ve Kayıt
try:
    conn = psycopg2.connect(
        host="db_servisi",
        database="projem_db",
        user="kullanici",
        password="sifre123"
    )
    cursor = conn.cursor()
    
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS kaggle_ml_sonuclar (
            id SERIAL PRIMARY KEY,
            deger_1 REAL,
            deger_2 REAL,
            sonuc INT
        );
    """)
    
    # creditcard.csv çok büyük (284k satır) olduğu için sadece ilk 100 satırı DB'ye yazıyoruz
    for _, row in veri.head(100).iterrows():
        cursor.execute(
            "INSERT INTO kaggle_ml_sonuclar (deger_1, deger_2, sonuc) VALUES (%s, %s, %s)",
            (float(row['Amount']), float(row['V1']), int(row['anomali_sonucu']))
        )
    
    conn.commit()
    print("\n[BAŞARILI] Şüpheli harcama analiz sonuçları PostgreSQL veritabanına kaydedildi!")
    
    cursor.close()
    conn.close()

except Exception as e:
    print(f"[HATA] Veritabanı bağlantı hatası: {e}")