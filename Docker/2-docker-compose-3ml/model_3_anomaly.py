import time
import pandas as pd
from sklearn.ensemble import IsolationForest
import psycopg2

print("Anomaly Servisi: Başlıyor...")
time.sleep(9)

try:
    veri = pd.read_csv('churn.csv')
    sutunlar = ['MonthlyCharges', 'tenure']
except Exception as e:
    print(f"Veri okuma hatası: {e}"); exit(1)

model = IsolationForest(contamination=0.02, random_state=42)
veri['anomali'] = model.fit_predict(veri[sutunlar])

try:
    conn = psycopg2.connect(host="db_servisi", database="projem_db", user="kullanici", password="sifre123")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS anomali_sonuclar (id SERIAL, monthly REAL, tenure REAL, sonuc INT);")
    
    for _, row in veri.head(50).iterrows():
        cursor.execute("INSERT INTO anomali_sonuclar (monthly, tenure, sonuc) VALUES (%s, %s, %s)",
                       (float(row['MonthlyCharges']), float(row['tenure']), int(row['anomali'])))
    conn.commit()
    print("[BAŞARILI] Anomali sonuçları DB'ye yazıldı!"); cursor.close(); conn.close()
except Exception as e:
    print(f"DB Hatası: {e}")