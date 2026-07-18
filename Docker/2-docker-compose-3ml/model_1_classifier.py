import time
import pandas as pd
from sklearn.linear_model import LogisticRegression
import psycopg2

print("Classifier Servisi: Başlıyor...")
time.sleep(5)

try:
    veri = pd.read_csv('churn.csv')

    X = veri[['MonthlyCharges', 'tenure']]
    y = veri['Churn']
except Exception as e:
    print(f"Veri okunurken hata çıktı: {e}"); exit(1)

model = LogisticRegression()
model.fit(X, y)
veri['tahmin_churn'] = model.predict(X)

try:
    conn = psycopg2.connect(host="db_servisi", database="projem_db", user="kullanici", password="sifre123")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS classifier_sonuclar (id SERIAL, monthly REAL, tenure REAL, tahmin INT);")
    
    for _, row in veri.head(50).iterrows():
        cursor.execute("INSERT INTO classifier_sonuclar (monthly, tenure, tahmin) VALUES (%s, %s, %s)",
                       (float(row['MonthlyCharges']), float(row['tenure']), int(row['tahmin_churn'])))
    conn.commit()
    print("[BAŞARILI] Classifier sonuçları DB'ye yazıldı!"); cursor.close(); conn.close()
except Exception as e:
    print(f"DB Hatası: {e}")