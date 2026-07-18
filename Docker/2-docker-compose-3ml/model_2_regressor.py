import time
import pandas as pd
from sklearn.linear_model import LinearRegression
import psycopg2

print("Regressor Servisi: Başlıyor...")
time.sleep(7) 

try:
    veri = pd.read_csv('churn.csv')
    X = veri[['tenure']] 
    y = veri['MonthlyCharges']
except Exception as e:
    print(f"Veri okuma hatası: {e}"); exit(1)

model = LinearRegression()
model.fit(X, y)
veri['tahmin_fatura'] = model.predict(X)

try:
    conn = psycopg2.connect(host="db_servisi", database="projem_db", user="kullanici", password="sifre123")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS regressor_sonuclar (id SERIAL, tenure REAL, tahmin_fatura REAL);")
    
    for _, row in veri.head(50).iterrows():
        cursor.execute("INSERT INTO regressor_sonuclar (tenure, tahmin_fatura) VALUES (%s, %s)",
                       (float(row['tenure']), float(row['tahmin_fatura'])))
    conn.commit()
    print("[BAŞARILI] Regressor sonuçları DB'ye yazıldı!"); cursor.close(); conn.close()
except Exception as e:
    print(f"DB Hatası: {e}")