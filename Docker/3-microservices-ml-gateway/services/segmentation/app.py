import os
from flask import Flask, jsonify
import pandas as pd
from sklearn.ensemble import IsolationForest

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        data_path = os.getenv("DATA_PATH", "/data/online_retail.csv")
        veri = pd.read_csv(data_path, encoding='ISO-8859-1')
        veri = veri.dropna(subset=['Quantity', 'UnitPrice'])
        X = veri[['Quantity', 'UnitPrice']].head(1000)

        model = IsolationForest(contamination=0.02, random_state=42)
        anomaliler = model.fit_predict(X)
        anomali_sayisi = int((anomaliler == -1).sum())

        return jsonify({"status": "success", "service": "Customer Segmentation (Isolation Forest)", "anomalies_found": anomali_sayisi})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5005))
    app.run(host='0.0.0.0', port=port)