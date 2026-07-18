import os
from flask import Flask, jsonify
import pandas as pd
from sklearn.linear_model import LogisticRegression

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        data_path = os.getenv("DATA_PATH", "/data/online_retail.csv")
        veri = pd.read_csv(data_path, encoding='ISO-8859-1')
        veri = veri.dropna(subset=['Quantity', 'UnitPrice'])
        X = veri[['Quantity', 'UnitPrice']].head(1000)
        y = (X['Quantity'] > 2).astype(int)

        model = LogisticRegression()
        model.fit(X, y)
        accuracy = model.score(X, y)

        return jsonify({"status": "success", "service": "Churn Prediction", "accuracy": float(accuracy)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5002))
    app.run(host='0.0.0.0', port=port)