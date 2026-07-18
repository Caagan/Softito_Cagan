import os
from flask import Flask, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    try:
        data_path = os.getenv("DATA_PATH", "/data/online_retail.csv")
        veri = pd.read_csv(data_path, encoding='ISO-8859-1')
        veri = veri.dropna(subset=['Quantity', 'UnitPrice'])
        X = veri[['Quantity']].head(1000)
        y = veri['UnitPrice'].head(1000)

        model = LinearRegression()
        model.fit(X, y)

        return jsonify({"status": "success", "service": "Sales Forecasting", "coef": float(model.coef_[0])})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5004))
    app.run(host='0.0.0.0', port=port)