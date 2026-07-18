import os
import requests
from flask import Flask, jsonify

app = Flask(__name__)

CHURN_URL = os.getenv("CHURN_URL", "http://ml-churn:5002")
FORECAST_URL = os.getenv("FORECAST_URL", "http://ml-forecast:5004")
SEGMENTATION_URL = os.getenv("SEGMENTATION_URL", "http://ml-segmentation:5005")

@app.route('/api/churn', methods=['GET'])
def route_churn():
    try:
        r = requests.get(f"{CHURN_URL}/predict")
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/forecast', methods=['GET'])
def route_forecast():
    try:
        r = requests.get(f"{FORECAST_URL}/predict")
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/segmentation', methods=['GET'])
def route_segmentation():
    try:
        r = requests.get(f"{SEGMENTATION_URL}/predict")
        return jsonify(r.json()), r.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port)