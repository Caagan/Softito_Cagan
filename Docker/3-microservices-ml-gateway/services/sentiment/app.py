import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    return jsonify({
        "status": "success", 
        "service": "Sentiment Analysis", 
        "message": "Müşteri yorum duyarlılığı: %84 Olumlu"
    })

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5001))
    app.run(host='0.0.0.0', port=port)