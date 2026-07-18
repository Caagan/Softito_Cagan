import os
from flask import Flask, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict():
    return jsonify({
        "status": "success", 
        "service": "Product Recommendation", 
        "recommended_products": ["WHITE HANGING HEART T-LIGHT HOLDER", "REGENCY CAKESTAND 3 TIER"]
    })

if __name__ == '__main__':
    port = int(os.getenv("PORT", 5003))
    app.run(host='0.0.0.0', port=port)