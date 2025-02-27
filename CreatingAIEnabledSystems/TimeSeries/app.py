from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
from model import Model

app = Flask(__name__)

deep_model = Model()

@app.route('/predict', methods=['POST'])
def predict():
    if request.get_json():
        data = request.get_json()
        date_str = str(data['date']) 
    elif request.files['file']:
        date_str = file.date
     
    
    prediction, fraud_prediction = deep_model.prediction(date_str)
    
    response = {
        'total_transactions': str(prediction),
        'fraudulent_transactions': str(fraud_prediction)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)