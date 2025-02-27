from flask import Flask, request, jsonify
from model import Fraud_Detector_Model
from werkzeug.utils import secure_filename
import pandas as pd
import os

app = Flask(__name__)
model = Fraud_Detector_Model() 

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads/', filename)
        file.save(file_path)
        
        # Pass the file path to the model's predict method
        prediction = model.predict(file_path)
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        return jsonify({'prediction': prediction})
    
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)