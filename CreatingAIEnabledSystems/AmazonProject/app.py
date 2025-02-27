from flask import Flask, request, jsonify
from model import Amazon_Model

app = Flask(__name__)
model = Amazon_Model()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    prediction, raw_pred = model.predict(data)
    
    response = {
        'Polarity': str(prediction),
        'Raw_polarity': str(raw_pred)
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)