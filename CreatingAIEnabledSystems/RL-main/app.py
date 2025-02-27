from flask import Flask, request, jsonify
from model import EmailCampaign

app = Flask(__name__)
model = EmailCampaign(load_from='email_campaign_model')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    
    print(data)

    state = model.convert_to_state(data['gender'], data['type'], data['age'], data['tenure'])

    action = model.get_best_action_for_state(state)
    
    response = {
        'Action To Take': str(action),
    }
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)