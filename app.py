
from flask import Flask, request, jsonify
import numpy as np
import pickle as pk

# Load the trained model
with open('Random_Forest_Classifier_100%.h5', 'rb') as file:
    loaded_model = pk.load(file)

# Function to determine possible diseases based on features
def determine_disease(inputs):
    diseases = []
    if inputs[2] >= 1 and inputs[8] == 1 and inputs[9] > 1.0:
        diseases.append("Heart Attack")
    if inputs[2] >= 1 and inputs[8] == 1 and inputs[10] <= 1:
        diseases.append("Angina")
    if inputs[11] >= 1 and inputs[3] > 140 and inputs[4] > 200:
        diseases.append("Coronary Artery Disease")
    if inputs[0] > 50 and inputs[3] > 140 and inputs[4] > 200:
        diseases.append("Peripheral Artery Disease")
    if inputs[0] > 50 and inputs[4] > 200 and inputs[3] > 140:
        diseases.append("Carotid Artery Disease")
    return diseases

# Prediction function
def predict(inputs):
    input_data = np.array([inputs])
    prediction = loaded_model.predict(input_data)
    if prediction[0] == 1:  # Positive for heart disease
        diseases = determine_disease(inputs)
        if diseases:
            result = f"Heart disease likelihood: Positive\nPossible conditions:\n" + "\n".join(diseases)
        else:
            result = "Heart disease likelihood: Positive\nNo specific condition identified based on the given features."
    else:
        result = "Heart disease likelihood: Negative"
    return result

# Flask app setup
app = Flask(__name__)

@app.route('/')
def home():
    return "Heart Disease Prediction API is running."

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
        data = request.json
        inputs = [
            data.get('age'), data.get('sex'), data.get('cp'),
            data.get('trestbps'), data.get('chol'), data.get('fbs'),
            data.get('restecg'), data.get('thalach'), data.get('exang'),
            data.get('oldpeak'), data.get('slope'), data.get('ca'), data.get('thal')
        ]
        inputs = list(map(float, inputs))
        result = predict(inputs)
        return jsonify({"result": result})
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
