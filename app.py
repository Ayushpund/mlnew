from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load the trained model
model_file = 'crop_prediction_model.pkl'
encoders_file = 'encoders.pkl'
try:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    with open(encoders_file, 'rb') as f:
        encoders = pickle.load(f)
    soil_encoder = encoders['soil_encoder']
    crop_encoder = encoders['crop_encoder']
    print("Model and encoders loaded successfully!")
except Exception as e:
    print(f"Error loading model or encoders: {e}")
    exit(1)

# Initialize Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return "Welcome to the Crop Prediction API! Use /predict endpoint to predict the crop type."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract input features from form data
        N = request.form.get('N', type=float)
        P = request.form.get('P', type=float)
        K = request.form.get('K', type=float)
        pH = request.form.get('pH', type=float)
        soil_type = request.form.get('Soil_type')

        # Validate inputs
        if N is None or P is None or K is None or pH is None or soil_type is None:
            return jsonify({"error": "All inputs (N, P, K, pH, Soil_type) are required."}), 400

        # Encode soil type
        try:
            soil_type_encoded = soil_encoder.transform([soil_type])[0]
        except ValueError:
            return jsonify({"error": "Invalid Soil_type provided."}), 400

        # Prepare input for prediction
        input_features = pd.DataFrame([[N, P, K, pH, soil_type_encoded]], columns=['N', 'P', 'K', 'pH', 'Soil_type'])

        # Perform prediction
        predicted_crop_numeric = model.predict(input_features)[0]
        predicted_crop = crop_encoder.inverse_transform([predicted_crop_numeric])[0]

        # Return prediction result
        return jsonify({
            "N": N,
            "P": P,
            "K": K,
            "pH": pH,
            "Soil_type": soil_type,
            "PredictedCrop": predicted_crop
        })

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
