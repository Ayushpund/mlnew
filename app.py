from flask import Flask, request, jsonify
import pandas as pd
import pickle

# Load the trained model
model_file = 'overflow_prediction_model.pkl'  # Replace with your model's file name
try:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

# Initialize Flask app
app = Flask(__name__)

# Mapping for levels
level_mapping = {"Normal": 0, "Semi-Critical": 1, "Critical": 2}
reverse_mapping = {v: k for k, v in level_mapping.items()}

# Default overflow threshold (can be modified as needed)
DEFAULT_THRESHOLD = 100

@app.route('/')
def index():
    return "Welcome to the Artesian Well Overflow Prediction API! Use /predict endpoint for predictions."

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract rainfall data and threshold from the request
        rainfall = request.form.get('rainfall', type=float)
        threshold = request.form.get('threshold', type=float, default=DEFAULT_THRESHOLD)

        if rainfall is None:
            return jsonify({"error": "Rainfall value is required."}), 400

        # Validate rainfall value
        if rainfall < 0:
            return jsonify({"error": "Rainfall value must be a non-negative number."}), 400

        # Create input for prediction (use the same column name as in training)
        user_input_df = pd.DataFrame([[rainfall]], columns=["612.3"])  # Replace "612.3" with the column name used in training

        # Perform prediction
        prediction_numeric = model.predict(user_input_df)[0]
        predicted_level = reverse_mapping[prediction_numeric]

        # Prepare suggestions based on prediction
        suggestions = {
            "Critical": "Take immediate action to store water or divert it to avoid flooding.",
            "Semi-Critical": "Monitor closely and prepare for potential overflow.",
            "Normal": "Conditions are normal. No immediate action needed."
        }
        suggestion = suggestions.get(predicted_level, "No suggestion available.")

        # Return prediction result
        return jsonify({
            "rainfall": rainfall,
            "threshold": threshold,
            "predicted_level": predicted_level,
            "suggestion": suggestion
        })

    except KeyError as ke:
        return jsonify({"error": f"Key error occurred: {str(ke)}"}), 500
    except ValueError as ve:
        return jsonify({"error": f"Value error occurred: {str(ve)}"}), 500
    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
