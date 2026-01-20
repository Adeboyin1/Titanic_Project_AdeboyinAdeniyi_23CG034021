from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Load the saved model and preprocessing objects
MODEL_PATH = 'model/titanic_survival_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
FEATURES_PATH = 'model/feature_names.pkl'

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    print("Model and preprocessing objects loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None
    scaler = None
    feature_names = None

@app.route('/')
def home():
    """Render the home page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        if model is None or scaler is None:
            return jsonify({
                'error': 'Model not loaded. Please ensure model files are in the correct location.'
            }), 500
        
        # Get data from form
        pclass = int(request.form['pclass'])
        sex = int(request.form['sex'])  # 0=female, 1=male
        age = float(request.form['age'])
        fare = float(request.form['fare'])
        embarked = int(request.form['embarked'])  # 0=C, 1=Q, 2=S
        
        # Validate inputs
        if not (1 <= pclass <= 3):
            return jsonify({'error': 'Pclass must be between 1 and 3'}), 400
        if age < 0 or age > 120:
            return jsonify({'error': 'Age must be between 0 and 120'}), 400
        if fare < 0:
            return jsonify({'error': 'Fare must be non-negative'}), 400
        
        # Create input dataframe
        input_data = pd.DataFrame({
            'Pclass': [pclass],
            'Sex': [sex],
            'Age': [age],
            'Fare': [fare],
            'Embarked': [embarked]
        })
        
        # Scale the input
        input_scaled = scaler.transform(input_data)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0]
        
        # Prepare response
        result = {
            'prediction': int(prediction),
            'survival_status': 'Survived' if prediction == 1 else 'Did Not Survive',
            'probability_survived': round(float(probability[1]) * 100, 2),
            'probability_not_survived': round(float(probability[0]) * 100, 2),
            'input_data': {
                'Passenger Class': pclass,
                'Sex': 'Male' if sex == 1 else 'Female',
                'Age': age,
                'Fare': f'${fare:.2f}',
                'Embarked': ['Cherbourg', 'Queenstown', 'Southampton'][embarked]
            }
        }
        
        return jsonify(result)
    
    except ValueError as e:
        return jsonify({'error': f'Invalid input: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': f'Prediction error: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

if __name__ == '__main__':
    # Create model directory if it doesn't exist
    os.makedirs('model', exist_ok=True)
    
    # Run the app
    app.run(debug=True, host='0.0.0.0', port=5000)