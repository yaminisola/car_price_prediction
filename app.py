from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for API calls

print("=" * 60)
print("🚗 CAR PRICE PREDICTION API")
print("=" * 60)

# Load models
print("\n📦 Loading models...")

try:
    with open('models/car_price_regressor.pkl', 'rb') as f:
        price_model = pickle.load(f)
    
    with open('models/feature_scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    
    with open('models/label_encoders.pkl', 'rb') as f:
        label_encoders = pickle.load(f)
    
    print("✅ All models loaded successfully!")
except FileNotFoundError:
    print("❌ ERROR: Model files not found!")
    print("   Please run 'python main.py' first to train the models.")
    exit(1)

@app.route('/')
def home():
    """Serve the HTML dashboard"""
    return render_template('index.html')

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "models_loaded": True})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Encode categorical variables
        brand_encoded = label_encoders['brand'].transform([data['brand']])[0]
        fuel_encoded = label_encoders['fuel_type'].transform([data['fuel_type']])[0]
        transmission_encoded = label_encoders['transmission'].transform([data['transmission']])[0]
        condition_encoded = label_encoders['condition'].transform([data['condition']])[0]
        
        # Prepare features
        features = np.array([[
            brand_encoded,
            data['model_year'],
            data['mileage'],
            fuel_encoded,
            transmission_encoded,
            data['engine_size'],
            data['horsepower'],
            condition_encoded
        ]])
        
        # Scale and predict
        features_scaled = scaler.transform(features)
        predicted_price = price_model.predict(features_scaled)[0]
        
        return jsonify({
            "success": True,
            "predicted_price": round(float(predicted_price), 2),
            "input_data": data
        })
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 400

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("🚀 Starting Flask API server...")
    print("=" * 60)
    print("\n📍 Dashboard will be available at: http://127.0.0.1:5000")
    print("📖 API Endpoints:")
    print("   GET  /         - Web Dashboard")
    print("   GET  /health   - Health check")
    print("   POST /predict  - Predict car price (API)")
    print("\n✋ Press CTRL+C to stop the server\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)