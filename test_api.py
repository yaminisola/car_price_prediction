import requests
import json

print("🧪 Testing Car Price Prediction API\n")

# Test data
test_car = {
    "brand": "Toyota",
    "model_year": 2020,
    "mileage": 30000,
    "fuel_type": "Petrol",
    "transmission": "Automatic",
    "engine_size": 2.5,
    "horsepower": 200,
    "condition": "Good"
}

try:
    response = requests.post(
        'http://127.0.0.1:5000/predict',
        json=test_car
    )
    
    result = response.json()
    print("✅ API Response:")
    print(json.dumps(result, indent=2))
    
except requests.exceptions.ConnectionError:
    print("❌ ERROR: Cannot connect to API")
    print("   Make sure the Flask server is running (python app.py)")
except Exception as e:
    print(f"❌ ERROR: {e}")