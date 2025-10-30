import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Create directories
os.makedirs('data', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('visualizations', exist_ok=True)

print("=" * 60)
print("🚗 CAR PRICE PREDICTION - ML PIPELINE")
print("=" * 60)

# Step 1: Generate synthetic dataset
print("\n[1/6] 📊 Generating synthetic car dataset...")
np.random.seed(42)

n_samples = 1000
data = {
    'brand': np.random.choice(['Toyota', 'Honda', 'BMW', 'Mercedes', 'Audi'], n_samples),
    'model_year': np.random.randint(2010, 2024, n_samples),
    'mileage': np.random.randint(5000, 200000, n_samples),
    'fuel_type': np.random.choice(['Petrol', 'Diesel', 'Electric', 'Hybrid'], n_samples),
    'transmission': np.random.choice(['Manual', 'Automatic'], n_samples),
    'engine_size': np.random.uniform(1.0, 5.0, n_samples),
    'horsepower': np.random.randint(80, 400, n_samples),
    'condition': np.random.choice(['Excellent', 'Good', 'Fair', 'Poor'], n_samples),
    'price': np.random.randint(5000, 80000, n_samples)
}

df = pd.DataFrame(data)
df.to_csv('data/car_dataset.csv', index=False)
print(f"✅ Dataset created: {len(df)} records")
print(f"   Saved to: data/car_dataset.csv")

# Step 2: Data Preprocessing
print("\n[2/6] 🔧 Preprocessing data...")

# Encode categorical variables
label_encoders = {}
categorical_cols = ['brand', 'fuel_type', 'transmission', 'condition']

for col in categorical_cols:
    le = LabelEncoder()
    df[col + '_encoded'] = le.fit_transform(df[col])
    label_encoders[col] = le

# Features and target
feature_cols = ['brand_encoded', 'model_year', 'mileage', 'fuel_type_encoded', 
                'transmission_encoded', 'engine_size', 'horsepower', 'condition_encoded']
X = df[feature_cols]
y = df['price']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✅ Training set: {len(X_train)} samples")
print(f"✅ Test set: {len(X_test)} samples")

# Step 3: Train Price Prediction Model
print("\n[3/6] 🤖 Training price prediction model...")
price_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
price_model.fit(X_train_scaled, y_train)

# Predictions
y_pred = price_model.predict(X_test_scaled)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"   RMSE: ${rmse:,.2f}")
print(f"   R² Score: {r2:.4f}")

# Step 4: Train Condition Classifier
print("\n[4/6] 🎯 Training condition classifier...")
condition_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
condition_X = df[['model_year', 'mileage', 'engine_size', 'horsepower']]
condition_y = df['condition']

condition_X_train, condition_X_test, condition_y_train, condition_y_test = train_test_split(
    condition_X, condition_y, test_size=0.2, random_state=42
)

condition_model.fit(condition_X_train, condition_y_train)
condition_pred = condition_model.predict(condition_X_test)
condition_accuracy = (condition_pred == condition_y_test).mean()

print(f"✅ Condition classifier trained!")
print(f"   Accuracy: {condition_accuracy:.2%}")

# Step 5: Save Models
print("\n[5/6] 💾 Saving models...")
with open('models/car_price_regressor.pkl', 'wb') as f:
    pickle.dump(price_model, f)

with open('models/car_condition_classifier.pkl', 'wb') as f:
    pickle.dump(condition_model, f)

with open('models/feature_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('models/label_encoders.pkl', 'wb') as f:
    pickle.dump(label_encoders, f)

print("✅ All models saved to 'models/' directory")

# Step 6: Create Visualizations
print("\n[6/6] 📊 Creating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Plot 1: Actual vs Predicted Prices
axes[0, 0].scatter(y_test, y_pred, alpha=0.5)
axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
axes[0, 0].set_xlabel('Actual Price ($)')
axes[0, 0].set_ylabel('Predicted Price ($)')
axes[0, 0].set_title('Price Prediction: Actual vs Predicted')

# Plot 2: Feature Importance
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': price_model.feature_importances_
}).sort_values('importance', ascending=False)

axes[0, 1].barh(feature_importance['feature'], feature_importance['importance'])
axes[0, 1].set_xlabel('Importance')
axes[0, 1].set_title('Feature Importance for Price Prediction')

# Plot 3: Price Distribution by Brand
axes[1, 0].boxplot([df[df['brand'] == brand]['price'] for brand in df['brand'].unique()],
                    labels=df['brand'].unique())
axes[1, 0].set_ylabel('Price ($)')
axes[1, 0].set_title('Price Distribution by Brand')
axes[1, 0].tick_params(axis='x', rotation=45)

# Plot 4: Mileage vs Price
axes[1, 1].scatter(df['mileage'], df['price'], alpha=0.5, c=df['model_year'], cmap='viridis')
axes[1, 1].set_xlabel('Mileage')
axes[1, 1].set_ylabel('Price ($)')
axes[1, 1].set_title('Price vs Mileage (colored by year)')
cbar = plt.colorbar(axes[1, 1].collections[0], ax=axes[1, 1])
cbar.set_label('Model Year')

plt.tight_layout()
plt.savefig('visualizations/model_evaluation_results.png', dpi=300, bbox_inches='tight')
print("✅ Visualizations saved to 'visualizations/model_evaluation_results.png'")

print("\n" + "=" * 60)
print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
print("=" * 60)
print("\n📁 Generated Files:")
print("   - data/car_dataset.csv")
print("   - models/car_price_regressor.pkl")
print("   - models/car_condition_classifier.pkl")
print("   - models/feature_scaler.pkl")
print("   - models/label_encoders.pkl")
print("   - visualizations/model_evaluation_results.png")
print("\n🚀 Next step: Run 'python app.py' to start the Flask API\n")