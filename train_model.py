import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import matplotlib.pyplot as plt

def train_model():
    """Train power plant performance prediction model"""
    
    print("Loading dataset...")
    try:
        data = pd.read_csv('dataset.csv')
        print(f"Dataset loaded successfully with {data.shape[0]} samples.")
    except FileNotFoundError:
        print("Error: Could not find dataset.csv file.")
        return None, None
    
    # Prepare features and target
    X = data[['AT', 'V', 'AP', 'RH']]
    y = data['PE']
    
    # Add polynomial features (quadratic terms)
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model
    print("\nTraining Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred = rf_model.predict(X_test_scaled)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    print(f"\nModel Evaluation:")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"R² Score: {r2:.4f}")
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model and scaler
    joblib.dump(rf_model, 'models/power_plant_model.pkl')
    joblib.dump(scaler, 'models/power_plant_scaler.pkl')
    
    print("\nModel and scaler saved successfully!")
    
    # Plot actual vs predicted values
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    plt.xlabel('Actual Power Output (MW)')
    plt.ylabel('Predicted Power Output (MW)')
    plt.title('Actual vs Predicted Power Output')
    
    # Save the plot
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/prediction_performance.png')
    plt.close()
    
    return rf_model, scaler

def predict_power_output(at, v, ap, rh):
    """Predict power output based on environmental parameters"""
    # Load model and scaler
    try:
        model = joblib.load('models/power_plant_model.pkl')
        scaler = joblib.load('models/power_plant_scaler.pkl')
        # We also need to load or recreate the polynomial features transformer
        poly = PolynomialFeatures(degree=2, include_bias=False)
    except FileNotFoundError:
        print("Error: Model files not found. Please train the model first.")
        return None
    
    # Create input array
    input_data = np.array([[at, v, ap, rh]])
    
    # Transform to polynomial features
    input_poly = poly.fit_transform(input_data)
    
    # Scale input
    input_scaled = scaler.transform(input_poly)
    
    # Make prediction
    prediction = model.predict(input_scaled)[0]
    
    return prediction

if __name__ == '__main__':
    train_model()