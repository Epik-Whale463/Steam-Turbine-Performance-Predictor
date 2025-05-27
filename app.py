from flask import Flask, render_template, request, jsonify, send_from_directory
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
import joblib
import os
from datetime import datetime
import json
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Global variables for model components
model = None
scaler = None
poly = None

def load_model_components():
    """Load trained model and preprocessing components"""
    global model, scaler, poly
    try:
        logger.info("Loading model components...")
        model = joblib.load('models/power_plant_model.pkl')
        scaler = joblib.load('models/power_plant_scaler.pkl')
        logger.info("Model and scaler loaded successfully")
        
        # Recreate polynomial features transformer with same parameters
        poly = PolynomialFeatures(degree=2, include_bias=False)
        
        # Create dummy data to fit the transformer
        dummy_data = pd.DataFrame(columns=['AT', 'V', 'AP', 'RH'])
        dummy_array = np.array([[20, 40, 1000, 75]])  # Sample values
        poly.fit(dummy_array)
        logger.info("Polynomial features transformer initialized")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model components: {str(e)}")
        return False

# Load model components on startup
model_loaded = load_model_components()
if not model_loaded:
    logger.warning("Failed to load model components. Please ensure model files exist.")

def calculate_efficiency(power_output, ambient_temp):
    """Calculate plant efficiency based on power output and ambient temperature"""
    base_efficiency = 45.0  # Base efficiency at standard conditions
    temp_coefficient = -0.1  # Efficiency change per degree C from standard temp (20°C)
    temp_diff = ambient_temp - 20
    efficiency = base_efficiency + (temp_coefficient * temp_diff)
    # Adjust efficiency based on power output
    efficiency *= (1 + (power_output - 450) / 1000)  # Slight increase/decrease based on output
    return min(max(efficiency, 30), 60)  # Keep efficiency between 30% and 60%

def calculate_heat_rate(efficiency):
    """Calculate heat rate from efficiency"""
    return 3412.142 / (efficiency / 100)

def calculate_fuel_consumption(power_output):
    """Calculate approximate fuel consumption in tons per hour"""
    # Assuming coal with average heat content of 24 MJ/kg
    heat_rate = 10500  # Typical heat rate in BTU/kWh
    coal_heat_content = 24000  # kJ/kg
    conversion_factor = 1.0548  # BTU to kJ
    
    fuel_consumption = (power_output * 1000 * heat_rate) / (coal_heat_content * conversion_factor * 1000)
    return fuel_consumption

def get_performance_category(power_output):
    """Determine performance category based on power output"""
    if power_output >= 500:
        category = "Excellent"
        description = "Plant is operating at peak efficiency"
    elif power_output >= 450:
        category = "Good"
        description = "Plant is operating within normal parameters"
    elif power_output >= 400:
        category = "Average"
        description = "Plant performance could be optimized"
    else:
        category = "Below Average"
        description = "Plant efficiency needs improvement"
        
    return {
        "category": category,
        "description": description
    }

@app.route('/')
def home():
    """Main dashboard page"""
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    if not model_loaded:
        logger.error("Prediction attempted but model is not loaded")
        return jsonify({
            'status': 'error',
            'message': 'Model not loaded. Please ensure model files exist.'
        })
    try:
        # Validate that all required fields are present
        required_fields = ['at', 'v', 'ap', 'rh']
        if not all(field in request.form for field in required_fields):
            missing_fields = [field for field in required_fields if field not in request.form]
            raise ValueError(f"Missing required fields: {', '.join(missing_fields)}")

        # Get and validate input parameters
        try:
            at = float(request.form['at'])  # Ambient Temperature (°C)
            v = float(request.form['v'])   # Exhaust Vacuum (cm Hg)
            ap = float(request.form['ap']) # Ambient Pressure (millibar)
            rh = float(request.form['rh']) # Relative Humidity (%)
        except ValueError as e:
            raise ValueError("All fields must be valid numbers")
            
        logger.info(f"Received prediction request with parameters: AT={at}, V={v}, AP={ap}, RH={rh}")
        
        # Define valid ranges
        valid_ranges = {
            'at': (1.81, 37.11, "Ambient Temperature"),
            'v': (25.36, 81.56, "Exhaust Vacuum"),
            'ap': (992.89, 1033.30, "Ambient Pressure"),
            'rh': (25.56, 100.16, "Relative Humidity")
        }
        
        # Check for extreme values
        warnings = []
        for param, value in [('at', at), ('v', v), ('ap', ap), ('rh', rh)]:
            min_val, max_val, name = valid_ranges[param]
            if value < min_val or value > max_val:
                warnings.append(f"{name} is outside typical range ({min_val} to {max_val})")
        
        # Clip values to prevent model issues while allowing predictions
        at = max(1.81, min(37.11, at))
        v = max(25.36, min(81.56, v))
        ap = max(992.89, min(1033.30, ap))
        rh = max(25.56, min(100.16, rh))
        logger.info(f"Adjusted parameters (if needed): AT={at}, V={v}, AP={ap}, RH={rh}")
        
        # Create input array
        input_data = np.array([[at, v, ap, rh]])
        logger.info("Input validated successfully")
        
        # Transform to polynomial features
        input_poly = poly.transform(input_data)
        
        # Scale input
        input_scaled = scaler.transform(input_poly)
        
        # Make prediction
        prediction = model.predict(input_scaled)[0]
        logger.info(f"Raw prediction: {prediction}")
        
        # Calculate additional metrics
        efficiency = calculate_efficiency(prediction, at)
        heat_rate = calculate_heat_rate(efficiency)
        fuel_consumption = calculate_fuel_consumption(prediction)
        performance_category = get_performance_category(prediction)
        
        result = {
            'status': 'success',
            'prediction': round(prediction, 2),
            'efficiency': round(efficiency, 2),
            'heat_rate': round(heat_rate, 0),
            'fuel_consumption': round(fuel_consumption, 2),
            'performance_category': performance_category,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'input_parameters': {
                'ambient_temp': at,
                'exhaust_vacuum': v,
                'ambient_pressure': ap,
                'relative_humidity': rh
            }
        }
        
        logger.info("Prediction successful")
        return jsonify(result)
        
    except ValueError as e:
        logger.warning(f"Validation error: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': f'An unexpected error occurred: {str(e)}'
        })

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """RESTful API endpoint for external integration"""
    if not model_loaded:
        return jsonify({'error': 'Model not available'}), 503
    
    try:
        data = request.get_json()
        
        # Extract parameters
        at = float(data['ambient_temp'])
        v = float(data['exhaust_vacuum'])
        ap = float(data['ambient_pressure'])
        rh = float(data['relative_humidity'])
        
        # Create input and predict
        input_data = np.array([[at, v, ap, rh]])
        input_poly = poly.transform(input_data)
        input_scaled = scaler.transform(input_poly)
        prediction = model.predict(input_scaled)[0]
        
        return jsonify({
            'power_output_mw': round(prediction, 2),
            'efficiency_percent': round(calculate_efficiency(prediction, at), 2),
            'performance_category': get_performance_category(prediction),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint for multiple scenarios"""
    if not model_loaded:
        return jsonify({'error': 'Model not available'}), 503
    
    try:
        data = request.get_json()
        scenarios = data['scenarios']
        
        results = []
        for i, scenario in enumerate(scenarios):
            input_data = np.array([[
                scenario['ambient_temp'],
                scenario['exhaust_vacuum'],
                scenario['ambient_pressure'],
                scenario['relative_humidity']
            ]])
            
            input_poly = poly.transform(input_data)
            input_scaled = scaler.transform(input_poly)
            prediction = model.predict(input_scaled)[0]
            
            results.append({
                'scenario_id': i + 1,
                'power_output_mw': round(prediction, 2),
                'efficiency_percent': round(calculate_efficiency(prediction, scenario['ambient_temp']), 2),
                'performance_category': get_performance_category(prediction)
            })
        
        return jsonify({
            'results': results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)

@app.route('/model-info')
def model_info():
    """Return model performance metrics"""
    return jsonify({
        'r2_score': 0.9641,
        'rmse': 3.2255,
        'model_type': 'Random Forest',
        'features': ['Ambient Temperature', 'Exhaust Vacuum', 'Ambient Pressure', 'Relative Humidity'],
        'polynomial_degree': 2,
        'n_estimators': 100,
        'max_depth': 15
    })

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)