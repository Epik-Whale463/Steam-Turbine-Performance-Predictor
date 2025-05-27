# Steam Turbine Performance Predictor

This project is a web-based application that predicts the performance of a combined cycle power plant using a **Random Forest** machine learning model. The application is deployed on an AWS EC2 instance and can be accessed at:

**[http://51.20.71.234:5000/](http://51.20.71.234:5000/)**

---

## Features

- **AI-Powered Predictions**: Predict power output based on environmental parameters using a Random Forest model with polynomial features.
- **Real-Time Analytics**: Provides instant performance metrics, including efficiency, heat rate, and fuel consumption calculations.
- **Performance Optimization**: Identifies optimal operating conditions to maximize power output and minimize fuel consumption.
- **Model Performance Visualization**: Displays R² Score, RMSE, and model parameters for transparency.

---

## Model Performance

- **R² Score**: 96.41%
- **RMSE**: 3.23 MW
- **Trees**: 100
- **Max Depth**: 15

---

## How to Use

1. **Access the Application**: Visit [http://51.20.71.234:5000/](http://51.20.71.234:5000/).
2. **Enter Parameters**:
   - Ambient Temperature (°C)
   - Exhaust Vacuum (cm Hg)
   - Ambient Pressure (millibar)
   - Relative Humidity (%)
3. **Submit the Form**: Click the "Predict" button to get the power output prediction.
4. **View Results**: The prediction results and performance metrics will be displayed instantly.

---

## Deployment

The application is deployed on an **AWS EC2 instance** using Flask. Below are the deployment details:

- **Framework**: Flask (Python)
- **Frontend**: HTML, CSS, JavaScript (Bootstrap)
- **Backend**: Flask API with a Random Forest model
- **Model**: Trained using scikit-learn
- **Hosting**: AWS EC2 instance

---

## Installation (For Local Development)

1. Clone the repository:
   ```bash
   git clone https://github.com/Epik-Whale463/Steam-Turbine-Performance-Predictor
   cd SteamTurbinePerformancePredictor
