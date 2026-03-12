# Biogas Methane Production Forecasting Engine
 
AI-powered forecasting system for predicting biogas methane production using machine learning and Physics-Informed Neural Networks (PINNs). The system forecasts methane output for the next 24 hours and provides operational insights through an interactive Streamlit dashboard.

Developed during my AI/ML Trainee Engineer Internship at NetSol Technologies.

# Project Overview

Biogas plants rely on stable biological processes to generate methane through anaerobic digestion. Predicting methane production helps operators optimize feeding, temperature, and environmental conditions to improve efficiency.

This project builds a hybrid forecasting engine that combines:

Machine Learning models for accurate time-series forecasting

Physics-informed neural networks to enforce real-world biogas digestion constraints

Feature engineering to capture temporal patterns

Interactive dashboards for visualization and monitoring

The system predicts methane production for 24 hours ahead (480 predictions) at 3-minute intervals.

# Key Features

Hybrid forecasting engine combining ML + physics modeling

XGBoost, LightGBM, and Gradient Boosting models

Physics-Informed Neural Network (PINN) for validating biogas process conditions

Feature engineering including lag features and rolling statistics

Real-time forecasting interface

Interactive Streamlit dashboard

Visualization of predictions and operational insights

Automated corrective action suggestions

#Forecasting Architecture

The system follows a multi-stage pipeline:

Biogas Plant Parameters
        │
        ▼
Feature Engineering
(Lag features, rolling statistics,
time-based features)
        │
        ▼
Machine Learning Models
• Gradient Boosting
• XGBoost
• LightGBM
        │
        ▼
Physics Validation
(Physics-Informed Neural Network)
        │
        ▼
24-Hour Methane Production Forecast
        │
        ▼
Interactive Streamlit Dashboard

#  Machine Learning Models

Three ML models are trained and compared:

Gradient Boosting

Traditional ensemble boosting model used as a baseline.

XGBoost

Optimized gradient boosting implementation designed for high predictive performance.

LightGBM

Efficient gradient boosting framework designed for fast training and large datasets.

These models predict methane production based on operational and environmental parameters.

 # Physics-Informed Neural Network (PINN)

A Physics-Informed Neural Network is integrated to ensure predictions follow real-world biogas digestion behavior.

The PINN incorporates physics-based constraints such as:

Monod kinetics

Temperature dependence

pH inhibition

Microbial growth dynamics

Substrate utilization relationships

Digestion stability constraints

The PINN module also:

Validates feature consistency

Detects unrealistic input conditions

Suggests optimal operating parameters

Recommends corrective actions

# Feature Engineering

To improve forecasting performance, several feature engineering techniques were applied:

Lag Features

Capture historical methane production patterns.

Examples:

methane_lag_1

methane_lag_2

methane_lag_6

methane_lag_24

Rolling Statistics

Capture short-term trends and volatility.

Examples:

rolling mean

rolling standard deviation

Time-Based Features

Hour of day

Day of week

Weekend indicator

Cyclical Features

To properly encode time:

hour_sin
hour_cos
day_sin
day_cos

These features help the models learn daily and weekly production cycles.

# Forecasting Method

The system generates 24-hour forecasts using recursive predictions.

Key characteristics:

Prediction horizon: 24 hours

Prediction frequency: 3-minute intervals

Total predictions: 480 per forecast

During forecasting:

Current operating conditions are input

Models generate the next prediction

Lag features are updated

Process repeats recursively

The system can also introduce controlled noise in temperature and pH to simulate real-world fluctuations.

# Streamlit Dashboard

The project includes a fully interactive dashboard built with Streamlit.

Dashboard capabilities include:

Model Training Interface

Load dataset

Train ML models

Compare performance metrics

Forecasting Panel

Input plant parameters

Generate 24-hour forecasts

Visualize methane production curves

Physics Validation

Validate operating conditions

Detect constraint violations

Suggest corrective actions

Analysis Dashboard

Historical methane production visualization

Feature correlation analysis

Parameter distribution plots

System performance metrics

# Input Parameters

The system models several operational variables of a biogas plant:

Feed volume

Total solids percentage

Volatile solids percentage

Chemical oxygen demand (COD)

Temperature

pH

Alkalinity

Retention time

Mixing intensity

These parameters directly influence methane production.

# Dataset

The original dataset was not publicly available, therefore a synthetic dataset was generated to simulate realistic biogas plant operating conditions.

The dataset includes parameters such as:

Feed characteristics

Environmental conditions

Reactor operating variables

Methane production output

The synthetic data was designed to reflect typical anaerobic digestion patterns and was used to train and evaluate the forecasting models.

# Evaluation Metrics

Model performance is evaluated using:

RMSE (Root Mean Squared Error)

MAE (Mean Absolute Error)

R² Score

These metrics compare predicted methane production against actual values.

# Technologies Used

Python ecosystem:

Python

Streamlit

PyTorch

Scikit-learn

XGBoost

LightGBM

Pandas

NumPy

Plotly

These libraries power the machine learning pipeline and interactive visualization system.

# Installation

Clone the repository:

git clone https://github.com/yourusername/biogas-methane-forecasting-engine.git
cd biogas-methane-forecasting-engine

Install dependencies:

pip install -r requirements.txt

# Run the Application

Start the Streamlit dashboard:

streamlit run streamlit_biogas_forecasting.py

Then open:

http://localhost:8501

# Project Structure

Example structure:

biogas-methane-forecasting-engine

├── streamlit_biogas_forecasting.py
├── physics_informed_neural_network.py
├── biogas_digestor_dataset.csv
├── best_pinn_model.pt
├── requirements.txt
└── README.md

# Future Improvements

Possible extensions for this project:

48-hour and 72-hour forecasting

Real-time sensor integration

LSTM or Transformer-based forecasting models

Automated hyperparameter tuning

Deployment using Docker

Cloud-based monitoring dashboard

# Author

Zeerak Ahmed

AI / Machine Learning Enthusiast
Focused on Machine Learning and Product Management

Developed during AI/ML Trainee Engineer Internship at NetSol Technologies

# License

This project is for educational and research purposes.

# Portfolio Impact

This project demonstrates:

Applied machine learning engineering

Time-series forecasting

Integration of physics-based modeling with AI

Feature engineering

Interactive ML dashboards

Industrial AI applications