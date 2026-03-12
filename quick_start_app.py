"""
Quick Start Version of Biogas Forecasting App
Focuses on ML forecasting with basic physics validation
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

# Page configuration
st.set_page_config(
    page_title="Biogas Methane Forecasting - Quick Start",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load and cache the biogas dataset"""
    try:
        df = pd.read_csv('biogas_digestor_dataset.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        return df
    except FileNotFoundError:
        st.error("❌ Dataset not found! Please ensure 'biogas_digestor_dataset.csv' is in the project directory.")
        return None

@st.cache_data
def prepare_ml_data(df):
    """Prepare data for machine learning"""
    # Create time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # Create lag features
    for lag in [1, 2, 3]:
        df[f'methane_lag_{lag}'] = df['methane_production_m3_day'].shift(lag)
    
    # Create rolling features
    df['methane_rolling_mean_5'] = df['methane_production_m3_day'].rolling(5).mean()
    df['methane_rolling_std_5'] = df['methane_production_m3_day'].rolling(5).std()
    
    # Remove NaN values
    df_clean = df.dropna()
    
    # Prepare features and target
    feature_cols = [col for col in df_clean.columns if col not in ['timestamp', 'methane_production_m3_day']]
    X = df_clean[feature_cols]
    y = df_clean['methane_production_m3_day']
    
    return X, y, feature_cols

@st.cache_resource
def train_model(X, y):
    """Train the forecasting model"""
    # Time series split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    model = GradientBoostingRegressor(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_scaled)
    metrics = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'MAE': mean_absolute_error(y_test, y_pred),
        'R²': r2_score(y_test, y_pred)
    }
    
    return model, scaler, metrics

def validate_physics(features):
    """Basic physics validation without PINN"""
    warnings = []
    
    # Check VS/TS ratio
    if features['volatile_solids_percent'] > features['total_solids_percent']:
        warnings.append("WARNING: Volatile solids cannot exceed total solids")
    
    vs_ts_ratio = features['volatile_solids_percent'] / features['total_solids_percent']
    if vs_ts_ratio < 0.6 or vs_ts_ratio > 0.9:
        warnings.append(f"WARNING: VS/TS ratio ({vs_ts_ratio:.2f}) outside typical range (0.6-0.9)")
    
    # Check temperature range
    if features['temperature_celsius'] < 32 or features['temperature_celsius'] > 42:
        warnings.append("WARNING: Temperature outside optimal mesophilic range (32-42°C)")
    
    # Check pH range
    if features['ph'] < 6.5 or features['ph'] > 8.0:
        warnings.append("WARNING: pH outside optimal range (6.5-8.0)")
    
    # Check loading rate
    loading_rate = features['feed_volume_m3_day'] * features['volatile_solids_percent'] / 100
    if loading_rate > 4.0:
        warnings.append(f"WARNING: High loading rate ({loading_rate:.2f} kg VS/m³/day) may cause inhibition")
    
    return warnings

def suggest_optimal_conditions(features):
    """Suggest optimal conditions based on biogas science"""
    optimal = features.copy()
    
    # Temperature optimization
    optimal['temperature_celsius'] = 37.0  # Mesophilic optimum
    
    # pH optimization
    optimal['ph'] = 7.0  # Neutral pH optimum
    
    # Alkalinity for good buffering
    optimal['alkalinity_mg_l'] = max(3000, features['alkalinity_mg_l'])
    
    # Optimize retention time
    if features['retention_time_days'] < 20:
        optimal['retention_time_days'] = 20.0
    
    # Optimize VS content
    if features['total_solids_percent'] < 10:
        optimal['total_solids_percent'] = 10.0
        optimal['volatile_solids_percent'] = optimal['total_solids_percent'] * 0.8
    
    return optimal

def generate_24h_forecast(model, scaler, current_features, feature_cols):
    """Generate 24-hour forecast"""
    predictions = []
    timestamps = []
    
    # Convert features to model format
    feature_array = np.zeros(len(feature_cols))
    
    # Map basic features
    basic_mapping = {
        'feed_volume_m3_day': 0, 'total_solids_percent': 1, 'volatile_solids_percent': 2,
        'cod_mg_l': 3, 'temperature_celsius': 4, 'ph': 5, 'alkalinity_mg_l': 6,
        'retention_time_days': 7, 'mixing_intensity_rpm': 8
    }
    
    for key, idx in basic_mapping.items():
        if idx < len(feature_array) and key in current_features:
            feature_array[idx] = current_features[key]
    
    # Add time features
    now = datetime.now()
    for i, col in enumerate(feature_cols):
        if 'hour_of_day' in col:
            feature_array[i] = now.hour
        elif 'day_of_week' in col:
            feature_array[i] = now.weekday()
        elif 'is_weekend' in col:
            feature_array[i] = 1 if now.weekday() >= 5 else 0
    
    # Generate 24-hour forecast (480 predictions for 3-minute intervals)
    current_time = now
    for step in range(480):  # 24 hours * 20 (3-minute intervals)
        # Scale and predict
        feature_scaled = scaler.transform(feature_array.reshape(1, -1))
        pred = model.predict(feature_scaled)[0]
        
        predictions.append(pred)
        timestamps.append(current_time + timedelta(minutes=3 * step))
        
        # Update lag features (simplified)
        for i, col in enumerate(feature_cols):
            if 'methane_lag_1' in col:
                feature_array[i] = pred
    
    return timestamps, predictions

def main():
    st.title("Biogas Methane Production Forecasting")
    st.markdown("*Quick Start Version - ML Forecasting with Basic Physics Validation*")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar
    st.sidebar.title("Navigation")
    mode = st.sidebar.selectbox("Select Mode", ["Model Training", "24-Hour Forecast", "Data Analysis"])
    
    if mode == "Model Training":
        st.header("📊 Model Training")
        
        with st.spinner("Preparing data and training model..."):
            X, y, feature_cols = prepare_ml_data(df)
            model, scaler, metrics = train_model(X, y)
        
        st.success("✅ Model trained successfully!")
        
        # Show metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("RMSE", f"{metrics['RMSE']:.2f}")
        with col2:
            st.metric("MAE", f"{metrics['MAE']:.2f}")
        with col3:
            st.metric("R²", f"{metrics['R²']:.3f}")
        
        # Store in session state
        st.session_state.model = model
        st.session_state.scaler = scaler
        st.session_state.feature_cols = feature_cols
    
    elif mode == "24-Hour Forecast":
        st.header("🔮 24-Hour Methane Production Forecast")
        
        if 'model' not in st.session_state:
            st.warning("⚠️ Please train the model first!")
            st.stop()
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("🔧 Operating Conditions")
            
            # Input parameters
            features = {}
            features['feed_volume_m3_day'] = st.number_input("Feed Volume (m³/day)", value=50.0, min_value=10.0, max_value=100.0)
            features['total_solids_percent'] = st.number_input("Total Solids (%)", value=10.0, min_value=1.0, max_value=20.0)
            features['volatile_solids_percent'] = st.number_input("Volatile Solids (%)", value=8.0, min_value=1.0, max_value=15.0)
            features['cod_mg_l'] = st.number_input("COD (mg/L)", value=25000, min_value=5000, max_value=45000, step=1000)
            features['temperature_celsius'] = st.number_input("Temperature (°C)", value=37.0, min_value=25.0, max_value=50.0)
            features['ph'] = st.number_input("pH", value=7.0, min_value=5.0, max_value=9.0, step=0.1)
            features['alkalinity_mg_l'] = st.number_input("Alkalinity (mg/L)", value=3000, min_value=1000, max_value=6000, step=100)
            features['retention_time_days'] = st.number_input("Retention Time (days)", value=20.0, min_value=5.0, max_value=40.0)
            features['mixing_intensity_rpm'] = st.number_input("Mixing Intensity (RPM)", value=15.0, min_value=5.0, max_value=35.0)
            
            # Physics validation
            st.subheader("🧪 Physics Validation")
            warnings = validate_physics(features)
            
            if warnings:
                for warning in warnings:
                    st.warning(warning)
            else:
                st.success("✅ All parameters within optimal ranges!")
            
            # Optimal conditions
            optimal = suggest_optimal_conditions(features)
            if st.button("🎯 Show Optimal Conditions"):
                st.subheader("🎯 Suggested Optimal Conditions")
                for key, value in optimal.items():
                    current_val = features[key]
                    if abs(value - current_val) > 0.01:
                        st.write(f"**{key.replace('_', ' ').title()}**: {current_val:.2f} → {value:.2f}")
            
            # Generate forecast
            if st.button("🔮 Generate Forecast", type="primary"):
                with st.spinner("Generating 24-hour forecast..."):
                    timestamps, predictions = generate_24h_forecast(
                        st.session_state.model, 
                        st.session_state.scaler, 
                        features, 
                        st.session_state.feature_cols
                    )
                    
                    st.session_state.forecast_timestamps = timestamps
                    st.session_state.forecast_predictions = predictions
                    st.success("✅ Forecast generated!")
        
        with col2:
            st.subheader("📈 Forecast Results")
            
            if 'forecast_predictions' in st.session_state:
                timestamps = st.session_state.forecast_timestamps
                predictions = st.session_state.forecast_predictions
                
                # Plot forecast
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=timestamps, y=predictions,
                    mode='lines', name='Forecast',
                    line=dict(color='blue', width=2)
                ))
                
                fig.update_layout(
                    title="24-Hour Methane Production Forecast",
                    xaxis_title="Time",
                    yaxis_title="Methane Production (m³/day)",
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Statistics
                col_a, col_b, col_c, col_d = st.columns(4)
                with col_a:
                    st.metric("Average", f"{np.mean(predictions):.2f} m³/day")
                with col_b:
                    st.metric("Maximum", f"{np.max(predictions):.2f} m³/day")
                with col_c:
                    st.metric("Minimum", f"{np.min(predictions):.2f} m³/day")
                with col_d:
                    st.metric("Std Dev", f"{np.std(predictions):.2f} m³/day")
                
                # Download option
                df_forecast = pd.DataFrame({
                    'timestamp': timestamps,
                    'methane_production_forecast': predictions
                })
                csv = df_forecast.to_csv(index=False)
                st.download_button(
                    "📥 Download Forecast CSV",
                    csv,
                    f"methane_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    "text/csv"
                )
            else:
                st.info("👈 Configure parameters and generate forecast to see results")
    
    elif mode == "Data Analysis":
        st.header("📈 Historical Data Analysis")
        
        # Time series plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'], 
            y=df['methane_production_m3_day'],
            mode='lines', name='Methane Production'
        ))
        fig.update_layout(
            title="Historical Methane Production",
            xaxis_title="Time",
            yaxis_title="Methane Production (m³/day)"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Average Production", f"{df['methane_production_m3_day'].mean():.2f} m³/day")
        with col2:
            st.metric("Maximum Production", f"{df['methane_production_m3_day'].max():.2f} m³/day")
        with col3:
            st.metric("Standard Deviation", f"{df['methane_production_m3_day'].std():.2f} m³/day")
        with col4:
            st.metric("Data Points", f"{len(df):,}")
        
        # Correlation analysis
        st.subheader("🔗 Feature Correlations")
        key_features = ['methane_production_m3_day', 'temperature_celsius', 'ph', 'feed_volume_m3_day', 'volatile_solids_percent']
        corr_matrix = df[key_features].corr()
        
        fig_corr = px.imshow(corr_matrix, aspect="auto", color_continuous_scale="RdBu_r")
        st.plotly_chart(fig_corr, use_container_width=True)

if __name__ == "__main__":
    main()
