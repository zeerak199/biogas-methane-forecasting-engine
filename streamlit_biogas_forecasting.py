"""
Streamlit Application for Biogas Methane Production Forecasting
Combines Physics Informed Neural Network (PINN) with Gradient Boosting
for 24-hour ahead methane production forecasting.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import torch
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Machine Learning imports
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb

# Import our PINN components
try:
    from physics_informed_neural_network import (
        PhysicsInformedNN, 
        BiogasFeatureUpdater,
        BiogasPhysicsLaws
    )
    PINN_AVAILABLE = True
except ImportError:
    PINN_AVAILABLE = False
    st.error("PINN module not found. Please ensure physics_informed_neural_network.py is available.")

# Page configuration
st.set_page_config(
    page_title="Biogas Methane Forecasting System",
    layout="wide",
    initial_sidebar_state="expanded"
)

class BiogasForecastingSystem:
    """
    Main forecasting system combining PINN physics validation with ML forecasting
    """
    
    def __init__(self):
        self.pinn_model = None
        self.pinn_updater = None
        self.gb_model = None
        self.xgb_model = None
        self.lgb_model = None
        self.scaler = None
        self.feature_names = None
        self.physics_laws = BiogasPhysicsLaws()
        
    def load_models(self):
        """Load trained PINN and initialize ML models"""
        try:
            # Try to load trained PINN model
            if PINN_AVAILABLE:
                self.pinn_model = self._load_pinn_with_correct_architecture()
                if self.pinn_model is not None:
                    self.pinn_updater = BiogasFeatureUpdater(self.pinn_model)
                    st.success("PINN model loaded successfully!")
                else:
                    st.warning("Pre-trained PINN model not found. Physics validation will use default parameters.")
            else:
                st.error("PINN not available. Please check physics_informed_neural_network.py")
                
        except Exception as e:
            st.error(f"Error loading models: {str(e)}")
    
    def _load_pinn_with_correct_architecture(self):
        """Load PINN model with correct architecture detection"""
        try:
            # Try to load the state dict to inspect its structure
            checkpoint = torch.load('best_pinn_model.pt', map_location='cpu')
            
            # Detect input dimensions from feature_weights
            if 'feature_weights' in checkpoint:
                input_dim = checkpoint['feature_weights'].shape[0]
            else:
                input_dim = 17  # Default fallback
            
            # Detect hidden layer architecture from first layer
            if 'network.0.weight' in checkpoint:
                first_hidden = checkpoint['network.0.weight'].shape[0]
                
                # Detect network architecture based on layer sizes
                if first_hidden == 64:
                    hidden_dims = [64, 128, 128, 64]
                elif first_hidden == 32:
                    hidden_dims = [32, 64, 32]
                else:
                    hidden_dims = [128, 256, 256, 128]  # Default
            else:
                hidden_dims = [64, 128, 128, 64]  # Default from training
            
            # Create model with detected architecture
            model = PhysicsInformedNN(
                input_dim=input_dim,
                hidden_dims=hidden_dims,
                output_dim=1,
                physics_weight=0.5
            )
            
            # Load the state dict
            model.load_state_dict(checkpoint)
            model.eval()
            
            st.info(f"	 PINN loaded: {input_dim} inputs, hidden layers: {hidden_dims}")
            return model
            
        except FileNotFoundError:
            st.warning("		 best_pinn_model.pt not found. Trying alternative model...")
            try:
                # Try alternative model file
                checkpoint = torch.load('biogas_pinn_complete.pt', map_location='cpu')
                
                # Same detection logic for alternative model
                input_dim = checkpoint.get('feature_weights', torch.zeros(17)).shape[0]
                first_hidden = checkpoint.get('network.0.weight', torch.zeros(64, 17)).shape[0]
                
                if first_hidden == 64:
                    hidden_dims = [64, 128, 128, 64]
                else:
                    hidden_dims = [32, 64, 32]
                
                model = PhysicsInformedNN(
                    input_dim=input_dim,
                    hidden_dims=hidden_dims,
                    output_dim=1,
                    physics_weight=0.5
                )
                
                model.load_state_dict(checkpoint)
                model.eval()
                
                st.info(f"	 Alternative PINN loaded: {input_dim} inputs, hidden layers: {hidden_dims}")
                return model
                
            except FileNotFoundError:
                st.warning("		 No pre-trained PINN models found. Creating fresh model for demonstration.")
                # Create a fresh model for demo purposes
                return PhysicsInformedNN(input_dim=17, hidden_dims=[64, 128, 128, 64])
                
        except Exception as e:
            st.warning(f"		 Could not load pre-trained PINN: {str(e)}")
            st.info("	 Creating fresh PINN model for physics validation (without pre-trained weights)")
            # Create a fresh model for physics validation
            return PhysicsInformedNN(input_dim=17, hidden_dims=[64, 128, 128, 64], physics_weight=0.3)
    
    def prepare_data(self, df):
        """Prepare data for training ML models"""
        # Feature engineering for time series
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Create lag features
        for lag in [1, 2, 3, 6, 12, 24]:  # 3min, 6min, 9min, 18min, 36min, 72min lags
            df[f'methane_lag_{lag}'] = df['methane_production_m3_day'].shift(lag)
        
        # Create rolling statistics
        for window in [5, 10, 20]:
            df[f'methane_rolling_mean_{window}'] = df['methane_production_m3_day'].rolling(window=window).mean()
            df[f'methane_rolling_std_{window}'] = df['methane_production_m3_day'].rolling(window=window).std()
        
        # Time-based features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Remove rows with NaN values
        df = df.dropna()
        
        return df
    
    def train_ml_models(self, df):
        """Train Gradient Boosting models for forecasting"""
        try:
            # Prepare features
            feature_cols = [col for col in df.columns if col not in ['timestamp', 'methane_production_m3_day']]
            
            X = df[feature_cols]
            y = df['methane_production_m3_day']
            
            # Time series split (maintain chronological order)
            split_idx = int(len(df) * 0.8)
            X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
            
            # Scale features
            self.scaler = StandardScaler()
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.feature_names = feature_cols
            
            # Train Gradient Boosting
            self.gb_model = GradientBoostingRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                subsample=0.8
            )
            self.gb_model.fit(X_train_scaled, y_train)
            
            # Train XGBoost
            self.xgb_model = xgb.XGBRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                subsample=0.8
            )
            self.xgb_model.fit(X_train_scaled, y_train)
            
            # Train LightGBM
            self.lgb_model = lgb.LGBMRegressor(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=6,
                random_state=42,
                subsample=0.8,
                verbose=-1
            )
            self.lgb_model.fit(X_train_scaled, y_train)
            
            # Evaluate models
            models = {
                'Gradient Boosting': self.gb_model,
                'XGBoost': self.xgb_model,
                'LightGBM': self.lgb_model
            }
            
            results = {}
            for name, model in models.items():
                y_pred = model.predict(X_test_scaled)
                results[name] = {
                    'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'MAE': mean_absolute_error(y_test, y_pred),
                    'R²': r2_score(y_test, y_pred)
                }
            
            return results, X_test, y_test
            
        except Exception as e:
            st.error(f"Error training models: {str(e)}")
            return None, None, None
    
    def forecast_24_hours(self, current_features, model_choice='XGBoost'):
        """Generate 24-hour forecast using selected model"""
        if model_choice == 'Gradient Boosting':
            model = self.gb_model
        elif model_choice == 'XGBoost':
            model = self.xgb_model
        else:
            model = self.lgb_model
        
        if model is None or self.scaler is None:
            st.error("Models not trained yet!")
            return None, None
        
        # Convert current features to the expected format
        current_data = self.prepare_current_features(current_features)
        
        predictions = []
        timestamps = []
        current_time = datetime.now()
        
        # Generate 24-hour forecast (480 predictions for 3-minute intervals)
        n_predictions = 480  # 24 hours * 20 (3-minute intervals per hour)
        
        for step in range(n_predictions):
            # Predict next value
            current_scaled = self.scaler.transform(current_data.reshape(1, -1))
            pred = model.predict(current_scaled)[0]
            predictions.append(pred)
            
            # Update timestamp
            next_time = current_time + timedelta(minutes=3 * (step + 1))
            timestamps.append(next_time)
            
            # Update lag features for next prediction
            current_data = self.update_lag_features(current_data, pred, step)
        
        return timestamps, predictions
    
    def prepare_current_features(self, features_dict):
        """Convert feature dictionary to model input format"""
        # Create a base array with expected features
        if self.feature_names is None:
            # If no ML models trained yet, create a simplified array for PINN
            if self.pinn_model is not None:
                input_dim = self.pinn_model.input_dim
                feature_array = np.zeros(input_dim)
                
                # Basic feature mapping for PINN (first 9 features)
                basic_features = [
                    'feed_volume_m3_day', 'total_solids_percent', 'volatile_solids_percent',
                    'cod_mg_l', 'temperature_celsius', 'ph', 'alkalinity_mg_l',
                    'retention_time_days', 'mixing_intensity_rpm'
                ]
                
                for i, feature_name in enumerate(basic_features):
                    if i < input_dim and feature_name in features_dict:
                        feature_array[i] = features_dict[feature_name]
                
                # Add derived features if model expects them
                if input_dim > 9:
                    # Add hour of day
                    if input_dim > 10:
                        feature_array[10] = datetime.now().hour
                    # Add day of week
                    if input_dim > 11:
                        feature_array[11] = datetime.now().weekday()
                    # Add is_weekend
                    if input_dim > 12:
                        feature_array[12] = 1 if datetime.now().weekday() >= 5 else 0
                
                return feature_array
            else:
                st.error("No models available for feature preparation.")
                return None
        
        feature_array = np.zeros(len(self.feature_names))
        
        # Map input features to expected positions
        feature_mapping = {
            'feed_volume_m3_day': 0,
            'total_solids_percent': 1,
            'volatile_solids_percent': 2,
            'cod_mg_l': 3,
            'temperature_celsius': 4,
            'ph': 5,
            'alkalinity_mg_l': 6,
            'retention_time_days': 7,
            'mixing_intensity_rpm': 8
        }
        
        for key, value in features_dict.items():
            if key in feature_mapping and feature_mapping[key] < len(feature_array):
                feature_array[feature_mapping[key]] = value
        
        # Add time-based features (current time)
        now = datetime.now()
        hour = now.hour
        day_of_week = now.weekday()
        
        # Find positions for time features
        for i, fname in enumerate(self.feature_names):
            if 'hour_sin' in fname:
                feature_array[i] = np.sin(2 * np.pi * hour / 24)
            elif 'hour_cos' in fname:
                feature_array[i] = np.cos(2 * np.pi * hour / 24)
            elif 'day_sin' in fname:
                feature_array[i] = np.sin(2 * np.pi * day_of_week / 7)
            elif 'day_cos' in fname:
                feature_array[i] = np.cos(2 * np.pi * day_of_week / 7)
            elif 'is_weekend' in fname:
                feature_array[i] = 1 if day_of_week >= 5 else 0
            elif 'hour_of_day' in fname:
                feature_array[i] = hour
            elif 'day_of_week' in fname:
                feature_array[i] = day_of_week
        
        return feature_array
    
    def update_lag_features(self, current_data, new_prediction, step):
        """Update lag features with new prediction"""
        # This is a simplified update - in practice, you'd maintain a proper history
        updated_data = current_data.copy()
        
        # Update lag_1 feature if it exists
        for i, fname in enumerate(self.feature_names):
            if 'methane_lag_1' in fname:
                updated_data[i] = new_prediction
            elif 'methane_rolling_mean' in fname:
                # Simple approximation - in practice, maintain proper rolling window
                updated_data[i] = (updated_data[i] + new_prediction) / 2
        
        return updated_data
    
    def apply_noise_to_features(self, features, temp_noise=0.5, ph_noise=0.1, enable_noise=True):
        """Apply random noise to pH and temperature features for realistic forecasting"""
        if not enable_noise:
            return features
        
        # Create a copy to avoid modifying original features
        noisy_features = features.copy()
        
        # Add noise to temperature (with bounds checking)
        if temp_noise > 0:
            temp_variation = np.random.normal(0, temp_noise)
            new_temp = features['temperature_celsius'] + temp_variation
            # Clamp to realistic bounds
            noisy_features['temperature_celsius'] = np.clip(new_temp, 25.0, 50.0)
        
        # Add noise to pH (with bounds checking)
        if ph_noise > 0:
            ph_variation = np.random.normal(0, ph_noise)
            new_ph = features['ph'] + ph_variation
            # Clamp to realistic bounds
            noisy_features['ph'] = np.clip(new_ph, 5.0, 9.0)
        
        return noisy_features
    
    def forecast_24_hours_with_noise(self, current_features, model_choice='XGBoost', 
                                   temp_noise=0.5, ph_noise=0.1, enable_noise=True):
        """Generate 24-hour forecast with realistic parameter variations"""
        if model_choice == 'Gradient Boosting':
            model = self.gb_model
        elif model_choice == 'XGBoost':
            model = self.xgb_model
        else:
            model = self.lgb_model
        
        if model is None or self.scaler is None:
            st.error("Models not trained yet!")
            return None, None
        
        predictions = []
        timestamps = []
        current_time = datetime.now()
        
        # Generate 24-hour forecast (480 predictions for 3-minute intervals)
        n_predictions = 480  # 24 hours * 20 (3-minute intervals per hour)
        
        for step in range(n_predictions):
            # Apply noise to features for this prediction step
            if enable_noise:
                noisy_features = self.apply_noise_to_features(
                    current_features, temp_noise, ph_noise, enable_noise
                )
            else:
                noisy_features = current_features
            
            # Convert to model input format
            current_data = self.prepare_current_features(noisy_features)
            
            # Predict next value
            current_scaled = self.scaler.transform(current_data.reshape(1, -1))
            pred = model.predict(current_scaled)[0]
            predictions.append(pred)
            
            # Update timestamp
            next_time = current_time + timedelta(minutes=3 * (step + 1))
            timestamps.append(next_time)
            
            # Update lag features for next prediction
            current_data = self.update_lag_features(current_data, pred, step)
        
        return timestamps, predictions
    
    def validate_with_pinn(self, features):
        """Validate features using PINN physics"""
        if self.pinn_updater is None:
            return {"status": "PINN not available"}, features
        
        try:
            # Validate consistency
            warnings = self.pinn_updater.validate_feature_consistency(features)
            
            # Get optimal conditions
            optimal = self.pinn_updater.suggest_optimal_conditions(features)
            
            # Get corrective actions based on current conditions
            actions = self.get_corrective_actions(features, optimal)
            
            return {
                "status": "validated",
                "warnings": warnings,
                "optimal_conditions": optimal,
                "corrective_actions": actions
            }, optimal
            
        except Exception as e:
            return {"status": f"error: {str(e)}"}, features
    
    def get_corrective_actions(self, current_features, optimal_features):
        """Generate corrective actions by comparing current vs optimal"""
        actions = []
        
        # Temperature recommendations
        temp_diff = optimal_features['temperature_celsius'] - current_features['temperature_celsius']
        if abs(temp_diff) > 1.0:
            if temp_diff > 0:
                actions.append(f"Increase temperature by {temp_diff:.1f}°C to reach optimal 37°C")
            else:
                actions.append(f"Decrease temperature by {abs(temp_diff):.1f}°C to reach optimal 37°C")
        
        # pH recommendations
        ph_diff = optimal_features['ph'] - current_features['ph']
        if abs(ph_diff) > 0.2:
            if ph_diff > 0:
                actions.append(f"Increase pH by {ph_diff:.1f} (add alkalinity or reduce organic loading)")
            else:
                actions.append(f"Decrease pH by {abs(ph_diff):.1f} (increase organic loading or reduce alkalinity)")
        
        # Feed volume recommendations
        feed_diff = optimal_features['feed_volume_m3_day'] - current_features['feed_volume_m3_day']
        if abs(feed_diff) > 5.0:
            if feed_diff > 0:
                actions.append(f"Increase feed volume by {feed_diff:.1f} m³/day")
            else:
                actions.append(f"Reduce feed volume by {abs(feed_diff):.1f} m³/day to prevent overloading")
        
        # Solids content recommendations
        vs_diff = optimal_features['volatile_solids_percent'] - current_features['volatile_solids_percent']
        if abs(vs_diff) > 1.0:
            if vs_diff > 0:
                actions.append(f"Increase volatile solids content by {vs_diff:.1f}% (improve feed quality)")
            else:
                actions.append(f"Reduce volatile solids by {abs(vs_diff):.1f}% or increase dilution")
        
        # Retention time recommendations
        rt_diff = optimal_features['retention_time_days'] - current_features['retention_time_days']
        if abs(rt_diff) > 2.0:
            if rt_diff > 0:
                actions.append(f"Increase retention time by {rt_diff:.1f} days for better conversion")
            else:
                actions.append(f"Reduce retention time by {abs(rt_diff):.1f} days (system may be over-retained)")
        
        # Alkalinity recommendations
        alk_diff = optimal_features['alkalinity_mg_l'] - current_features['alkalinity_mg_l']
        if abs(alk_diff) > 500:
            if alk_diff > 0:
                actions.append(f"Increase alkalinity by {alk_diff:.0f} mg/L for better pH buffering")
            else:
                actions.append(f"Reduce alkalinity by {abs(alk_diff):.0f} mg/L to prevent inhibition")
        
        # Process stability checks
        if current_features['ph'] < 6.5:
            actions.append("URGENT: pH too low - risk of process failure. Reduce feeding rate immediately.")
        
        if current_features['temperature_celsius'] < 32:
            actions.append("WARNING: Temperature too low for optimal biogas production. Increase heating.")
        
        if current_features['temperature_celsius'] > 42:
            actions.append("WARNING: Temperature too high - may inhibit methanogens. Reduce heating.")
        
        # Loading rate check
        loading_rate = current_features['feed_volume_m3_day'] * current_features['volatile_solids_percent'] / 100
        if loading_rate > 3.5:
            actions.append("HIGH LOADING: Reduce feed rate or increase reactor volume to prevent acid buildup")
        
        if not actions:
            actions.append("Current conditions are near optimal - maintain steady operation")
        
        return actions


# Initialize the forecasting system
@st.cache_resource
def load_forecasting_system():
    system = BiogasForecastingSystem()
    system.load_models()
    return system

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("Biogas Methane Production Forecasting System")
    st.markdown("*Physics Informed Neural Network + Gradient Boosting for 24-Hour Forecasting*")
    
    # Initialize system
    forecasting_system = load_forecasting_system()
    
    # Sidebar for navigation
    st.sidebar.title("Control Panel")
    app_mode = st.sidebar.selectbox(
        "Choose Application Mode",
        ["Home", "Model Training", "24-Hour Forecast", "PINN Validation", "Analysis Dashboard"]
    )
    
    if app_mode == "Home":
        show_home_page()
    
    elif app_mode == "Model Training":
        show_model_training(forecasting_system)
    
    elif app_mode == "24-Hour Forecast":
        show_forecasting_page(forecasting_system)
    
    elif app_mode == "PINN Validation":
        show_pinn_validation(forecasting_system)
    
    elif app_mode == "Analysis Dashboard":
        show_analysis_dashboard(forecasting_system)

def show_home_page():
    """Show home page with system overview"""
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("	 System Overview")
        st.markdown("""
        This advanced forecasting system combines:
        
        **	 Physics Informed Neural Network (PINN)**
        - Incorporates biogas digestion physics laws
        - Validates feature consistency
        - Suggests optimal operating conditions
        
        **	 Gradient Boosting Models**
        - Traditional ML for time series forecasting
        - XGBoost, LightGBM, and Scikit-learn implementations
        - 24-hour ahead methane production predictions
        
        **	 Real-time Integration**
        - Physics-validated feature updates
        - Live operational parameter monitoring
        - Automated corrective action suggestions
        """)
    
    with col2:
        st.header("	 Quick Start")
        st.markdown("""
        1. **Train Models** 	
           - Load biogas dataset
           - Train ML models
           
        2. **Input Parameters** 	
           - Set current operating conditions
           - Validate with PINN
           
        3. **Generate Forecast** 	
           - 24-hour methane prediction
           - Visual analysis
           
        4. **Monitor & Optimize** 	
           - Track performance
           - Apply recommendations
        """)
    
    # Key metrics display
    st.header("🔑 Key Features")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    
    with metrics_col1:
        st.metric(
            label="Physics Laws",
            value="6",
            help="Monod kinetics, Arrhenius, pH inhibition, etc."
        )
    
    with metrics_col2:
        st.metric(
            label="ML Models",
            value="3",
            help="Gradient Boosting, XGBoost, LightGBM"
        )
    
    with metrics_col3:
        st.metric(
            label="Forecast Horizon",
            value="24h",
            help="480 predictions at 3-minute intervals"
        )
    
    with metrics_col4:
        st.metric(
            label="Features",
            value="18+",
            help="Operational parameters + engineered features"
        )
    
    # Dataset info
    st.header("	 Dataset Information")
    try:
        df = pd.read_csv('biogas_digestor_dataset.csv')
        
        info_col1, info_col2, info_col3 = st.columns(3)
        
        with info_col1:
            st.metric("Total Samples", f"{len(df):,}")
        
        with info_col2:
            st.metric("Time Span", "2 days")
        
        with info_col3:
            st.metric("Interval", "3 minutes")
        
        # Show sample data
        st.subheader("Sample Data")
        st.dataframe(df.head(10))
        
    except FileNotFoundError:
        st.error("Dataset not found. Please ensure 'biogas_digestor_dataset.csv' is available.")

def show_model_training(forecasting_system):
    """Show model training interface"""
    
    st.header("	 Model Training & Evaluation")
    
    try:
        # Load and display dataset info
        df = pd.read_csv('biogas_digestor_dataset.csv')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("	 Dataset Overview")
            st.write(f"**Total samples:** {len(df):,}")
            st.write(f"**Features:** {len(df.columns)-2}")  # Exclude timestamp and target
            st.write(f"**Target:** methane_production_m3_day")
            
            # Show data distribution
            fig = px.histogram(df, x='methane_production_m3_day', 
                             title="Methane Production Distribution",
                             nbins=50)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("🏋	 Training Configuration")
            
            # Training parameters
            test_size = st.slider("Test Size (%)", 10, 40, 20)
            n_estimators = st.slider("Number of Estimators", 50, 500, 200)
            learning_rate = st.selectbox("Learning Rate", [0.01, 0.05, 0.1, 0.2], index=2)
            max_depth = st.slider("Max Depth", 3, 10, 6)
        
        # Train models button
        if st.button("	 Train Models", type="primary"):
            with st.spinner("Training models... This may take a few minutes."):
                
                # Prepare data
                df_prepared = forecasting_system.prepare_data(df)
                st.success(f"	 Data prepared: {len(df_prepared)} samples after feature engineering")
                
                # Update model parameters
                for model in [forecasting_system.gb_model, forecasting_system.xgb_model, forecasting_system.lgb_model]:
                    if hasattr(model, 'n_estimators'):
                        model.n_estimators = n_estimators
                    if hasattr(model, 'learning_rate'):
                        model.learning_rate = learning_rate
                    if hasattr(model, 'max_depth'):
                        model.max_depth = max_depth
                
                # Train models
                results, X_test, y_test = forecasting_system.train_ml_models(df_prepared)
                
                if results:
                    st.success("	 Models trained successfully!")
                    
                    # Display results
                    st.subheader("	 Model Performance")
                    
                    results_df = pd.DataFrame(results).T
                    st.dataframe(results_df.round(4))
                    
                    # Visualize performance
                    fig = go.Figure()
                    
                    models = list(results.keys())
                    rmse_values = [results[model]['RMSE'] for model in models]
                    mae_values = [results[model]['MAE'] for model in models]
                    r2_values = [results[model]['R²'] for model in models]
                    
                    fig.add_trace(go.Bar(name='RMSE', x=models, y=rmse_values))
                    fig.add_trace(go.Bar(name='MAE', x=models, y=mae_values))
                    
                    fig.update_layout(
                        title="Model Performance Comparison",
                        xaxis_title="Models",
                        yaxis_title="Error Metrics",
                        barmode='group'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # R² comparison
                    fig_r2 = px.bar(x=models, y=r2_values, 
                                   title="R² Score Comparison",
                                   labels={'y': 'R² Score', 'x': 'Models'})
                    st.plotly_chart(fig_r2, use_container_width=True)
                    
                    # Save models
                    st.info("	 Models are automatically saved for forecasting use.")
                    
    except FileNotFoundError:
        st.error("	 Dataset not found. Please ensure 'biogas_digestor_dataset.csv' is available.")
    except Exception as e:
        st.error(f"	 Error during training: {str(e)}")

def show_forecasting_page(forecasting_system):
    """Show 24-hour forecasting interface"""
    
    st.header("	 24-Hour Methane Production Forecast")
    
    # Check if models are trained
    if forecasting_system.gb_model is None:
        st.warning("		 Models not trained yet. Please go to 'Model Training' first.")
        return
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("	 Current Operating Conditions")
        
        # Input current operational parameters
        features = {}
        
        # Physical parameters
        st.write("**	 Feed Characteristics**")
        features['feed_volume_m3_day'] = st.slider(
            "Feed Volume (m³/day)", 
            min_value=10.0, max_value=100.0, value=50.0, step=1.0
        )
        features['total_solids_percent'] = st.slider(
            "Total Solids (%)", 
            min_value=1.0, max_value=20.0, value=10.0, step=0.1
        )
        features['volatile_solids_percent'] = st.slider(
            "Volatile Solids (%)", 
            min_value=1.0, max_value=15.0, value=8.0, step=0.1
        )
        features['cod_mg_l'] = st.slider(
            "COD (mg/L)", 
            min_value=5000, max_value=45000, value=25000, step=1000
        )
        
        st.write("**🌡	 Environmental Parameters**")
        features['temperature_celsius'] = st.slider(
            "Temperature (°C)", 
            min_value=25.0, max_value=50.0, value=37.0, step=0.1
        )
        features['ph'] = st.slider(
            "pH", 
            min_value=5.0, max_value=9.0, value=7.0, step=0.1
        )
        features['alkalinity_mg_l'] = st.slider(
            "Alkalinity (mg/L)", 
            min_value=1000, max_value=6000, value=3000, step=100
        )
        
        st.write("**⚙	 Operational Parameters**")
        features['retention_time_days'] = st.slider(
            "Retention Time (days)", 
            min_value=5.0, max_value=40.0, value=20.0, step=0.5
        )
        features['mixing_intensity_rpm'] = st.slider(
            "Mixing Intensity (RPM)", 
            min_value=5.0, max_value=35.0, value=15.0, step=0.5
        )
        
        # Model selection
        model_choice = st.selectbox(
            "	 Select Forecasting Model",
            ["XGBoost", "Gradient Boosting", "LightGBM"]
        )
        
        # Noise variation controls
        st.write("**🎛️ Forecasting Variations**")
        st.markdown("*Add realistic noise to simulate natural parameter fluctuations*")
        
        temp_noise = st.slider(
            "Temperature Variation (±°C)", 
            min_value=0.0, max_value=3.0, value=0.5, step=0.1,
            help="Random temperature fluctuations during forecasting (±°C)"
        )
        
        ph_noise = st.slider(
            "pH Variation (±units)", 
            min_value=0.0, max_value=0.5, value=0.1, step=0.01,
            help="Random pH fluctuations during forecasting (±pH units)"
        )
        
        enable_noise = st.checkbox(
            "Enable Parameter Variations", 
            value=True,
            help="Apply random noise to pH and temperature during forecasting"
        )
        
        # Generate forecast button
        if st.button("	 Generate 24-Hour Forecast", type="primary"):
            
            with st.spinner("Generating forecast..."):
                
                # PINN validation first
                if forecasting_system.pinn_updater:
                    validation_result, optimal_features = forecasting_system.validate_with_pinn(features)
                    
                    if validation_result["warnings"]:
                        st.warning("		 PINN detected potential issues:")
                        for warning in validation_result["warnings"].values():
                            st.write(f"• {warning}")
                
                # Generate forecast with noise variations
                timestamps, predictions = forecasting_system.forecast_24_hours_with_noise(
                    features, model_choice, temp_noise, ph_noise, enable_noise
                )
                
                if timestamps and predictions:
                    # Store in session state for visualization
                    st.session_state.forecast_timestamps = timestamps
                    st.session_state.forecast_predictions = predictions
                    st.session_state.forecast_model = model_choice
                    
                    st.success(f"	 24-hour forecast generated using {model_choice}!")
                else:
                    st.error("	 Failed to generate forecast")
    
    with col2:
        st.subheader("	 Forecast Visualization")
        
        # Show forecast if available
        if hasattr(st.session_state, 'forecast_predictions'):
            
            timestamps = st.session_state.forecast_timestamps
            predictions = st.session_state.forecast_predictions
            model_name = st.session_state.forecast_model
            
            # Create forecast plot
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=timestamps,
                y=predictions,
                mode='lines',
                name=f'{model_name} Forecast',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(
                title=f"24-Hour Methane Production Forecast ({model_name})",
                xaxis_title="Time",
                yaxis_title="Methane Production (m³/day)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Forecast statistics
            st.subheader("	 Forecast Statistics")
            
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            
            with stats_col1:
                st.metric("Average", f"{np.mean(predictions):.2f} m³/day")
            
            with stats_col2:
                st.metric("Maximum", f"{np.max(predictions):.2f} m³/day")
            
            with stats_col3:
                st.metric("Minimum", f"{np.min(predictions):.2f} m³/day")
            
            with stats_col4:
                st.metric("Std Dev", f"{np.std(predictions):.2f} m³/day")
            
            # Hourly aggregation
            st.subheader("📅 Hourly Production Summary")
            
            # Group predictions by hour
            df_forecast = pd.DataFrame({
                'timestamp': timestamps,
                'methane_production': predictions
            })
            df_forecast['hour'] = pd.to_datetime(df_forecast['timestamp']).dt.hour
            hourly_avg = df_forecast.groupby('hour')['methane_production'].mean().reset_index()
            
            fig_hourly = px.bar(
                hourly_avg, 
                x='hour', 
                y='methane_production',
                title="Average Hourly Methane Production",
                labels={'methane_production': 'Methane Production (m³/day)', 'hour': 'Hour of Day'}
            )
            
            st.plotly_chart(fig_hourly, use_container_width=True)
            
            # Download forecast data
            st.subheader("	 Export Forecast")
            
            csv_data = df_forecast.to_csv(index=False)
            st.download_button(
                label="	 Download Forecast CSV",
                data=csv_data,
                file_name=f"methane_forecast_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
        
        else:
            st.info("	 Configure operating conditions and generate forecast to see results here.")

def show_pinn_validation(forecasting_system):
    """Show PINN validation interface"""
    
    st.header("	 PINN Physics Validation & Optimization")
    
    if not PINN_AVAILABLE or forecasting_system.pinn_updater is None:
        st.error("	 PINN not available. Please check physics_informed_neural_network.py and model files.")
        return
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("⚗	 Current Operating Conditions")
        
        # Input parameters
        features = {}
        features['feed_volume_m3_day'] = st.slider("Feed Volume (m³/day)", min_value=10.0, max_value=100.0, value=50.0, step=1.0)
        features['total_solids_percent'] = st.slider("Total Solids (%)", min_value=1.0, max_value=20.0, value=10.0, step=0.1)
        features['volatile_solids_percent'] = st.slider("Volatile Solids (%)", min_value=1.0, max_value=15.0, value=8.0, step=0.1)
        features['cod_mg_l'] = st.slider("COD (mg/L)", min_value=5000, max_value=45000, value=25000, step=1000)
        features['temperature_celsius'] = st.slider("Temperature (°C)", min_value=25.0, max_value=50.0, value=37.0, step=0.1)
        features['ph'] = st.slider("pH", min_value=5.0, max_value=9.0, value=7.0, step=0.1)
        features['alkalinity_mg_l'] = st.slider("Alkalinity (mg/L)", min_value=1000, max_value=6000, value=3000, step=100)
        features['retention_time_days'] = st.slider("Retention Time (days)", min_value=5.0, max_value=40.0, value=20.0, step=0.5)
        features['mixing_intensity_rpm'] = st.slider("Mixing Intensity (RPM)", min_value=5.0, max_value=35.0, value=15.0, step=0.5)
        
        if st.button("	 Validate with PINN", type="primary"):
            validation_result, optimal_features = forecasting_system.validate_with_pinn(features)
            
            st.session_state.validation_result = validation_result
            st.session_state.optimal_features = optimal_features
            st.session_state.current_features = features
    
    with col2:
        st.subheader("	 Validation Results")
        
        if hasattr(st.session_state, 'validation_result'):
            result = st.session_state.validation_result
            optimal = st.session_state.optimal_features
            current = st.session_state.current_features
            
            # Consistency warnings
            if "warnings" in result and result["warnings"]:
                st.error("		 Physics Consistency Issues:")
                for key, warning in result["warnings"].items():
                    st.write(f"• **{key}**: {warning}")
            else:
                st.success("	 All parameters are physically consistent!")
            
            # Optimal conditions
            if "optimal_conditions" in result:
                st.subheader("	 Optimal Conditions")
                
                comparison_data = []
                for param in current.keys():
                    if param in optimal:
                        current_val = current[param]
                        optimal_val = optimal[param]
                        change = optimal_val - current_val
                        change_pct = (change / current_val) * 100 if current_val != 0 else 0
                        
                        comparison_data.append({
                            'Parameter': param.replace('_', ' ').title(),
                            'Current': f"{current_val:.2f}",
                            'Optimal': f"{optimal_val:.2f}",
                            'Change': f"{change:+.2f}",
                            'Change (%)': f"{change_pct:+.1f}%"
                        })
                
                comparison_df = pd.DataFrame(comparison_data)
                st.dataframe(comparison_df, use_container_width=True)
            
            # Corrective actions
            if "corrective_actions" in result:
                st.subheader("	 Recommended Actions")
                for action in result["corrective_actions"]:
                    st.write(f"• {action}")
        
        else:
            st.info("	 Enter operating conditions and validate to see results here.")

def show_analysis_dashboard(forecasting_system):
    """Show analysis dashboard"""
    
    st.header("	 Analysis Dashboard")
    
    try:
        df = pd.read_csv('biogas_digestor_dataset.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Time series analysis
        st.subheader("	 Historical Data Analysis")
        
        # Main time series plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=df['methane_production_m3_day'],
            mode='lines',
            name='Methane Production',
            line=dict(color='blue', width=1)
        ))
        
        fig.update_layout(
            title="Historical Methane Production",
            xaxis_title="Time",
            yaxis_title="Methane Production (m³/day)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Correlation analysis
        st.subheader("	 Feature Correlation Analysis")
        
        # Select key features for correlation
        key_features = [
            'methane_production_m3_day', 'feed_volume_m3_day', 'temperature_celsius',
            'ph', 'volatile_solids_percent', 'cod_mg_l', 'retention_time_days'
        ]
        
        corr_matrix = df[key_features].corr()
        
        fig_corr = px.imshow(
            corr_matrix,
            aspect="auto",
            color_continuous_scale="RdBu_r",
            title="Feature Correlation Matrix"
        )
        
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Distribution analysis
        st.subheader("	 Parameter Distributions")
        
        # Select parameter to analyze
        param_choice = st.selectbox(
            "Select Parameter for Distribution Analysis",
            ['methane_production_m3_day', 'temperature_celsius', 'ph', 'feed_volume_m3_day']
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histogram
            fig_hist = px.histogram(
                df, 
                x=param_choice,
                title=f"Distribution of {param_choice.replace('_', ' ').title()}",
                nbins=30
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Box plot
            fig_box = px.box(
                df, 
                y=param_choice,
                title=f"Box Plot of {param_choice.replace('_', ' ').title()}"
            )
            st.plotly_chart(fig_box, use_container_width=True)
        
        # Performance metrics
        st.subheader("	 System Performance Metrics")
        
        metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
        
        with metrics_col1:
            avg_production = df['methane_production_m3_day'].mean()
            st.metric("Average Production", f"{avg_production:.2f} m³/day")
        
        with metrics_col2:
            max_production = df['methane_production_m3_day'].max()
            st.metric("Maximum Production", f"{max_production:.2f} m³/day")
        
        with metrics_col3:
            stability = df['methane_production_m3_day'].std()
            st.metric("Production Stability (σ)", f"{stability:.2f} m³/day")
        
        with metrics_col4:
            efficiency = (avg_production / max_production) * 100
            st.metric("Average Efficiency", f"{efficiency:.1f}%")
        
    except FileNotFoundError:
        st.error("	 Dataset not found for analysis.")
    except Exception as e:
        st.error(f"	 Error in analysis: {str(e)}")

if __name__ == "__main__":
    main()
