"""
Physics Informed Neural Network (PINN) for Biogas Digestor System

This PINN incorporates the fundamental physics and biochemistry of anaerobic digestion
to ensure scientifically logical feature updates and predictions.

Key Physics Incorporated:
1. Monod kinetics for microbial growth
2. First-order kinetics for biodegradation
3. Temperature dependency (Arrhenius equation)
4. pH inhibition functions
5. Mass balance constraints
6. Thermodynamic relationships
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class BiogasPhysicsLaws:
    """
    Encapsulates the fundamental physics and biochemistry of biogas production
    """
    
    @staticmethod
    def monod_kinetics(substrate_conc: torch.Tensor, k_s: float = 1000.0, mu_max: float = 0.5) -> torch.Tensor:
        """
        Monod kinetics for microbial growth rate
        μ = μ_max * S / (K_s + S)
        """
        return mu_max * substrate_conc / (k_s + substrate_conc)
    
    @staticmethod
    def arrhenius_temperature(temp_celsius: torch.Tensor, 
                            temp_opt: float = 37.0, 
                            activation_energy: float = 69000.0) -> torch.Tensor:
        """
        Arrhenius equation for temperature dependency
        k(T) = k_ref * exp(-Ea/R * (1/T - 1/T_ref))
        """
        R = 8.314  # Gas constant J/(mol·K)
        temp_kelvin = temp_celsius + 273.15
        temp_opt_kelvin = temp_opt + 273.15
        
        return torch.exp(-activation_energy / R * (1/temp_kelvin - 1/temp_opt_kelvin))
    
    @staticmethod
    def ph_inhibition(ph: torch.Tensor, 
                     ph_opt: float = 7.0, 
                     ph_lower: float = 6.5, 
                     ph_upper: float = 8.0) -> torch.Tensor:
        """
        pH inhibition function - Gaussian-like response with sharp drops
        """
        # Optimal range
        optimal_mask = (ph >= ph_lower) & (ph <= ph_upper)
        
        # Lower inhibition (acidic)
        lower_mask = ph < ph_lower
        lower_inhibition = torch.exp(-((ph - ph_lower) / 0.5) ** 2)
        
        # Upper inhibition (basic)
        upper_mask = ph > ph_upper
        upper_inhibition = torch.exp(-((ph - ph_upper) / 0.5) ** 2)
        
        # Combine
        inhibition = torch.ones_like(ph)
        inhibition = torch.where(lower_mask, lower_inhibition, inhibition)
        inhibition = torch.where(upper_mask, upper_inhibition, inhibition)
        
        return inhibition
    
    @staticmethod
    def loading_rate_inhibition(loading_rate: torch.Tensor, 
                              optimal_loading: float = 2.0,
                              max_loading: float = 4.0) -> torch.Tensor:
        """
        Organic loading rate inhibition - optimal curve with overloading penalty
        """
        # Optimal range
        optimal_factor = torch.minimum(loading_rate / optimal_loading, 
                                     torch.ones_like(loading_rate))
        
        # Overloading inhibition
        overload_mask = loading_rate > optimal_loading
        overload_factor = torch.exp(-(loading_rate - optimal_loading) / 
                                  (max_loading - optimal_loading))
        
        result = torch.where(overload_mask, overload_factor, optimal_factor)
        
        return result
    
    @staticmethod
    def alkalinity_buffering(ph: torch.Tensor, alkalinity: torch.Tensor) -> torch.Tensor:
        """
        Alkalinity provides buffering capacity - relationship between pH and alkalinity
        """
        # Higher alkalinity provides better pH stability
        buffering_capacity = torch.tanh(alkalinity / 2000.0)  # Normalize around 2000 mg/L
        ph_deviation = torch.abs(ph - 7.0)
        
        return buffering_capacity * torch.exp(-ph_deviation)
    
    @staticmethod
    def first_order_kinetics(substrate: torch.Tensor, 
                           rate_constant: float = 0.1,
                           retention_time: torch.Tensor = None) -> torch.Tensor:
        """
        First-order kinetics for substrate degradation
        S_out = S_in * exp(-k * HRT)
        """
        if retention_time is None:
            retention_time = torch.ones_like(substrate, device=substrate.device) * 20.0  # Default 20 days
        
        return substrate * (1 - torch.exp(-rate_constant * retention_time))


class PhysicsInformedNN(nn.Module):
    """
    Physics Informed Neural Network for Biogas Digestor System
    """
    
    def __init__(self, 
                 input_dim: int = 18,
                 hidden_dims: List[int] = [128, 256, 256, 128],
                 output_dim: int = 1,
                 physics_weight: float = 1.0,
                 device: str = None):
        
        super(PhysicsInformedNN, self).__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.physics_weight = physics_weight
        self.device = device if device is not None else ('mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu')
        self.physics_laws = BiogasPhysicsLaws()
        
        # Neural network layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Physics-aware output scaling
        self.output_scaling = nn.Parameter(torch.tensor([50.0]))  # Typical methane production scale
        
        # Feature importance weights (learnable)
        self.feature_weights = nn.Parameter(torch.ones(input_dim))
        
        self.to(self.device)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with physics constraints
        """
        # Apply learned feature importance
        weighted_x = x * torch.softmax(self.feature_weights, dim=0)
        
        # Neural network prediction
        nn_output = self.network(weighted_x)
        
        # Apply physics-based scaling
        output = torch.relu(nn_output * self.output_scaling)  # Ensure positive methane production
        
        return output
    
    def physics_loss(self, x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """
        Calculate physics-based loss terms
        """
        batch_size = x.shape[0]
        
        # Extract features (assuming specific order from dataset)
        feed_volume = x[:, 0]
        total_solids = x[:, 1]
        volatile_solids = x[:, 2]
        cod = x[:, 3]
        temperature = x[:, 4]
        ph = x[:, 5]
        alkalinity = x[:, 6]
        retention_time = x[:, 7]
        mixing_intensity = x[:, 8]
        loading_rate = x[:, -1] if x.shape[1] > 17 else torch.ones(batch_size, device=x.device) * 2.0
        
        # Physics constraints
        losses = []
        
        # 1. Temperature dependency (Arrhenius)
        temp_factor = self.physics_laws.arrhenius_temperature(temperature)
        temp_loss = torch.mean((y_pred - y_pred * temp_factor) ** 2)
        losses.append(temp_loss)
        
        # 2. pH inhibition
        ph_factor = self.physics_laws.ph_inhibition(ph)
        ph_loss = torch.mean((y_pred - y_pred * ph_factor) ** 2)
        losses.append(ph_loss)
        
        # 3. Loading rate optimization
        loading_factor = self.physics_laws.loading_rate_inhibition(loading_rate)
        loading_loss = torch.mean((y_pred - y_pred * loading_factor) ** 2)
        losses.append(loading_loss)
        
        # 4. Substrate quality constraint (VS should be related to COD)
        vs_cod_ratio = volatile_solids / (cod / 1000.0)  # Normalize COD
        vs_cod_consistency = torch.mean((vs_cod_ratio - 0.8) ** 2)  # Expected ratio ~0.8
        losses.append(vs_cod_consistency)
        
        # 5. Mass balance constraint (simplified)
        # Methane yield should be proportional to VS destroyed
        theoretical_yield = volatile_solids * feed_volume * 0.35  # ~350 L CH4/kg VS
        mass_balance_loss = torch.mean((y_pred - theoretical_yield) ** 2) / 1000.0
        losses.append(mass_balance_loss)
        
        # 6. Retention time constraint (first-order kinetics)
        retention_factor = 1 - torch.exp(-0.1 * retention_time)
        retention_loss = torch.mean((y_pred - y_pred * retention_factor) ** 2)
        losses.append(retention_loss)
        
        return torch.sum(torch.stack(losses))
    
    def total_loss(self, x: torch.Tensor, y_true: torch.Tensor, y_pred: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Calculate total loss including physics constraints
        """
        # Data loss (MSE)
        data_loss = nn.MSELoss()(y_pred, y_true)
        
        # Physics loss
        physics_loss = self.physics_loss(x, y_pred)
        
        # Total loss
        total = data_loss + self.physics_weight * physics_loss
        
        return {
            'total': total,
            'data': data_loss,
            'physics': physics_loss
        }


class BiogasPINNTrainer:
    """
    Trainer class for the Physics Informed Neural Network
    """
    
    def __init__(self, 
                 model: PhysicsInformedNN,
                 learning_rate = 0.001,
                 device: str = None):
        
        self.model = model
        self.device = device if device is not None else model.device
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=20
        )
        
        # Loss history
        self.loss_history = {
            'train_total': [],
            'train_data': [],
            'train_physics': [],
            'val_total': [],
            'val_data': [],
            'val_physics': []
        }
    
    def train_epoch(self, train_loader) -> Dict[str, float]:
        """
        Train for one epoch
        """
        self.model.train()
        epoch_losses = {'total': 0, 'data': 0, 'physics': 0}
        num_batches = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            y_pred = self.model(batch_x)
            
            # Calculate losses
            losses = self.model.total_loss(batch_x, batch_y, y_pred)
            
            # Backward pass
            losses['total'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Accumulate losses
            for key in epoch_losses:
                epoch_losses[key] += losses[key].item()
            num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate_epoch(self, val_loader) -> Dict[str, float]:
        """
        Validate for one epoch
        """
        self.model.eval()
        epoch_losses = {'total': 0, 'data': 0, 'physics': 0}
        num_batches = 0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                
                # Forward pass
                y_pred = self.model(batch_x)
                
                # Calculate losses
                losses = self.model.total_loss(batch_x, batch_y, y_pred)
                
                # Accumulate losses
                for key in epoch_losses:
                    epoch_losses[key] += losses[key].item()
                num_batches += 1
        
        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def train(self, 
              train_loader, 
              val_loader, 
              epochs: int = 100,
              patience: int = 30) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping
        """
        best_val_loss = float('inf')
        patience_counter = 0
        
        print("Starting Physics Informed Neural Network Training...")
        print(f"Physics weight: {self.model.physics_weight}")
        
        for epoch in range(epochs):
            # Training
            train_losses = self.train_epoch(train_loader)
            
            # Validation
            val_losses = self.validate_epoch(val_loader)
            
            # Learning rate scheduling
            self.scheduler.step(val_losses['total'])
            
            # Store losses
            self.loss_history['train_total'].append(train_losses['total'])
            self.loss_history['train_data'].append(train_losses['data'])
            self.loss_history['train_physics'].append(train_losses['physics'])
            self.loss_history['val_total'].append(val_losses['total'])
            self.loss_history['val_data'].append(val_losses['data'])
            self.loss_history['val_physics'].append(val_losses['physics'])
            
            # Early stopping
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                patience_counter = 0
                # Save best model
                torch.save(self.model.state_dict(), 'best_pinn_model.pt')
            else:
                patience_counter += 1
            
            # Print progress
            if epoch % 10 == 0 or patience_counter == 0:
                print(f"Epoch {epoch:3d} | "
                      f"Train Loss: {train_losses['total']:.4f} "
                      f"(Data: {train_losses['data']:.4f}, Physics: {train_losses['physics']:.4f}) | "
                      f"Val Loss: {val_losses['total']:.4f} "
                      f"(Data: {val_losses['data']:.4f}, Physics: {val_losses['physics']:.4f})")
            
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch}")
                break
        
        # Load best model
        self.model.load_state_dict(torch.load('best_pinn_model.pt'))
        
        return self.loss_history
    
    def plot_training_history(self):
        """
        Plot training history
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Total loss
        axes[0].plot(self.loss_history['train_total'], label='Train', alpha=0.8)
        axes[0].plot(self.loss_history['val_total'], label='Validation', alpha=0.8)
        axes[0].set_title('Total Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Data loss
        axes[1].plot(self.loss_history['train_data'], label='Train', alpha=0.8)
        axes[1].plot(self.loss_history['val_data'], label='Validation', alpha=0.8)
        axes[1].set_title('Data Loss (MSE)')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Physics loss
        axes[2].plot(self.loss_history['train_physics'], label='Train', alpha=0.8)
        axes[2].plot(self.loss_history['val_physics'], label='Validation', alpha=0.8)
        axes[2].set_title('Physics Loss')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Loss')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()


class BiogasFeatureUpdater:
    """
    Uses PINN to provide logical and scientific feature value updates
    """
    
    def __init__(self, trained_model: PhysicsInformedNN):
        self.model = trained_model
        self.model.eval()
        self.physics_laws = BiogasPhysicsLaws()
    
    def suggest_optimal_conditions(self, 
                                 current_features: Dict[str, float],
                                 target_methane: float = None) -> Dict[str, float]:
        """
        Suggest optimal operating conditions for desired methane production
        """
        # Define optimization bounds based on dataset documentation
        bounds = {
            'feed_volume_m3_day': (13.38, 85.67),
            'total_solids_percent': (2.0, 15.0),
            'volatile_solids_percent': (1.5, 12.0),  # 70-85% of total solids
            'cod_mg_l': (5000, 40000),
            'temperature_celsius': (30.0, 45.0),
            'ph': (5.5, 8.5),
            'alkalinity_mg_l': (1000, 5000),
            'retention_time_days': (10.0, 35.0),
            'mixing_intensity_rpm': (5.0, 30.0)
        }
        
        # Current feature tensor
        feature_names = list(current_features.keys())
        current_tensor = torch.tensor([current_features[name] for name in feature_names], 
                                    dtype=torch.float32).unsqueeze(0)
        
        # If target not specified, optimize for maximum production
        if target_methane is None:
            return self._optimize_for_maximum(current_features, bounds)
        else:
            return self._optimize_for_target(current_features, target_methane, bounds)
    
    def _optimize_for_maximum(self, 
                            current_features: Dict[str, float], 
                            bounds: Dict[str, Tuple[float, float]]) -> Dict[str, float]:
        """
        Optimize for maximum methane production
        """
        optimal_conditions = {}
        
        # Physics-based optimal values
        optimal_conditions['temperature_celsius'] = 37.0  # Mesophilic optimum
        optimal_conditions['ph'] = 7.0  # Neutral pH optimum
        optimal_conditions['alkalinity_mg_l'] = 3000.0  # Good buffering capacity
        
        # Calculate optimal loading based on current VS content
        vs_content = current_features.get('volatile_solids_percent', 8.0)
        optimal_loading = 60.0  # From documentation: optimal around 60 m³/day
        optimal_conditions['feed_volume_m3_day'] = optimal_loading
        
        # Optimize solids content for methane potential
        optimal_conditions['total_solids_percent'] = 12.0  # Higher is generally better
        optimal_conditions['volatile_solids_percent'] = optimal_conditions['total_solids_percent'] * 0.8
        
        # COD should be consistent with VS
        optimal_conditions['cod_mg_l'] = optimal_conditions['volatile_solids_percent'] * 1000 * 1.5
        
        # Retention time for good conversion
        optimal_conditions['retention_time_days'] = 25.0  # Good balance
        
        # Moderate mixing for good contact
        optimal_conditions['mixing_intensity_rpm'] = 15.0
        
        # Apply bounds
        for key, value in optimal_conditions.items():
            if key in bounds:
                min_val, max_val = bounds[key]
                optimal_conditions[key] = max(min_val, min(max_val, value))
        
        return optimal_conditions
    
    def validate_feature_consistency(self, features: Dict[str, float]) -> Dict[str, str]:
        """
        Validate that feature values are physically consistent
        """
        warnings = {}
        
        # VS should be less than TS
        if features['volatile_solids_percent'] > features['total_solids_percent']:
            warnings['vs_ts_ratio'] = "Volatile solids cannot exceed total solids"
        
        # VS/TS ratio should be realistic (70-85%)
        vs_ts_ratio = features['volatile_solids_percent'] / features['total_solids_percent']
        if vs_ts_ratio < 0.6 or vs_ts_ratio > 0.9:
            warnings['vs_ts_ratio'] = f"VS/TS ratio ({vs_ts_ratio:.2f}) is outside typical range (0.6-0.9)"
        
        # COD and VS relationship
        expected_cod = features['volatile_solids_percent'] * 1000 * 1.4  # Approximate relationship
        cod_ratio = features['cod_mg_l'] / expected_cod
        if cod_ratio < 0.8 or cod_ratio > 2.0:
            warnings['cod_vs_ratio'] = f"COD/VS ratio ({cod_ratio:.2f}) may be inconsistent"
        
        # pH and alkalinity relationship
        if features['ph'] < 6.5 and features['alkalinity_mg_l'] > 3000:
            warnings['ph_alkalinity'] = "Low pH despite high alkalinity suggests process upset"
        
        # Loading rate calculation
        loading_rate = (features['feed_volume_m3_day'] * features['volatile_solids_percent'] / 100.0)
        if loading_rate > 4.0:
            warnings['overloading'] = f"High loading rate ({loading_rate:.2f} kg VS/m³/day) may cause inhibition"
        
        # Temperature range
        if features['temperature_celsius'] < 32 or features['temperature_celsius'] > 42:
            warnings['temperature'] = "Temperature outside optimal mesophilic range (32-42°C)"
        
        return warnings
    
    def suggest_corrective_actions(self, 
                                 current_features: Dict[str, float],
                                 current_methane: float) -> List[str]:
        """
        Suggest corrective actions based on current performance
        """
        actions = []
        
        # Low methane production
        if current_methane < 30.0:
            # Check pH
            if current_features['ph'] < 6.8:
                actions.append("Increase alkalinity or reduce feeding rate to raise pH")
            
            # Check temperature
            if current_features['temperature_celsius'] < 35:
                actions.append("Increase temperature to optimal range (37°C)")
            
            # Check loading
            loading = current_features['feed_volume_m3_day'] * current_features['volatile_solids_percent'] / 100
            if loading > 3.0:
                actions.append("Reduce organic loading rate to prevent inhibition")
            
            # Check retention time
            if current_features['retention_time_days'] < 15:
                actions.append("Increase retention time for better conversion")
        
        # High but unstable production
        elif current_methane > 70.0:
            if current_features['ph'] > 7.5:
                actions.append("Monitor for ammonia inhibition, consider reducing protein-rich feedstock")
            
            actions.append("Maintain current conditions but monitor for process stability")
        
        # Moderate production - optimization opportunities
        else:
            optimal = self.suggest_optimal_conditions(current_features)
            
            if abs(current_features['temperature_celsius'] - 37.0) > 2:
                actions.append(f"Adjust temperature toward 37°C (currently {current_features['temperature_celsius']:.1f}°C)")
            
            if abs(current_features['ph'] - 7.0) > 0.3:
                actions.append(f"Adjust pH toward 7.0 (currently {current_features['ph']:.2f})")
        
        if not actions:
            actions.append("Current conditions appear optimal - maintain steady operation")
        
        return actions
    
    def predict_impact(self, 
                      current_features: Dict[str, float],
                      proposed_changes: Dict[str, float]) -> Dict[str, float]:
        """
        Predict the impact of proposed feature changes
        """
        # Current prediction
        current_tensor = torch.tensor(list(current_features.values()), 
                                    dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            current_prediction = self.model(current_tensor).item()
        
        # Modified features
        modified_features = current_features.copy()
        modified_features.update(proposed_changes)
        
        modified_tensor = torch.tensor(list(modified_features.values()), 
                                     dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            modified_prediction = self.model(modified_tensor).item()
        
        return {
            'current_methane': current_prediction,
            'predicted_methane': modified_prediction,
            'change': modified_prediction - current_prediction,
            'percent_change': ((modified_prediction - current_prediction) / current_prediction) * 100
        }


def create_data_loaders(df: pd.DataFrame, 
                       feature_cols: List[str], 
                       target_col: str,
                       batch_size: int = 32,
                       train_ratio: float = 0.8) -> Tuple:
    """
    Create PyTorch data loaders for training
    """
    from torch.utils.data import DataLoader, TensorDataset
    
    # Prepare features and target
    X = df[feature_cols].values.astype(np.float32)
    y = df[target_col].values.astype(np.float32).reshape(-1, 1)
    
    # Time series split
    split_idx = int(len(df) * train_ratio)
    
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Convert to tensors
    X_train_tensor = torch.tensor(X_train)
    y_train_tensor = torch.tensor(y_train)
    X_val_tensor = torch.tensor(X_val)
    y_val_tensor = torch.tensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)  # No shuffle for time series
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, feature_cols


if __name__ == "__main__":
    # Example usage
    print("Physics Informed Neural Network for Biogas Digestor System")
    print("=" * 60)
    
    # Load and prepare data
    df = pd.read_csv('biogas_digestor_dataset.csv')
    
    # Define feature columns (excluding target and timestamp)
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'methane_production_m3_day']]
    target_col = 'methane_production_m3_day'
    
    # Create data loaders
    train_loader, val_loader, feature_names = create_data_loaders(
        df, feature_cols, target_col, batch_size=32
    )
    
    # Initialize model
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PhysicsInformedNN(
        input_dim=len(feature_cols),
        hidden_dims=[128, 256, 256, 128],
        output_dim=1,
        physics_weight=0.5,  # Balance between data and physics
        device=device
    )
    
    # Initialize trainer
    trainer = BiogasPINNTrainer(model, learning_rate=0.001, device=device)
    
    # Train model
    print(f"Training on device: {device}")
    loss_history = trainer.train(train_loader, val_loader, epochs=100, patience=20)
    
    # Plot training history
    trainer.plot_training_history()
    
    # Initialize feature updater
    feature_updater = BiogasFeatureUpdater(model)
    
    # Example feature update suggestions
    current_features = {
        'feed_volume_m3_day': 50.0,
        'total_solids_percent': 10.0,
        'volatile_solids_percent': 8.0,
        'cod_mg_l': 25000,
        'temperature_celsius': 35.0,
        'ph': 6.8,
        'alkalinity_mg_l': 2500,
        'retention_time_days': 20.0,
        'mixing_intensity_rpm': 12.0
    }
    
    print("\nCurrent Operating Conditions:")
    for key, value in current_features.items():
        print(f"{key}: {value}")
    
    # Get optimal conditions
    optimal = feature_updater.suggest_optimal_conditions(current_features)
    print("\nSuggested Optimal Conditions:")
    for key, value in optimal.items():
        print(f"{key}: {value:.2f}")
    
    # Validate consistency
    warnings = feature_updater.validate_feature_consistency(current_features)
    if warnings:
        print("\nPhysics Consistency Warnings:")
        for key, warning in warnings.items():
            print(f"- {warning}")
    
    # Get corrective actions
    actions = feature_updater.suggest_corrective_actions(current_features, 45.0)
    print("\nSuggested Corrective Actions:")
    for action in actions:
        print(f"- {action}")
    
    print("\nPINN training and feature updating system ready!")
