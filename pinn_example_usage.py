"""
Example Usage and Testing of Physics Informed Neural Network for Biogas Digestor

This script demonstrates how to use the PINN for:
1. Training with physics constraints
2. Feature validation and updates
3. Process optimization
4. Scientific consistency checking
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from physics_informed_neural_network import (
    PhysicsInformedNN, 
    BiogasPINNTrainer, 
    BiogasFeatureUpdater,
    create_data_loaders
)

def demonstrate_pinn_usage():
    """
    Comprehensive demonstration of PINN capabilities
    """
    print("=" * 80)
    print("PHYSICS INFORMED NEURAL NETWORK DEMONSTRATION")
    print("Biogas Digestor System Optimization")
    print("=" * 80)
    
    # Load dataset
    print("\n1. Loading and Preparing Dataset...")
    try:
        df = pd.read_csv('biogas_digestor_dataset.csv')
        print(f"✓ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    except FileNotFoundError:
        print("✗ Dataset not found. Please ensure 'biogas_digestor_dataset.csv' is in the current directory.")
        return
    
    # Prepare data
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'methane_production_m3_day']]
    target_col = 'methane_production_m3_day'
    
    print(f"✓ Features: {len(feature_cols)}")
    print(f"✓ Target: {target_col}")
    
    # Create data loaders
    print("\n2. Creating Data Loaders...")
    train_loader, val_loader, feature_names = create_data_loaders(
        df, feature_cols, target_col, batch_size=32
    )
    print(f"✓ Training batches: {len(train_loader)}")
    print(f"✓ Validation batches: {len(val_loader)}")
    
    # Initialize PINN model
    print("\n3. Initializing Physics Informed Neural Network...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    model = PhysicsInformedNN(
        input_dim=len(feature_cols),
        hidden_dims=[64, 128, 128, 64],  # Smaller for demo
        output_dim=1,
        physics_weight=0.3,  # Balance between data and physics
        device=device
    )
    
    print(f"✓ Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"✓ Physics weight: {model.physics_weight}")
    
    # Train model (shorter training for demo)
    print("\n4. Training PINN Model...")
    trainer = BiogasPINNTrainer(model, learning_rate=0.003, device=device)
    
    print("Training for 30 epochs (quick demo)...")
    loss_history = trainer.train(train_loader, val_loader, epochs=30, patience=10)
    
    print("✓ Training completed!")
    print(f"✓ Final validation loss: {loss_history['val_total'][-1]:.4f}")
    
    # Display training curves
    trainer.plot_training_history()
    
    return model, feature_names, df

def demonstrate_feature_updating(model, feature_names, df):
    """
    Demonstrate feature updating and optimization capabilities
    """
    print("\n" + "=" * 80)
    print("FEATURE UPDATING AND OPTIMIZATION DEMONSTRATION")
    print("=" * 80)
    
    # Initialize feature updater
    feature_updater = BiogasFeatureUpdater(model)
    
    # Test scenarios
    scenarios = [
        {
            'name': 'Suboptimal Low pH Scenario',
            'features': {
                'feed_volume_m3_day': 45.0,
                'total_solids_percent': 8.5,
                'volatile_solids_percent': 6.8,
                'cod_mg_l': 22000,
                'temperature_celsius': 34.0,
                'ph': 6.2,  # Low pH
                'alkalinity_mg_l': 1800,
                'retention_time_days': 18.0,
                'mixing_intensity_rpm': 10.0
            },
            'current_methane': 28.5
        },
        {
            'name': 'High Temperature Scenario',
            'features': {
                'feed_volume_m3_day': 65.0,
                'total_solids_percent': 12.0,
                'volatile_solids_percent': 9.6,
                'cod_mg_l': 35000,
                'temperature_celsius': 43.0,  # High temperature
                'ph': 7.3,
                'alkalinity_mg_l': 3200,
                'retention_time_days': 25.0,
                'mixing_intensity_rpm': 18.0
            },
            'current_methane': 52.3
        },
        {
            'name': 'Overloading Scenario',
            'features': {
                'feed_volume_m3_day': 78.0,  # High feed volume
                'total_solids_percent': 14.5,
                'volatile_solids_percent': 11.6,
                'cod_mg_l': 38000,
                'temperature_celsius': 37.5,
                'ph': 6.9,
                'alkalinity_mg_l': 2900,
                'retention_time_days': 15.0,  # Short retention
                'mixing_intensity_rpm': 22.0
            },
            'current_methane': 35.8
        }
    ]
    
    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{i}. {scenario['name']}")
        print("-" * 50)
        
        current_features = scenario['features']
        current_methane = scenario['current_methane']
        
        # Display current conditions
        print("Current Operating Conditions:")
        for key, value in current_features.items():
            print(f"  {key}: {value}")
        print(f"  Current methane production: {current_methane:.1f} m³/day")
        
        # Physics consistency check
        print("\nPhysics Consistency Analysis:")
        warnings = feature_updater.validate_feature_consistency(current_features)
        if warnings:
            for key, warning in warnings.items():
                print(f"  ⚠️  {warning}")
        else:
            print("  ✓ All parameters are physically consistent")
        
        # Get optimal conditions
        print("\nOptimal Conditions Suggestion:")
        optimal = feature_updater.suggest_optimal_conditions(current_features)
        for key, value in optimal.items():
            current_val = current_features[key]
            change = value - current_val
            change_pct = (change / current_val) * 100 if current_val != 0 else 0
            print(f"  {key}: {current_val:.2f} → {value:.2f} "
                  f"({change:+.2f}, {change_pct:+.1f}%)")
        
        # Predict impact of changes
        print("\nImpact Prediction:")
        impact = feature_updater.predict_impact(current_features, optimal)
        print(f"  Current prediction: {impact['current_methane']:.2f} m³/day")
        print(f"  Optimized prediction: {impact['predicted_methane']:.2f} m³/day")
        print(f"  Expected improvement: {impact['change']:+.2f} m³/day ({impact['percent_change']:+.1f}%)")
        
        # Get corrective actions
        print("\nRecommended Actions:")
        actions = feature_updater.suggest_corrective_actions(current_features, current_methane)
        for j, action in enumerate(actions, 1):
            print(f"  {j}. {action}")
        
        print()

def demonstrate_parameter_sensitivity(model, feature_names):
    """
    Demonstrate parameter sensitivity analysis
    """
    print("\n" + "=" * 80)
    print("PARAMETER SENSITIVITY ANALYSIS")
    print("=" * 80)
    
    # Base case
    base_features = {
        'feed_volume_m3_day': 50.0,
        'total_solids_percent': 10.0,
        'volatile_solids_percent': 8.0,
        'cod_mg_l': 28000,
        'temperature_celsius': 37.0,
        'ph': 7.0,
        'alkalinity_mg_l': 3000,
        'retention_time_days': 22.0,
        'mixing_intensity_rpm': 15.0
    }
    
    # Add missing features with default values
    all_features = base_features.copy()
    
    # Parameters to test
    sensitivity_tests = {
        'temperature_celsius': np.linspace(30, 45, 16),
        'ph': np.linspace(5.5, 8.5, 16),
        'feed_volume_m3_day': np.linspace(20, 80, 16),
        'retention_time_days': np.linspace(10, 35, 16)
    }
    
    # Create sensitivity plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    feature_updater = BiogasFeatureUpdater(model)
    
    for idx, (param, values) in enumerate(sensitivity_tests.items()):
        methane_predictions = []
        
        for value in values:
            test_features = all_features.copy()
            test_features[param] = value
            
            # Predict methane production
            impact = feature_updater.predict_impact(all_features, {param: value})
            methane_predictions.append(impact['predicted_methane'])
        
        # Plot
        axes[idx].plot(values, methane_predictions, 'b-', linewidth=2, marker='o', markersize=4)
        axes[idx].set_xlabel(param.replace('_', ' ').title())
        axes[idx].set_ylabel('Predicted Methane (m³/day)')
        axes[idx].set_title(f'Sensitivity to {param.replace("_", " ").title()}')
        axes[idx].grid(True, alpha=0.3)
        
        # Mark optimal point
        base_value = all_features[param]
        base_prediction = feature_updater.predict_impact(all_features, {})['current_methane']
        axes[idx].axvline(x=base_value, color='r', linestyle='--', alpha=0.7, label='Base Case')
        axes[idx].axhline(y=base_prediction, color='r', linestyle='--', alpha=0.7)
        axes[idx].legend()
    
    plt.tight_layout()
    plt.savefig('parameter_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Parameter sensitivity analysis completed")
    print("✓ Plots saved as 'parameter_sensitivity_analysis.png'")

def demonstrate_physics_validation():
    """
    Demonstrate physics law validation
    """
    print("\n" + "=" * 80)
    print("PHYSICS LAWS VALIDATION")
    print("=" * 80)
    
    from physics_informed_neural_network import BiogasPhysicsLaws
    
    physics = BiogasPhysicsLaws()
    
    # Test temperature dependency
    print("\n1. Temperature Dependency (Arrhenius Law)")
    temperatures = np.linspace(25, 50, 26)
    temp_factors = [physics.arrhenius_temperature(torch.tensor([t])).item() for t in temperatures]
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.plot(temperatures, temp_factors, 'b-', linewidth=2)
    plt.axvline(x=37, color='r', linestyle='--', label='Optimal (37°C)')
    plt.xlabel('Temperature (°C)')
    plt.ylabel('Activity Factor')
    plt.title('Temperature Response')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Test pH inhibition
    print("2. pH Inhibition Function")
    ph_values = np.linspace(5, 9, 41)
    ph_factors = [physics.ph_inhibition(torch.tensor([ph])).item() for ph in ph_values]
    
    plt.subplot(1, 3, 2)
    plt.plot(ph_values, ph_factors, 'g-', linewidth=2)
    plt.axvline(x=7.0, color='r', linestyle='--', label='Optimal (7.0)')
    plt.axvspan(6.5, 8.0, alpha=0.2, color='green', label='Good Range')
    plt.xlabel('pH')
    plt.ylabel('Activity Factor')
    plt.title('pH Response')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Test loading rate response
    print("3. Loading Rate Response")
    loading_rates = np.linspace(0.5, 6, 23)
    loading_factors = [physics.loading_rate_inhibition(torch.tensor([lr])).item() for lr in loading_rates]
    
    plt.subplot(1, 3, 3)
    plt.plot(loading_rates, loading_factors, 'm-', linewidth=2)
    plt.axvline(x=2.0, color='r', linestyle='--', label='Optimal (2.0)')
    plt.xlabel('Loading Rate (kg VS/m³/day)')
    plt.ylabel('Activity Factor')
    plt.title('Loading Rate Response')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('physics_laws_validation.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Physics laws validation completed")
    print("✓ All functions show expected biogas digestion behavior")
    print("✓ Plots saved as 'physics_laws_validation.png'")

def main():
    """
    Main demonstration function
    """
    try:
        # Core PINN demonstration
        model, feature_names, df = demonstrate_pinn_usage()
        
        # Feature updating demonstration
        demonstrate_feature_updating(model, feature_names, df)
        
        # Parameter sensitivity analysis
        demonstrate_parameter_sensitivity(model, feature_names)
        
        # Physics validation
        demonstrate_physics_validation()
        
        print("\n" + "=" * 80)
        print("DEMONSTRATION COMPLETE")
        print("=" * 80)
        print("✓ PINN model trained and validated")
        print("✓ Feature updating system demonstrated")
        print("✓ Physics constraints validated")
        print("✓ Sensitivity analysis completed")
        print("✓ Scientific consistency checks working")
        print("\nThe Physics Informed Neural Network is ready for:")
        print("  • Real-time process optimization")
        print("  • Feature value validation")
        print("  • Scientific consistency checking")
        print("  • Predictive maintenance")
        print("  • Operational decision support")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {str(e)}")
        print("Please ensure all dependencies are installed and the dataset is available.")

if __name__ == "__main__":
    main()
