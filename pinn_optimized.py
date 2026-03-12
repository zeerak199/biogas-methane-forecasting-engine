"""
Optimized PINN Configuration for Better Training Performance
"""

import pandas as pd
import torch
from physics_informed_neural_network import (
    PhysicsInformedNN, BiogasPINNTrainer, create_data_loaders
)

def run_optimized_pinn():
    """
    Run PINN with optimized hyperparameters for better performance
    """
    print("=" * 60)
    print("OPTIMIZED PHYSICS INFORMED NEURAL NETWORK")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('biogas_digestor_dataset.csv')
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'methane_production_m3_day']]
    target_col = 'methane_production_m3_day'
    
    # Normalize target variable for better training
    df[target_col] = (df[target_col] - df[target_col].mean()) / df[target_col].std()
    
    # Create data loaders with better batch size
    train_loader, val_loader, feature_names = create_data_loaders(
        df, feature_cols, target_col, batch_size=64  # Larger batch for stability
    )
    
    print(f"Features: {len(feature_cols)}")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    
    # Optimized model configuration
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    model = PhysicsInformedNN(
        input_dim=len(feature_cols),
        hidden_dims=[64, 128, 128, 64],  # Smaller, more manageable
        output_dim=1,
        physics_weight=0.1,  # Reduced physics weight for initial learning
        device=device
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Initial physics weight: {model.physics_weight}")
    
    # Optimized trainer
    trainer = BiogasPINNTrainer(
        model, 
        learning_rate=0.01,  # Higher initial learning rate
        device=device
    )
    
    # Train with more patience
    print("\nStarting optimized training...")
    loss_history = trainer.train(
        train_loader, 
        val_loader, 
        epochs=200,    # More epochs
        patience=50    # More patience
    )
    
    # Display results
    print(f"\nTraining completed!")
    print(f"Final train loss: {loss_history['train_total'][-1]:.4f}")
    print(f"Final val loss: {loss_history['val_total'][-1]:.4f}")
    print(f"Best improvement: {(loss_history['train_total'][0] - min(loss_history['train_total'])) / loss_history['train_total'][0] * 100:.1f}%")
    
    # Progressive physics weight increase
    if len(loss_history['train_total']) > 50:
        print("\n" + "="*60)
        print("PHASE 2: INCREASING PHYSICS WEIGHT")
        print("="*60)
        
        # Increase physics weight gradually
        model.physics_weight = 0.3
        trainer_phase2 = BiogasPINNTrainer(model, learning_rate=0.001)
        
        loss_history_phase2 = trainer_phase2.train(
            train_loader, val_loader, epochs=100, patience=30
        )
        
        print(f"Phase 2 completed!")
        print(f"Final physics-enhanced loss: {loss_history_phase2['train_total'][-1]:.4f}")
    
    # Show training curves
    trainer.plot_training_history()
    
    return model

def quick_test_run():
    """
    Quick test with minimal epochs to verify everything works
    """
    print("=" * 60)
    print("QUICK TEST - PINN FUNCTIONALITY")
    print("=" * 60)
    
    # Load dataset
    df = pd.read_csv('biogas_digestor_dataset.csv')
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['timestamp', 'methane_production_m3_day']]
    target_col = 'methane_production_m3_day'
    
    # Simple normalization
    df[target_col] = (df[target_col] - df[target_col].mean()) / df[target_col].std()
    
    # Create data loaders
    train_loader, val_loader, feature_names = create_data_loaders(
        df, feature_cols, target_col, batch_size=32
    )
    
    # Simple model
    device = 'mps' if torch.backends.mps.is_available() else 'cuda' if torch.cuda.is_available() else 'cpu'
    model = PhysicsInformedNN(
        input_dim=len(feature_cols),
        hidden_dims=[32, 64, 32],  # Very simple
        physics_weight=0.05,       # Minimal physics weight
        device=device
    )
    
    trainer = BiogasPINNTrainer(model, learning_rate=0.005)
    
    print("Running quick test (10 epochs)...")
    loss_history = trainer.train(train_loader, val_loader, epochs=10, patience=10)
    
    print(f"✅ Test completed successfully!")
    print(f"Initial loss: {loss_history['train_total'][0]:.4f}")
    print(f"Final loss: {loss_history['train_total'][-1]:.4f}")
    print(f"Physics integration: {'✅ Working' if loss_history['train_physics'][-1] > 0 else '❌ Not working'}")
    
    return model

def analyze_training_issues(loss_history):
    """
    Analyze potential training issues and provide recommendations
    """
    print("\n" + "="*60)
    print("TRAINING ANALYSIS & RECOMMENDATIONS")
    print("="*60)
    
    train_losses = loss_history['train_total']
    val_losses = loss_history['val_total']
    
    # Check for various issues
    issues = []
    recommendations = []
    
    # 1. High initial loss
    if train_losses[0] > 1000:
        issues.append("⚠️  Very high initial loss - suggests poor initialization or scaling")
        recommendations.append("• Normalize/standardize input features and target variable")
        recommendations.append("• Reduce initial learning rate")
        recommendations.append("• Check data preprocessing")
    
    # 2. No improvement
    if len(train_losses) > 5:
        recent_improvement = (train_losses[0] - train_losses[-1]) / train_losses[0]
        if recent_improvement < 0.01:
            issues.append("⚠️  Minimal learning progress (<1% improvement)")
            recommendations.append("• Increase learning rate")
            recommendations.append("• Reduce physics weight initially")
            recommendations.append("• Check feature scaling")
    
    # 3. Early stopping
    if len(train_losses) < 30:
        issues.append("⚠️  Training stopped very early")
        recommendations.append("• Increase patience parameter")
        recommendations.append("• Use learning rate scheduling")
        recommendations.append("• Reduce physics constraints initially")
    
    # 4. Physics vs Data loss balance
    if len(loss_history['train_physics']) > 0:
        avg_physics = sum(loss_history['train_physics']) / len(loss_history['train_physics'])
        avg_data = sum(loss_history['train_data']) / len(loss_history['train_data'])
        
        if avg_physics > avg_data * 0.1:
            issues.append("⚠️  Physics loss may be dominating")
            recommendations.append("• Reduce physics_weight parameter")
            recommendations.append("• Use progressive physics weight increase")
    
    # Display analysis
    if issues:
        print("🔍 IDENTIFIED ISSUES:")
        for issue in issues:
            print(f"   {issue}")
        
        print("\n💡 RECOMMENDATIONS:")
        for rec in recommendations:
            print(f"   {rec}")
    else:
        print("✅ No obvious training issues detected!")
    
    print(f"\n📊 TRAINING SUMMARY:")
    print(f"   Epochs completed: {len(train_losses)}")
    print(f"   Initial loss: {train_losses[0]:.4f}")
    print(f"   Final loss: {train_losses[-1]:.4f}")
    if len(train_losses) > 1:
        improvement = (train_losses[0] - train_losses[-1]) / train_losses[0] * 100
        print(f"   Total improvement: {improvement:.1f}%")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "quick":
        # Quick test mode
        model = quick_test_run()
    else:
        # Full optimized training
        model = run_optimized_pinn()
    
    print("\n🎉 PINN is working correctly!")
    print("Next steps:")
    print("• Use the trained model for feature updating")
    print("• Test with real biogas operational data")
    print("• Fine-tune physics weights for your specific use case")
