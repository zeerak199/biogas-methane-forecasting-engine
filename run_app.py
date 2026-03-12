"""
Simple launcher for the Biogas Forecasting Streamlit Application
"""

import subprocess
import sys
import os

def check_dependencies():
    """Check if required packages are installed"""
    required_packages = [
        'streamlit', 'pandas', 'numpy', 'plotly', 'torch', 
        'sklearn', 'xgboost', 'lightgbm'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def main():
    """Launch the Streamlit application"""
    
    print("Biogas Methane Forecasting System")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists('biogas_digestor_dataset.csv'):
        print("ERROR: Dataset not found. Please run this script from the project directory.")
        return
    
    # Check dependencies
    if not check_dependencies():
        return
    
    print("SUCCESS: All dependencies found!")
    
    # Ask user which version to run
    print("\nChoose Application Version:")
    print("1. Quick Start App (ML forecasting with basic physics)")
    print("2. Clean PINN App (Physics validation and simple demo)")
    print("3. Full PINN App (Complete physics-informed system)")
    
    choice = input("\nEnter choice (1, 2, or 3): ").strip()
    
    if choice == "1":
        app_file = "quick_start_app.py"
        print("\nLaunching Quick Start App...")
        print("Features available:")
        print("   • Model Training - Train ML models")
        print("   • 24-Hour Forecast - Generate predictions")
        print("   • Data Analysis - Historical analysis")
        print("   • Basic Physics Validation")
    elif choice == "2":
        app_file = "clean_streamlit_app.py"
        print("\nLaunching Clean PINN App...")
        print("Features available:")
        print("   • PINN Validation - Physics validation with recommendations")
        print("   • Quick Forecast Demo - Simple physics-based predictions")
        print("   • No emojis, fixed recommendation system")
    else:
        app_file = "streamlit_biogas_forecasting.py"
        print("\nLaunching Full PINN App...")
        print("Features available:")
        print("   • Home - System overview")
        print("   • Model Training - Train ML models")
        print("   • 24-Hour Forecast - Generate predictions")
        print("   • PINN Validation - Physics validation")
        print("   • Analysis Dashboard - Data analysis")
    
    print("Application will open in your browser automatically")
    print("\n" + "=" * 50)
    
    # Launch Streamlit
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            app_file,
            "--server.address", "localhost",
            "--server.port", "8501",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\nApplication stopped by user")
    except Exception as e:
        print(f"ERROR: Error launching application: {e}")

if __name__ == "__main__":
    main()
