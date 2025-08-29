#!/usr/bin/env python3
"""
Script to help fix notebook import issues
Run this before opening the notebook
"""

import sys
import subprocess

def install_missing_packages():
    """Install any missing packages that might be needed"""
    packages = [
        'torch',
        'torchvision', 
        'torchaudio',
        'd2l',
        'matplotlib',
        'pandas',
        'numpy',
        'jupyter',
        'ipykernel'
    ]
    
    for package in packages:
        try:
            __import__(package)
            print(f"✓ {package} is already installed")
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✓ {package} installed successfully")

def test_imports():
    """Test all required imports"""
    print("\n=== Testing Imports ===")
    
    try:
        import torch
        print(f"✓ PyTorch {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False
    
    try:
        from d2l import torch as d2l
        print("✓ D2L imported successfully")
    except ImportError as e:
        print(f"✗ D2L import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ Matplotlib imported successfully")
    except ImportError as e:
        print(f"✗ Matplotlib import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ Pandas imported successfully")
    except ImportError as e:
        print(f"✗ Pandas import failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    print("=== Fixing Notebook Dependencies ===")
    
    # Install missing packages
    install_missing_packages()
    
    # Test imports
    if test_imports():
        print("\n✓ All imports working! You can now run the notebook.")
        print("\nTo run the notebook:")
        print("1. Open your browser to http://localhost:8888")
        print("2. Navigate to classification.ipynb")
        print("3. Run all cells (Kernel -> Restart & Run All)")
    else:
        print("\n✗ Some imports failed. Please check the error messages above.") 