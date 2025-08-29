#!/usr/bin/env python3

# Test script to verify all imports work correctly
try:
    import torch
    print("✓ PyTorch imported successfully")
    print(f"  PyTorch version: {torch.__version__}")
except ImportError as e:
    print(f"✗ PyTorch import failed: {e}")

try:
    from d2l import torch as d2l
    print("✓ D2L imported successfully")
except ImportError as e:
    print(f"✗ D2L import failed: {e}")

try:
    import matplotlib.pyplot as plt
    print("✓ Matplotlib imported successfully")
except ImportError as e:
    print(f"✗ Matplotlib import failed: {e}")

try:
    import pandas as pd
    print("✓ Pandas imported successfully")
except ImportError as e:
    print(f"✗ Pandas import failed: {e}")

print("\nAll imports completed!") 