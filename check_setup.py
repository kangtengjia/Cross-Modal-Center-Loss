#!/usr/bin/env python3
"""
Script to check if all required components for Cross-Modal Center Loss are properly set up.
"""

import os
import sys
import importlib.util

def check_python_packages():
    """Check if required Python packages are installed."""
    required_packages = [
        'torch',
        'torchvision',
        'numpy',
        'scipy',
        'sklearn',
        'matplotlib',
        'tensorboard',
        'h5py',
        'PIL',
        'tqdm'
    ]
    
    print("Checking Python packages...")
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'sklearn':
                importlib.util.find_spec('sklearn')
            elif package == 'PIL':
                importlib.util.find_spec('PIL')
            else:
                importlib.util.find_spec(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing_packages.append(package)
    
    return len(missing_packages) == 0

def check_project_files():
    """Check if required project files exist."""
    required_files = [
        'config.py',
        'train.py',
        'evaluate_retrieval.py',
        'run_evaluation.py',
        'run_tensorboard.py',
        'demo.py',
        'requirements.txt',
        'README.md',
        'SUMMARY.md'
    ]
    
    print("\nChecking project files...")
    missing_files = []
    
    for file in required_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_model_files():
    """Check if model files exist."""
    model_files = [
        'models/corrnet.py',
        'models/dgcnn.py',
        'models/meshnet.py',
        'models/resnet.py'
    ]
    
    print("\nChecking model files...")
    missing_files = []
    
    for file in model_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_tool_files():
    """Check if tool files exist."""
    tool_files = [
        'tools/dataloader.py',
        'tools/test_dataloader.py'
    ]
    
    print("\nChecking tool files...")
    missing_files = []
    
    for file in tool_files:
        if os.path.exists(file):
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (missing)")
            missing_files.append(file)
    
    return len(missing_files) == 0

def check_directories():
    """Check if required directories exist."""
    required_dirs = [
        'models',
        'tools',
        'dataset',
        'checkpoints'
    ]
    
    print("\nChecking directories...")
    missing_dirs = []
    
    for directory in required_dirs:
        if os.path.exists(directory) and os.path.isdir(directory):
            print(f"  ✓ {directory}")
        else:
            print(f"  ✗ {directory} (missing)")
            missing_dirs.append(directory)
    
    return len(missing_dirs) == 0

def check_gpu():
    """Check if CUDA is available."""
    print("\nChecking GPU support...")
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available with {torch.cuda.device_count()} device(s)")
            print(f"  ✓ Current device: {torch.cuda.get_device_name(0)}")
            return True
        else:
            print("  ✗ CUDA not available")
            return False
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False

def main():
    print("Cross-Modal Center Loss Setup Check")
    print("===================================")
    
    # Run all checks
    checks = [
        ("Python Packages", check_python_packages),
        ("Project Files", check_project_files),
        ("Model Files", check_model_files),
        ("Tool Files", check_tool_files),
        ("Directories", check_directories),
        ("GPU Support", check_gpu)
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n{check_name}:")
        result = check_func()
        results.append(result)
    
    # Summary
    print("\n" + "="*50)
    print("Setup Check Summary:")
    all_passed = all(results)
    
    if all_passed:
        print("  ✓ All checks passed! The setup is complete.")
        print("\nYou can now proceed with training:")
        print("  python train.py --dataset ModelNet40 --num_classes 40 --batch_size 96")
    else:
        print("  ✗ Some checks failed. Please address the issues above.")
        print("\nCommon fixes:")
        print("  - Install missing Python packages: pip install -r requirements.txt")
        print("  - Ensure all project files are in place")
        print("  - Check that CUDA is properly installed for GPU support")

if __name__ == "__main__":
    main()