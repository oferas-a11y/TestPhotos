#!/usr/bin/env python3
"""
Test runner script for historical photo analysis tools
Runs all tests and generates coverage report
"""

import pytest
import sys
import os
from pathlib import Path

def main():
    """Run all tests with coverage reporting"""
    
    # Add project root to Python path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    # Test arguments
    test_args = [
        str(Path(__file__).parent),  # Test directory
        '-v',                        # Verbose output
        '--tb=short',               # Short traceback format
        '--durations=10',           # Show 10 slowest tests
        '--cov=opencv_analysis',    # Coverage for opencv_analysis
        '--cov=photo_clustering',   # Coverage for photo_clustering  
        '--cov=yolo_detection',     # Coverage for yolo_detection
        '--cov=clip_analysis',      # Coverage for clip_analysis
        '--cov-report=html',        # HTML coverage report
        '--cov-report=term',        # Terminal coverage report
        '--cov-fail-under=70',      # Fail if coverage under 70%
    ]
    
    # Add markers for conditional testing
    print("Running Historical Photo Analysis Test Suite")
    print("=" * 60)
    print()
    
    # Check available packages
    packages_status = check_package_availability()
    print("Package Availability:")
    for package, available in packages_status.items():
        status = "✓ Available" if available else "✗ Not Available"
        print(f"  {package}: {status}")
    print()
    
    # Run tests
    exit_code = pytest.main(test_args)
    
    # Print summary
    print("\n" + "=" * 60)
    if exit_code == 0:
        print("🎉 All tests passed!")
        print("📊 Coverage report generated in htmlcov/index.html")
    else:
        print("❌ Some tests failed")
        print(f"Exit code: {exit_code}")
    
    return exit_code

def check_package_availability():
    """Check which packages are available for testing"""
    packages = {
        'OpenCV (cv2)': False,
        'scikit-learn': False,
        'YOLO (ultralytics)': False,
        'CLIP': False,
        'PyTorch': False,
        'PIL/Pillow': False,
        'matplotlib': False,
        'numpy': False
    }
    
    try:
        import cv2
        packages['OpenCV (cv2)'] = True
    except ImportError:
        pass
    
    try:
        import sklearn
        packages['scikit-learn'] = True
    except ImportError:
        pass
    
    try:
        from ultralytics import YOLO
        packages['YOLO (ultralytics)'] = True
    except ImportError:
        pass
    
    try:
        import clip
        packages['CLIP'] = True
    except ImportError:
        pass
    
    try:
        import torch
        packages['PyTorch'] = True
    except ImportError:
        pass
    
    try:
        from PIL import Image
        packages['PIL/Pillow'] = True
    except ImportError:
        pass
    
    try:
        import matplotlib
        packages['matplotlib'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        packages['numpy'] = True
    except ImportError:
        pass
    
    return packages

def run_specific_tests():
    """Run specific test categories"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run specific test categories')
    parser.add_argument('--opencv', action='store_true', help='Run OpenCV tests only')
    parser.add_argument('--clustering', action='store_true', help='Run clustering tests only')
    parser.add_argument('--yolo', action='store_true', help='Run YOLO tests only')
    parser.add_argument('--clip', action='store_true', help='Run CLIP tests only')
    parser.add_argument('--integration', action='store_true', help='Run integration tests only')
    parser.add_argument('--fast', action='store_true', help='Skip slow tests')
    
    args = parser.parse_args()
    
    test_files = []
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    if args.opencv:
        test_files.append('test_opencv_analysis.py')
    if args.clustering:
        test_files.append('test_photo_clustering.py')
    if args.yolo:
        test_files.append('test_yolo_detection.py')
    if args.clip:
        test_files.append('test_clip_analysis.py')
    if args.integration:
        test_files.append('test_integration.py')
    
    if not test_files:
        print("No specific test category selected. Running all tests.")
        return main()
    
    test_args = []
    for test_file in test_files:
        test_args.append(str(Path(__file__).parent / test_file))
    
    test_args.extend(['-v', '--tb=short'])
    
    if args.fast:
        test_args.extend(['-m', 'not slow'])
    
    print(f"Running specific tests: {', '.join(test_files)}")
    exit_code = pytest.main(test_args)
    
    return exit_code

if __name__ == '__main__':
    import sys
    
    # Check if specific test arguments provided
    if len(sys.argv) > 1 and any(arg.startswith('--') for arg in sys.argv[1:]):
        exit_code = run_specific_tests()
    else:
        exit_code = main()
    
    sys.exit(exit_code)