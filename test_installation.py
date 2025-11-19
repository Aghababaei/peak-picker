#!/usr/bin/env python3
"""
Test script to verify PeakPicker installation
"""
import sys

def test_imports():
    """Test that all required packages can be imported"""
    print("Testing package imports...")

    packages = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'sklearn',
        'scipy',
        'joblib',
        'xgboost',
        'lightgbm',
        'tqdm',
        'pytz'
    ]

    failed = []

    for package in packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError as e:
            print(f"  ✗ {package}: {e}")
            failed.append(package)

    if failed:
        print(f"\n❌ Failed to import: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All packages imported successfully!")
        return True


def test_modules():
    """Test that all custom modules can be imported"""
    print("\nTesting custom modules...")

    modules = [
        'data_loader',
        'feature_engineering',
        'return_period_calculator',
        'model_trainer',
        'plotter',
        'peakpicker'
    ]

    failed = []

    for module in modules:
        try:
            __import__(module)
            print(f"  ✓ {module}")
        except ImportError as e:
            print(f"  ✗ {module}: {e}")
            failed.append(module)

    if failed:
        print(f"\n❌ Failed to import modules: {', '.join(failed)}")
        return False
    else:
        print("\n✓ All custom modules imported successfully!")
        return True


def test_data_files():
    """Test that required data files exist"""
    print("\nTesting data files...")

    from pathlib import Path

    required_files = [
        'manual_added_peaks.csv',
        'return_periods.csv',
        'gages'
    ]

    missing = []

    for file in required_files:
        path = Path(file)
        if path.exists():
            if path.is_dir():
                num_files = len(list(path.glob('*_Obs.csv')))
                print(f"  ✓ {file} (found {num_files} gage files)")
            else:
                print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} (not found)")
            missing.append(file)

    if missing:
        print(f"\n⚠ Missing files: {', '.join(missing)}")
        print("  Note: This is expected if you haven't set up the data yet")
        return True  # Don't fail on missing data files
    else:
        print("\n✓ All required files found!")
        return True


def test_basic_functionality():
    """Test basic functionality of key components"""
    print("\nTesting basic functionality...")

    try:
        from data_loader import DataLoader
        loader = DataLoader()
        print("  ✓ DataLoader initialized")

        from feature_engineering import FeatureEngineer
        engineer = FeatureEngineer()
        print("  ✓ FeatureEngineer initialized")

        from return_period_calculator import ReturnPeriodCalculator
        calculator = ReturnPeriodCalculator()
        print("  ✓ ReturnPeriodCalculator initialized")

        from plotter import HydrographPlotter
        plotter = HydrographPlotter()
        print("  ✓ HydrographPlotter initialized")

        print("\n✓ All components initialized successfully!")
        return True

    except Exception as e:
        print(f"\n❌ Error during initialization: {e}")
        return False


def main():
    """Run all tests"""
    print("="*60)
    print("PeakPicker Installation Test")
    print("="*60 + "\n")

    results = []

    results.append(("Package Imports", test_imports()))
    results.append(("Custom Modules", test_modules()))
    results.append(("Data Files", test_data_files()))
    results.append(("Basic Functionality", test_basic_functionality()))

    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)

    for test_name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    all_passed = all(result[1] for result in results)

    if all_passed:
        print("\n✅ All tests passed! PeakPicker is ready to use.")
        print("\nNext steps:")
        print("  1. Train the model: python peakpicker.py --train")
        print("  2. Test on a gage: python peakpicker.py --gage 03408500")
        print("  3. See QUICKSTART.md for detailed instructions")
        return 0
    else:
        print("\n❌ Some tests failed. Please fix the issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
