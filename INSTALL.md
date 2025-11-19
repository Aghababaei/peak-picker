# Installation Guide

## Quick Install (Virtual Environment Already Created)

The virtual environment is already set up with all packages installed!

### Activate the Environment

```bash
cd /Users/amin/Downloads/peakpicker
source venv/bin/activate
```

## Known Issues

### XGBoost OpenMP Error on macOS

If you encounter this error:
```
XGBoost Library (libxgboost.dylib) could not be loaded.
Mac OSX users: Run `brew install libomp` to install OpenMP runtime.
```

**Solutions:**

#### Option 1: Install libomp via Homebrew (Recommended)

```bash
brew install libomp
```

If brew has issues with your macOS version, try:
```bash
# Update Homebrew first
brew update

# Then install libomp
brew install libomp
```

#### Option 2: Use LightGBM Instead (No OpenMP Required)

LightGBM is already installed and works without OpenMP. Simply train using:

```bash
python peakpicker.py --train --model-type lightgbm
```

LightGBM provides similar performance to XGBoost and is often faster.

#### Option 3: Use Random Forest (No Dependencies)

Random Forest is part of scikit-learn and has no additional dependencies:

```bash
python peakpicker.py --train --model-type random_forest
```

### Verify Installation

Run the test script to check everything is working:

```bash
python test_installation.py
```

This will test:
- All Python packages are installed correctly
- Custom modules can be imported
- Data files are present
- Basic functionality works

## Manual Installation (If Needed)

If you need to recreate the virtual environment:

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Upgrade pip

```bash
pip install --upgrade pip
```

### 3. Install Packages

```bash
pip install -r requirements.txt
```

This installs:
- numpy (numerical computing)
- pandas (data manipulation)
- matplotlib, seaborn (visualization)
- scikit-learn (machine learning)
- scipy (scientific computing)
- xgboost (gradient boosting)
- lightgbm (gradient boosting alternative)
- tqdm (progress bars)
- pytz (timezones)
- joblib (model serialization)

## System Requirements

- **Python**: 3.8 or higher (3.13 recommended)
- **OS**: macOS, Linux, or Windows
- **RAM**: 4GB minimum, 8GB+ recommended for large datasets
- **Disk**: 500MB for packages, additional space for data and models

## Package Versions

Minimum versions (specified in requirements.txt):
- numpy >= 1.21.0
- pandas >= 1.3.0
- scikit-learn >= 1.0.0
- xgboost >= 1.5.0
- lightgbm >= 3.3.0

## Troubleshooting

### ImportError: No module named 'module_name'

Make sure the virtual environment is activated:
```bash
source venv/bin/activate
```

Then verify packages are installed:
```bash
pip list
```

### Memory Errors

If you run out of memory during training:
1. Process fewer gages at once
2. Use a lighter model (random_forest or lightgbm)
3. Reduce feature window sizes in feature_engineering.py

### Pandas FutureWarning about fillna

This is a deprecation warning that can be safely ignored. It will be fixed in future updates but doesn't affect functionality.

## Development Installation

If you want to modify the code:

### Install in Editable Mode

```bash
pip install -e .
```

### Install Development Dependencies

```bash
pip install pytest black flake8 mypy
```

## Updating Packages

To update all packages to their latest versions:

```bash
pip install --upgrade -r requirements.txt
```

## Uninstallation

To remove the virtual environment:

```bash
deactivate
rm -rf venv
```

## Next Steps

After successful installation:

1. **Read QUICKSTART.md** for step-by-step usage instructions
2. **Run test_installation.py** to verify everything works
3. **Train the model**: `python peakpicker.py --train`
4. **Test on a gage**: `python peakpicker.py --gage 03408500`

## Support

If you encounter installation issues:
1. Check this guide for solutions
2. Verify Python version: `python --version`
3. Check package versions: `pip list`
4. Try using LightGBM instead of XGBoost
