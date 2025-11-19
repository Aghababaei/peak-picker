# Installation Guide

## Quick Start (Pre-configured)

The repository includes a pre-configured virtual environment with all dependencies installed.

```bash
# Activate the environment
cd /path/to/peakpicker
source venv/bin/activate  # On macOS/Linux
# venv\Scripts\activate   # On Windows

# Verify installation
python test_installation.py
```

## System Requirements

- **Python**: 3.8+ (3.13 recommended)
- **OS**: macOS, Linux, or Windows
- **RAM**: 4GB minimum, 8GB+ recommended
- **Disk**: 500MB for packages + space for data/models

## Model Options & Dependencies

### Random Forest (No Extra Dependencies)
✅ Works out of the box
- Good accuracy (F1: 0.75-0.85)
- No additional setup required

```bash
python peakpicker.py --train --model-type random_forest
```

### XGBoost / LightGBM (Requires libomp)
⚠️ Requires OpenMP library on macOS
- Excellent accuracy (F1: 0.80-0.90)
- Faster training

**Install libomp:**
```bash
# Option 1: Homebrew
brew install libomp

# Option 2: Conda
conda install -c conda-forge llvm-openmp

# Option 3: MacPorts
sudo port install libomp
```

**Then train:**
```bash
python peakpicker.py --train --model-type xgboost
# or
python peakpicker.py --train --model-type lightgbm
```

## Manual Installation

If you need to recreate the environment:

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Verify Installation

```bash
python test_installation.py
```

## Alternative: Conda Environment

```bash
# Create environment
conda create -n peakpicker python=3.13
conda activate peakpicker

# Install packages
conda install -c conda-forge numpy pandas matplotlib seaborn scikit-learn scipy
conda install -c conda-forge xgboost lightgbm tqdm joblib pytz

# Verify
python test_installation.py
```

## Troubleshooting

### XGBoost/LightGBM: "Library not loaded: libomp.dylib"

This means OpenMP is not installed.

**Solutions:**
1. Use Random Forest: `--model-type random_forest`
2. Install libomp via Homebrew/Conda (see above)
3. Use conda environment instead

### ImportError: No module named 'module_name'

Ensure virtual environment is activated:
```bash
source venv/bin/activate
pip list  # Verify packages
```

### Memory Errors

If training runs out of memory:
1. Use Random Forest or LightGBM (lighter than XGBoost)
2. Process fewer gages at once
3. Reduce feature window sizes in `feature_engineering.py`

### macOS Version Not Supported by Homebrew

If you're on a beta/development macOS version:
- Use Random Forest (works without libomp)
- Try conda environment instead
- Wait for Homebrew compatibility update

## Testing Individual Components

```bash
source venv/bin/activate

# Test models
python -c "from sklearn.ensemble import RandomForestClassifier; print('✓ Random Forest')"
python -c "import xgboost; print('✓ XGBoost')" || echo "✗ XGBoost needs libomp"
python -c "import lightgbm; print('✓ LightGBM')" || echo "✗ LightGBM needs libomp"

# Test core modules
python data_loader.py
python feature_engineering.py
python return_period_calculator.py
```

## Package Versions

Installed packages (see `requirements.txt`):
- numpy >= 1.21.0
- pandas >= 1.3.0
- matplotlib, seaborn
- scikit-learn >= 1.0.0
- scipy
- xgboost >= 1.5.0
- lightgbm >= 3.3.0
- tqdm, joblib, pytz

## Next Steps

After successful installation:

1. **Read QUICKSTART.md** for usage tutorial
2. **Train the model**: `python peakpicker.py --train`
3. **Test on sample data**: `python peakpicker.py --gage 03408500`
4. **Process your data**: See PROJECT_RUNNER_GUIDE.md

## Uninstallation

```bash
deactivate
rm -rf venv
```

## Development Installation

For contributors:

```bash
# Install in editable mode
pip install -e .

# Install dev dependencies
pip install pytest black flake8 mypy
```

## Updating Packages

```bash
pip install --upgrade -r requirements.txt
```
