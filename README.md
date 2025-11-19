# PeakPicker - Automated Flood Peak Detection

An ML-based system for automatically detecting flood peaks from hydrograph data using manually labeled peaks for training.

## Overview

PeakPicker uses machine learning to automate the identification of flood peaks in river gage streamflow data. The system is trained on manually selected peaks and can:

- Train models on gages with known return periods
- Detect peaks in existing gage data
- Process new gage files and generate predictions
- Calculate return periods when not available
- Generate visualizations of hydrographs with detected peaks

## Background

This project originated during research for the paper "How Well Do U.S. National Water Model Short-Range Forecasts Predict Flood Event Timing and Magnitude?" While analyzing flood events for selected gages, we needed to optimally identify peaks in hydrographs that corresponded to actual flood events with specific characteristics. The manual labeling process for 306 gages proved to be time-intensive and challenging.

To streamline this process and enable analysis of larger datasets in future studies, we developed PeakPicker—a machine learning model that automates the identification of event-related peaks in streamflow hydrographs. What began as an internal tool has evolved into a robust system that can assist researchers and practitioners in efficiently detecting flood peaks across extensive gage networks.

## Getting Started

Choose your use case:

### 🧪 **Testing the Model** (Demonstration & Development)

If you want to see how the model works, train it, and test on sample data:

1. **Follow the [Quick Start](#quick-start)** section below to:
   - Train the model on included sample data
   - Test peak detection on existing gages
   - See example outputs and visualizations

2. **Read the [QUICKSTART.md](QUICKSTART.md)** for step-by-step tutorial

3. **Check [INSTALLATION.md](INSTALLATION.md)** for installation help

### 🚀 **Production Use** (Process Your Own Data)

If you have your own gage data to process in a custom project folder:

1. **Use the project runner**: `project/run_project.py`

2. **Read [project/PROJECT_RUNNER_GUIDE.md](project/PROJECT_RUNNER_GUIDE.md)** for complete instructions on:
   - Setting up your project folder structure
   - Preparing your data files (gages, return periods, mappings)
   - Running batch processing on multiple gages
   - Customizing output locations

3. **Quick example**:
   ```bash
   # Process all gages in your project folder
   python project/run_project.py --project-dir /path/to/your/project

   # Process a single gage
   python project/run_project.py --gage 01200600
   ```

**Key Difference**: The main `peakpicker.py` script uses the included sample data, while `project/run_project.py` is designed for processing your own custom datasets with your own folder structure.

---

## Project Structure

```
peakpicker/
├── Core Package (for testing/development)
│   ├── data_loader.py              # Data loading and preprocessing
│   ├── feature_engineering.py      # Feature extraction from time series
│   ├── return_period_calculator.py # Return period calculations
│   ├── model_trainer.py            # ML model training
│   ├── plotter.py                  # Visualization tools
│   ├── peakpicker.py               # Main script for sample data
│   ├── requirements.txt            # Python dependencies
│   ├── venv/                       # Virtual environment
│   ├── gages/                      # Full sample gage data files (300+ gages)
│   ├── gages_samples/              # 10 sample gages for quick testing
│   │   └── *_Obs.csv              # Format: datetime_utc, discharge_cms
│   ├── manual_added_peaks.csv      # Manually labeled peaks
│   ├── return_periods.csv          # Return period values
│   └── plots/                      # Generated plots (created automatically)
│
├── Production Runner (for your data)
│   └── project/                    # Project folder with runner and data
│       ├── run_project.py          # Project runner script
│       ├── PROJECT_RUNNER_GUIDE.md # Complete usage guide
│       ├── gages/                  # Your gage CSV files
│       ├── data/                   # Your return periods & mappings
│       ├── plots/                  # Output plots (auto-created)
│       └── results/                # Output CSVs (auto-created)
│
└── Documentation
    ├── README.md                   # This file
    ├── QUICKSTART.md               # Step-by-step tutorial
    └── INSTALLATION.md             # Installation & setup guide
```

## Installation

### 1. Activate the Virtual Environment

```bash
# On macOS/Linux
source venv/bin/activate

# On Windows
venv\Scripts\activate
```

The virtual environment is already created with all necessary packages installed.

### 2. Verify Installation

```bash
python -c "import pandas, sklearn, xgboost; print('All packages installed!')"
```

## Quick Start

### Step 1: Train the Model

Train a peak detection model on all gages with return periods:

```bash
python peakpicker.py --train
```

This will:
- Load all gages with return period data
- Extract features from streamflow time series
- Train an XGBoost classifier
- Evaluate performance on test data
- Save the model to `peak_model.pkl`

### Step 2: Detect Peaks for an Existing Gage

```bash
python peakpicker.py --gage 03408500
```

This will:
- Load the gage data
- Apply the trained model
- Detect flood peaks
- Generate plots
- Save results to `03408500_detected_peaks.csv`

### Step 3: Process a New Gage File

For a new gage with format: `datetime,streamflow`

```bash
python peakpicker.py --file new_gage.csv --gage-number 12345678
```

This will:
- Load the new file
- Calculate return periods from the data
- Detect peaks
- Generate plots
- Save results

## Usage Examples

### Train Different Model Types

```bash
# XGBoost (default, recommended)
python peakpicker.py --train --model-type xgboost

# LightGBM (faster training)
python peakpicker.py --train --model-type lightgbm

# Random Forest
python peakpicker.py --train --model-type random_forest
```

### Adjust Detection Threshold

```bash
# More sensitive (detects more peaks)
python peakpicker.py --gage 03408500 --threshold 0.3

# Less sensitive (detects fewer peaks)
python peakpicker.py --gage 03408500 --threshold 0.7
```

### Adjust Minimum Peak Distance

```bash
# Peaks must be at least 72 hours apart
python peakpicker.py --gage 03408500 --min-distance 72

# Peaks must be at least 24 hours apart
python peakpicker.py --gage 03408500 --min-distance 24
```

### Process New File with Custom Column Names

```bash
python peakpicker.py \
  --file data.csv \
  --gage-number 12345678 \
  --datetime-col timestamp \
  --discharge-col flow_rate
```

### Disable Plotting

```bash
python peakpicker.py --gage 03408500 --no-plot
```

### Save to Custom Output File

```bash
python peakpicker.py --gage 03408500 --output my_peaks.csv
```

## Data Format

### Input Gage Data

CSV files should contain:
- **datetime column**: Timestamps (any standard format)
- **discharge column**: Streamflow values (in cms or any consistent unit)

Example:
```csv
datetime,streamflow
2021-01-01 00:00:00,45.2
2021-01-01 00:15:00,45.5
2021-01-01 00:30:00,45.8
```

### Output Format

Detected peaks are saved as CSV with columns:
- `peak_time_utc`: Timestamp of detected peak
- `peak_flow_cms`: Discharge value at peak
- `peak_probability`: Model confidence (0-1)
- `site_no`: Gage number

## Key Concepts

### Return Periods

Return periods represent the statistical recurrence interval of flood events:
- **2-year**: Expected once every 2 years (common flood)
- **5-year**: Expected once every 5 years
- **10-year**: Expected once every 10 years
- **25-year**: Expected once every 25 years
- **50-year**: Expected once every 50 years
- **100-year**: Expected once every 100 years (rare, extreme flood)

The system can:
1. Use pre-calculated return periods from `return_periods.csv`
2. Calculate return periods from streamflow data using statistical methods

### Features Used for Peak Detection

The model uses 100+ features including:

**Temporal Features**:
- Hour, day, month, season

**Statistical Features**:
- Rolling means, standard deviations
- Percentiles (50th, 75th, 90th, 95th, 99th)
- Skewness and kurtosis

**Derivative Features**:
- Rate of change (velocity, acceleration)
- Sign changes

**Local Features**:
- Local maxima detection
- Distance to local peaks
- Peak prominence and width

**Return Period Features**:
- Ratios to return period thresholds
- Exceedance indicators

### Gage Number Matching

The system handles gage numbers with/without leading zeros:
- `03408500` and `3408500` are treated as the same gage
- Files are matched flexibly (e.g., `03408500_Obs.csv` or `3408500_Obs.csv`)

## Python API

You can also use PeakPicker programmatically:

```python
from peakpicker import PeakPicker

# Initialize
picker = PeakPicker(model_path='peak_model.pkl')

# Train model
picker.train_model(model_type='xgboost')

# Detect peaks for existing gage
detected_peaks = picker.detect_peaks_for_gage(
    site_no='03408500',
    probability_threshold=0.5,
    min_peak_distance_hours=48,
    plot=True
)

# Process new file
detected_peaks = picker.process_new_gage_file(
    file_path='new_gage.csv',
    gage_number='12345678',
    datetime_col='datetime',
    discharge_col='streamflow',
    plot=True
)
```

## Generated Plots

The system creates three types of plots:

1. **Hydrograph with Peaks** (`{gage}_hydrograph.png`)
   - Time series of streamflow
   - Detected peaks marked in red
   - Manual peaks marked in green (if available)
   - Return period thresholds
   - Peak probability overlay

2. **Peak Comparison** (`{gage}_peak_comparison.png`)
   - Comparison between detected and manual peaks
   - Distribution analysis
   - Monthly patterns
   - Statistical summary

3. **Annual Maxima** (`{gage}_annual_maxima.png`)
   - Annual maximum flows
   - Return period thresholds
   - Trend analysis

All plots are saved to the `plots/` directory (created automatically).

## Model Performance

The model is evaluated using:
- **Precision**: Proportion of detected peaks that are correct
- **Recall**: Proportion of actual peaks that were detected
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve

During training, you'll see:
```
Metrics:
  precision: 0.8523
  recall: 0.7891
  f1_score: 0.8194
  auc: 0.9234
```

## Troubleshooting

### Model Not Found Error

```bash
FileNotFoundError: Model file not found: peak_model.pkl
```

**Solution**: Train the model first:
```bash
python peakpicker.py --train
```

### Gage Data Not Found

```bash
Warning: Could not find data file for gage 12345678
```

**Solutions**:
1. Check that the gage file exists in the `gages/` directory
2. Verify the file name format: `{gage_number}_Obs.csv`
3. Try with/without leading zeros

### Memory Issues

If you encounter memory errors during training:

1. Process fewer gages at once (modify `data_loader.py`)
2. Reduce feature window sizes in `feature_engineering.py`
3. Use a lighter model: `--model-type random_forest`

## Advanced Usage

### Testing Individual Modules

```bash
# Test data loader
python data_loader.py

# Test feature engineering
python feature_engineering.py

# Test return period calculator
python return_period_calculator.py

# Test plotter
python plotter.py

# Test model trainer
python model_trainer.py
```

### Customize Model Parameters

Edit `model_trainer.py` to adjust:
- Number of estimators
- Learning rate
- Max depth
- Class weights (for handling imbalance)

### Add Custom Features

Edit `feature_engineering.py` to add new features:

```python
def add_custom_features(self, df):
    # Add your custom features here
    df['my_feature'] = ...
    return df
```

Then add to `engineer_features()` method.

## Files and Data

### Input Files (Required)

- `gages/*_Obs.csv`: Gage streamflow data
- `manual_added_peaks.csv`: Manually labeled peaks for training
- `return_periods.csv`: Return period values (optional, can be calculated)

### Output Files (Generated)

- `peak_model.pkl`: Trained ML model
- `{gage}_detected_peaks.csv`: Detected peaks for each gage
- `plots/*.png`: Visualizations

## Production Use: Processing Your Own Data

If you have your own gage datasets to process, use the **project runner** instead of the main `peakpicker.py` script:

### Setup Your Project Folder

```
your_project/
├── gages/              # Your gage CSV files (datetime, discharge)
├── data/
│   ├── usgsid_comid.csv       # Maps USGS IDs to COMIDs
│   └── return_periods.csv     # Return periods by COMID
├── plots/              # Output plots (auto-created)
└── results/            # Output CSVs (auto-created)
```

### Run the Project Processor

```bash
# Process all gages
python project/run_project.py --project-dir /path/to/your_project

# Process single gage
python project/run_project.py --project-dir /path/to/your_project --gage 01200600

# Adjust sensitivity
python project/run_project.py --project-dir /path/to/your_project --threshold 0.6
```

### Complete Documentation

See **[project/PROJECT_RUNNER_GUIDE.md](project/PROJECT_RUNNER_GUIDE.md)** for:
- Detailed setup instructions
- Data file format specifications
- Batch processing examples
- Troubleshooting tips
- Customization options

**Key Features**:
- ✅ Batch process multiple gages
- ✅ Custom folder structure
- ✅ Automatic return period matching
- ✅ Handles USGS IDs with/without leading zeros
- ✅ Generates combined results CSV
- ✅ Beautiful visualizations with colored return period zones

## Citation

If you use this tool in your research, please cite:

```
PeakPicker - Automated Flood Peak Detection
Brigham Young University
2025
```

## Support

For issues or questions:
1. Check this README for solutions
2. Review the example outputs
3. Examine the generated plots for insights

## License

[Specify your license here]

## Acknowledgments

This project was developed as part of research at **Brigham Young University** for automated flood peak detection using machine learning on hydrologic time series data.

**Note**: This repository represents a side result from a research paper and project conducted at BYU. Upon completion of necessary approvals, this repository will be transferred to the official [BYU Hydroinformatics](https://github.com/BYU-Hydroinformatics) organization.

### Development

Developed at Brigham Young University's Hydroinformatics Lab for advancing automated hydrologic analysis and flood detection methodologies.
