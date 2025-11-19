# Quick Start Guide

## Step-by-Step Instructions

### 1. Activate Virtual Environment

```bash
cd /Users/amin/Downloads/peakpicker
source venv/bin/activate
```

You should see `(venv)` in your terminal prompt.

### 2. Train the Model (First Time Only)

```bash
python peakpicker.py --train
```

**What this does:**
- Loads all gages with return period data
- Extracts 100+ features from streamflow time series
- Trains an XGBoost model
- Evaluates on test data
- Saves model to `peak_model.pkl`

**Expected time:** 5-15 minutes depending on data size

**Expected output:**
```
Processing gages: 100%|████████████| 200/200
Total samples: 1500000
Positive samples (peaks): 2500
Training xgboost model...
Metrics:
  precision: 0.85
  recall: 0.79
  f1_score: 0.82
Model saved to peak_model.pkl
```

### 3. Test on an Existing Gage

```bash
python peakpicker.py --gage 03408500
```

**What this does:**
- Loads gage data from `gages/03408500_Obs.csv`
- Gets return periods from `return_periods.csv`
- Engineers features
- Predicts peaks
- Creates plots in `plots/` directory
- Saves results to `03408500_detected_peaks.csv`

**Expected output:**
```
DETECTING PEAKS FOR GAGE 03408500
Loading gage data...
Data loaded: 50000 time steps
Date range: 2021-04-19 to 2023-10-31

Getting return periods...
Using pre-calculated return periods:
  return_period_2: 745.29 cms
  return_period_5: 1088.12 cms
  return_period_10: 1315.10 cms
  return_period_25: 1601.90 cms
  return_period_50: 1814.66 cms
  return_period_100: 2025.85 cms

Engineering features...
Created 150 total columns

Predicting peaks...
Found 45 potential peaks (threshold: 0.5)
After distance filtering: 12 peaks

Detected Peaks:
  2022-01-02 12:00:00: 421.92 cms (probability: 0.892)
  2022-02-04 12:45:00: 404.93 cms (probability: 0.856)
  2022-02-23 17:30:00: 702.26 cms (probability: 0.945)
  ...

Manual peaks available: 6
Peak Detection Performance:
  Matched peaks: 5
  Precision: 83.33%
  Recall: 83.33%
  F1-Score: 83.33%

Creating plots...
Plot saved to plots/03408500_hydrograph.png
Comparison plot saved to plots/03408500_peak_comparison.png
Annual maxima plot saved to plots/03408500_annual_maxima.png

Detected peaks saved to: 03408500_detected_peaks.csv

PEAK DETECTION COMPLETE
```

### 4. View the Results

Check the generated files:

```bash
# View detected peaks
cat 03408500_detected_peaks.csv

# Open plots (macOS)
open plots/03408500_hydrograph.png
open plots/03408500_peak_comparison.png
open plots/03408500_annual_maxima.png
```

### 5. Process a New Gage File

If you have a new gage file (format: datetime,streamflow):

```bash
python peakpicker.py --file your_data.csv --gage-number 12345678
```

**Important:** The CSV file should have columns named `datetime` and `streamflow` (or specify custom names with `--datetime-col` and `--discharge-col`).

Example new file format:
```csv
datetime,streamflow
2020-01-01 00:00:00,45.2
2020-01-01 00:15:00,45.5
2020-01-01 00:30:00,45.8
```

## Common Commands

### Adjust Sensitivity

```bash
# More sensitive (more peaks detected)
python peakpicker.py --gage 03408500 --threshold 0.3

# Less sensitive (fewer peaks detected)
python peakpicker.py --gage 03408500 --threshold 0.7
```

### Adjust Peak Spacing

```bash
# Peaks must be at least 24 hours apart
python peakpicker.py --gage 03408500 --min-distance 24

# Peaks must be at least 72 hours apart
python peakpicker.py --gage 03408500 --min-distance 72
```

### Process Multiple Gages

```bash
# Create a simple bash loop
for gage in 03408500 03409500 03410210; do
    python peakpicker.py --gage $gage
done
```

## Interpreting Results

### Peak Probability

- **0.9 - 1.0**: Very confident this is a peak
- **0.7 - 0.9**: Likely a peak
- **0.5 - 0.7**: Moderate confidence (default threshold is 0.5)
- **0.3 - 0.5**: Low confidence
- **0.0 - 0.3**: Unlikely to be a peak

### Hydrograph Plot

- **Blue line**: Streamflow discharge
- **Red circles**: Detected peaks
- **Green triangles**: Manual peaks (if available)
- **Orange dashed lines**: Return period thresholds
- **Purple shading**: Peak probability (0-1)

### Performance Metrics

- **Precision**: What % of detected peaks are correct?
- **Recall**: What % of actual peaks were detected?
- **F1-Score**: Overall balance of precision and recall

Good performance: F1-Score > 0.75

## Troubleshooting

### "Model not found" error

```bash
# Solution: Train the model first
python peakpicker.py --train
```

### "Gage data not found" error

Check:
1. File exists in `gages/` directory
2. File name format: `{gage_number}_Obs.csv`
3. Try with/without leading zeros (e.g., `03408500` vs `3408500`)

### Pandas deprecation warning about `fillna(method=...)`

This is a known warning in the current version and will be fixed in a future update. It doesn't affect functionality.

### No peaks detected

Try:
1. Lower the threshold: `--threshold 0.3`
2. Check your data has actual flood events
3. Verify streamflow values are reasonable

### Too many peaks detected

Try:
1. Raise the threshold: `--threshold 0.7`
2. Increase minimum distance: `--min-distance 72`

## What's Next?

1. **Review the plots** to understand model behavior
2. **Adjust thresholds** based on your specific needs
3. **Process all your gages** once satisfied with performance
4. **Compare results** with your manual picks (if available)

## Deactivate Virtual Environment

When you're done:

```bash
deactivate
```

## Getting Help

```bash
# Show all available options
python peakpicker.py --help
```

For detailed documentation, see [README.md](README.md)
