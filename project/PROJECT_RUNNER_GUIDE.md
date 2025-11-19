# Project Runner Guide

This guide explains how to use `run_project.py` to process gage files from your custom project folder.

## Overview

`run_project.py` is a standalone script that uses the PeakPicker package to process gage files from a custom folder structure without modifying the core package.

## Project Structure

```
project/
├── gages/          # Input CSV files (datetime, discharge columns)
├── data/           # Mapping and return period data
│   ├── usgsid_comid.csv       # Maps USGS IDs to COMIDs
│   └── return_periods.csv     # Return period values by COMID
├── plots/          # Output plots (auto-created)
└── results/        # Output CSV files (auto-created)
```

## Input Data Format

### Gage Files (`gages/*.csv`)
```csv
datetime,discharge
2023-09-20 00:00,2250
2023-09-20 00:15,2230
```
- **Columns**: `datetime`, `discharge` (NOT discharge_cms)
- **Filename**: USGS ID with .csv extension (e.g., `01200600.csv`)

### USGS-COMID Mapping (`data/usgsid_comid.csv`)
```csv
USGSID,COMID
01154950,9331072
14330000,23923628
```
- Maps USGS gage IDs to NWM COMID values
- Handles IDs with/without leading zeros automatically

### Return Periods (`data/return_periods.csv`)
```csv
feature_id,return_period_2,return_period_5,return_period_10,return_period_25,return_period_50,return_period_100
1239619,7.08,11.93,15.14,19.2,22.21,25.19
```
- `feature_id` is the COMID
- Return period values in cms

## Usage

### Process All Gages
```bash
source venv/bin/activate
python run_project.py
```

### Process Single Gage
```bash
python run_project.py --gage 01200600
```

### Custom Threshold
```bash
python run_project.py --threshold 0.6
```
Lower threshold = more peaks detected

### Custom Minimum Distance
```bash
python run_project.py --min-distance 72
```
Minimum hours between detected peaks (default: 48)

### Custom Project Directory
```bash
python run_project.py --project-dir /path/to/your/project
```

### All Options
```bash
python run_project.py \
  --project-dir project \
  --model-path peak_model.pkl \
  --threshold 0.5 \
  --min-distance 48 \
  --gage 01200600
```

## Output Files

### Individual Gage Results
**Location**: `project/results/{USGSID}_detected_peaks.csv`

**Format**:
```csv
peak_time_utc,peak_flow_cms,peak_probability,site_no
2023-09-25 17:00:00,6050.0,0.858,01200600
2023-09-30 13:30:00,8940.0,0.986,01200600
```

### Combined Results
**Location**: `project/results/all_detected_peaks.csv`

Contains all peaks from all gages in a single file.

### Plots
**Location**: `project/plots/`

For each gage:
- `{USGSID}_hydrograph.png` - Full hydrograph with colored return period zones and detected peaks
- `{USGSID}_annual_maxima.png` - Annual maximum flow plot

## Recent Run Summary

**Date**: November 19, 2025
**Gages Processed**: 9
**Total Peaks Detected**: 337

| USGS ID | Peaks | Notes |
|---------|-------|-------|
| 01200600 | 21 | 717 non-numeric values removed |
| 01201487 | 46 | 76 non-numeric values removed |
| 01202501 | 30 | No data issues |
| 01203510 | 49 | 1,270 non-numeric values removed |
| 01203600 | 36 | 3,496 non-numeric values removed |
| 01203805 | 40 | 3,023 non-numeric values removed |
| 01204000 | 41 | 213 non-numeric values removed |
| 01205500 | 40 | No data issues |
| 01206900 | 34 | 1,966 non-numeric values removed |

## Return Period Matching

The script automatically:
1. Extracts USGS ID from filename (e.g., `01200600.csv` → `01200600`)
2. Looks up COMID in `usgsid_comid.csv` (handles leading zeros)
3. Finds return periods in `return_periods.csv` using COMID
4. If not found, calculates statistical estimates from data

**Note**: If return periods aren't found, the script still works by calculating estimates from the data itself, but results may be less accurate.

## Troubleshooting

### "No return periods found for USGS XXXXX"
This is normal if the gage isn't in your mapping files. The script will use statistical estimates instead.

### "Removed N non-numeric discharge values"
Common USGS qualifiers like "Ice", "Eqp", "Dis" are automatically cleaned. This is normal for winter/equipment issues.

### Model Not Found
Make sure you've trained a model first:
```bash
python peakpicker.py --train --model-type random_forest
```

### Wrong Columns in Gage Files
Gage CSV files must have columns named exactly `datetime` and `discharge` (case-sensitive).

## Performance

- **Single gage**: ~30-60 seconds
- **All 9 gages**: ~5-8 minutes
- **Memory usage**: ~1-2 GB
- **CPU**: Uses all available cores

## Advanced Customization

If you need to modify the script behavior, edit `run_project.py`:
- Line 17: Window sizes for feature engineering
- Line 243: Threshold and distance filtering
- Line 330: Plot styling
- Line 372: Peak filtering logic

## Next Steps

1. **Review Results**: Check `project/results/all_detected_peaks.csv`
2. **Inspect Plots**: Open plots in `project/plots/` to visually verify peaks
3. **Adjust Threshold**: If too many/few peaks, adjust `--threshold`
4. **Compare with Manual Peaks**: If you have manual peak labels, compare against detected peaks
5. **Batch Process**: Run on all your gages at once for consistency

## Example Workflow

```bash
# 1. Activate environment
cd /Users/amin/Downloads/peakpicker
source venv/bin/activate

# 2. Test on single gage first
python run_project.py --gage 01200600

# 3. Review the plot
open project/plots/01200600_hydrograph.png

# 4. If results look good, process all gages
python run_project.py

# 5. Check combined results
head project/results/all_detected_peaks.csv

# 6. (Optional) Adjust threshold and re-run
python run_project.py --threshold 0.6
```

## Important Notes

- **Non-Destructive**: This script doesn't modify any core PeakPicker files
- **Reusable**: Can be run multiple times on the same data
- **Parallel Processing**: Could be extended to process gages in parallel
- **Custom Formats**: Easy to adapt for different input formats by editing the script

## Support

- Main package documentation: `README.md`
- Installation help: `SETUP_INSTRUCTIONS.md`
- Quick start guide: `QUICKSTART.md`
- Project summary: `PROJECT_SUMMARY.md`

---
**Script Location**: `/Users/amin/Downloads/peakpicker/run_project.py`
**Last Updated**: November 19, 2025
