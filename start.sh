#!/bin/bash

# PeakPicker Startup Script
# This script helps you get started with PeakPicker

echo "============================================================"
echo "PeakPicker - Automated Flood Peak Detection"
echo "============================================================"
echo ""

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

if [ $? -ne 0 ]; then
    echo "Error: Could not activate virtual environment"
    echo "Make sure you're in the peakpicker directory"
    exit 1
fi

echo "✓ Virtual environment activated"
echo ""

# Check if model exists
if [ -f "peak_model.pkl" ]; then
    echo "✓ Model file found: peak_model.pkl"
    echo ""
    echo "You can now run peak detection:"
    echo "  python peakpicker.py --gage 03408500"
    echo ""
else
    echo "⚠ Model file not found"
    echo ""
    echo "You need to train the model first."
    echo ""
    echo "Recommended (works on all systems):"
    echo "  python peakpicker.py --train --model-type lightgbm"
    echo ""
    echo "Alternative (requires libomp on macOS):"
    echo "  python peakpicker.py --train --model-type xgboost"
    echo ""
    echo "Would you like to train the model now? (y/n)"
    read -p "> " response

    if [ "$response" = "y" ] || [ "$response" = "Y" ]; then
        echo ""
        echo "Training model with LightGBM (recommended)..."
        python peakpicker.py --train --model-type lightgbm

        if [ $? -eq 0 ]; then
            echo ""
            echo "✓ Model trained successfully!"
            echo ""
            echo "Next steps:"
            echo "  1. Test on a gage: python peakpicker.py --gage 03408500"
            echo "  2. See QUICKSTART.md for more examples"
        else
            echo ""
            echo "✗ Model training failed"
            echo "See error messages above for details"
        fi
    else
        echo ""
        echo "Skipping training. Run the command manually when ready."
    fi
fi

echo ""
echo "============================================================"
echo "Quick Commands:"
echo "  --help           Show all options"
echo "  --train          Train a new model"
echo "  --gage NUMBER    Detect peaks for a gage"
echo "  --file PATH      Process a new file"
echo ""
echo "For detailed instructions: cat QUICKSTART.md"
echo "For installation help: cat INSTALL.md"
echo "============================================================"
echo ""

# Keep shell active with venv
exec $SHELL
