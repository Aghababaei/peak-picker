#!/usr/bin/env python3
"""
PeakPicker - Automated Peak Detection for Hydrograph Data
Main script for training models and detecting flood peaks
"""
import argparse
import pandas as pd
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from feature_engineering import FeatureEngineer
from return_period_calculator import ReturnPeriodCalculator
from model_trainer import PeakDetectionModel, train_peak_detection_model
from plotter import HydrographPlotter


class PeakPicker:
    """Main class for automated peak picking"""

    def __init__(self, model_path: str = 'peak_model.pkl'):
        self.model_path = model_path
        self.loader = DataLoader()
        self.engineer = FeatureEngineer()
        self.calculator = ReturnPeriodCalculator()
        self.plotter = HydrographPlotter()
        self.model = None

    def train_model(self, model_type: str = 'random_forest', test_size: float = 0.2):
        """
        Train a new peak detection model

        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm')
            test_size: Fraction of data for testing
        """
        print("\n" + "="*60)
        print("TRAINING PEAK DETECTION MODEL")
        print("="*60 + "\n")

        self.model = train_peak_detection_model(
            model_type=model_type,
            test_size=test_size,
            save_path=self.model_path
        )

        print("\n" + "="*60)
        print("MODEL TRAINING COMPLETE")
        print("="*60 + "\n")

    def load_model(self):
        """Load a pre-trained model"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(
                f"Model file not found: {self.model_path}\n"
                "Please train a model first using the --train flag"
            )

        self.model = PeakDetectionModel.load_model(self.model_path)

    def detect_peaks_for_gage(
        self,
        site_no: str,
        probability_threshold: float = 0.5,
        min_peak_distance_hours: int = 48,
        plot: bool = True,
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Detect peaks for a specific gage

        Args:
            site_no: Gage site number
            probability_threshold: Threshold for peak detection (0-1)
            min_peak_distance_hours: Minimum time between peaks
            plot: Whether to create plots
            output_csv: Path to save detected peaks CSV

        Returns:
            DataFrame with detected peaks
        """
        print("\n" + "="*60)
        print(f"DETECTING PEAKS FOR GAGE {site_no}")
        print("="*60 + "\n")

        # Load gage data
        print("Loading gage data...")
        gage_data = self.loader.load_gage_data(site_no)
        if gage_data is None:
            raise ValueError(f"Could not load data for gage {site_no}")

        print(f"Data loaded: {len(gage_data)} time steps")
        print(f"Date range: {gage_data['datetime_utc'].min()} to {gage_data['datetime_utc'].max()}")

        # Get or calculate return periods
        print("\nGetting return periods...")
        return_periods = self.loader.get_return_periods_for_gage(site_no)

        if return_periods is None:
            print("No pre-calculated return periods found. Calculating from data...")
            return_periods = self.calculator.calculate_return_periods(
                gage_data,
                method='mixed'
            )

            if return_periods:
                print("Calculated return periods:")
                for key, value in return_periods.items():
                    print(f"  {key}: {value:.2f} cms")
            else:
                print("Warning: Could not calculate return periods")
        else:
            print("Using pre-calculated return periods:")
            for key, value in return_periods.items():
                print(f"  {key}: {value:.2f} cms")

        # Engineer features
        print("\nEngineering features...")
        gage_features = self.engineer.engineer_features(
            gage_data,
            return_periods=return_periods
        )

        # Load model if not already loaded
        if self.model is None:
            print("\nLoading model...")
            self.load_model()

        # Predict peaks
        print("\nPredicting peaks...")
        feature_cols = self.engineer.get_feature_columns(gage_features)
        X = gage_features[feature_cols]

        y_pred, y_pred_proba = self.model.predict(X)

        # Add predictions to dataframe
        gage_features['is_predicted_peak'] = y_pred
        gage_features['peak_probability'] = y_pred_proba

        # Filter peaks based on probability threshold
        potential_peaks = gage_features[
            gage_features['peak_probability'] >= probability_threshold
        ].copy()

        print(f"Found {len(potential_peaks)} potential peaks (threshold: {probability_threshold})")

        # Apply minimum distance constraint
        if len(potential_peaks) > 0:
            detected_peaks = self._filter_peaks_by_distance(
                potential_peaks,
                min_hours=min_peak_distance_hours
            )

            print(f"After distance filtering: {len(detected_peaks)} peaks")
            print(f"(Minimum distance: {min_peak_distance_hours} hours)")
        else:
            detected_peaks = potential_peaks

        # Get manual peaks if available (for comparison)
        manual_peaks = self.loader.get_manual_peaks_for_gage(site_no)

        # Display detected peaks
        if len(detected_peaks) > 0:
            print("\nDetected Peaks:")
            print("-" * 60)
            for _, row in detected_peaks.iterrows():
                print(f"  {row['datetime_utc']}: {row['discharge_cms']:.2f} cms "
                      f"(probability: {row['peak_probability']:.3f})")
        else:
            print("\nNo peaks detected.")

        # Compare with manual peaks if available
        if len(manual_peaks) > 0:
            print(f"\nManual peaks available: {len(manual_peaks)}")
            self._compare_peaks(detected_peaks, manual_peaks)

        # Create plots
        if plot:
            print("\nCreating plots...")

            # Main hydrograph plot
            self.plotter.plot_hydrograph_with_peaks(
                df=gage_features,
                detected_peaks=detected_peaks,
                manual_peaks=manual_peaks if len(manual_peaks) > 0 else None,
                return_periods=return_periods,
                site_no=site_no,
                show_probability=False  # Removed probability overlay
            )

            # Comparison plot if manual peaks available
            if len(manual_peaks) > 0 and len(detected_peaks) > 0:
                self.plotter.plot_peak_comparison(
                    detected_peaks=detected_peaks,
                    manual_peaks=manual_peaks,
                    site_no=site_no
                )

            # Annual maxima plot
            self.plotter.plot_annual_maxima(
                df=gage_data,
                return_periods=return_periods,
                site_no=site_no
            )

        # Save detected peaks to CSV
        if output_csv or len(detected_peaks) > 0:
            output_path = output_csv if output_csv else f'{site_no}_detected_peaks.csv'

            output_df = detected_peaks[[
                'datetime_utc', 'discharge_cms', 'peak_probability'
            ]].copy()
            output_df.columns = ['peak_time_utc', 'peak_flow_cms', 'peak_probability']
            output_df['site_no'] = site_no

            output_df.to_csv(output_path, index=False)
            print(f"\nDetected peaks saved to: {output_path}")

        print("\n" + "="*60)
        print("PEAK DETECTION COMPLETE")
        print("="*60 + "\n")

        return detected_peaks

    def process_new_gage_file(
        self,
        file_path: str,
        gage_number: str,
        datetime_col: str = 'datetime',
        discharge_col: str = 'streamflow',
        probability_threshold: float = 0.5,
        min_peak_distance_hours: int = 48,
        plot: bool = True,
        output_csv: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Process a new gage file with custom format

        Args:
            file_path: Path to CSV file
            gage_number: Gage number for labeling
            datetime_col: Name of datetime column
            discharge_col: Name of discharge column
            probability_threshold: Threshold for peak detection
            min_peak_distance_hours: Minimum time between peaks
            plot: Whether to create plots
            output_csv: Path to save detected peaks

        Returns:
            DataFrame with detected peaks
        """
        print("\n" + "="*60)
        print(f"PROCESSING NEW GAGE FILE: {file_path}")
        print("="*60 + "\n")

        # Load the file
        df = pd.read_csv(file_path)
        print(f"Loaded {len(df)} time steps")

        # Rename columns to standard format
        df = df.rename(columns={
            datetime_col: 'datetime_utc',
            discharge_col: 'discharge_cms'
        })

        # Convert datetime
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

        # Clean discharge data - convert non-numeric values to NaN
        # Common USGS qualifiers: 'Ice', 'Eqp', 'Dis', 'Mnt', etc.
        df['discharge_cms'] = pd.to_numeric(df['discharge_cms'], errors='coerce')

        # Count and report removed values
        n_missing = df['discharge_cms'].isna().sum()
        if n_missing > 0:
            print(f"Removed {n_missing} non-numeric discharge values (e.g., 'Ice', equipment issues)")

        # Remove rows with missing discharge
        df = df.dropna(subset=['discharge_cms'])
        print(f"Valid discharge records: {len(df)} time steps")

        df = df.sort_values('datetime_utc').reset_index(drop=True)

        print(f"Date range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}")

        # Calculate return periods
        print("\nCalculating return periods from data...")
        return_periods = self.calculator.calculate_return_periods(df, method='mixed')

        if return_periods:
            print("Calculated return periods:")
            for key, value in return_periods.items():
                print(f"  {key}: {value:.2f} cms")

        # Engineer features
        print("\nEngineering features...")
        df_features = self.engineer.engineer_features(df, return_periods=return_periods)

        # Load model
        if self.model is None:
            print("\nLoading model...")
            self.load_model()

        # Predict peaks
        print("\nPredicting peaks...")
        feature_cols = self.engineer.get_feature_columns(df_features)
        X = df_features[feature_cols]

        y_pred, y_pred_proba = self.model.predict(X)

        df_features['is_predicted_peak'] = y_pred
        df_features['peak_probability'] = y_pred_proba

        # Filter peaks
        potential_peaks = df_features[
            df_features['peak_probability'] >= probability_threshold
        ].copy()

        print(f"Found {len(potential_peaks)} potential peaks")

        if len(potential_peaks) > 0:
            detected_peaks = self._filter_peaks_by_distance(
                potential_peaks,
                min_hours=min_peak_distance_hours
            )
            print(f"After distance filtering: {len(detected_peaks)} peaks")
        else:
            detected_peaks = potential_peaks

        # Display peaks
        if len(detected_peaks) > 0:
            print("\nDetected Peaks:")
            print("-" * 60)
            for _, row in detected_peaks.iterrows():
                print(f"  {row['datetime_utc']}: {row['discharge_cms']:.2f} cms "
                      f"(probability: {row['peak_probability']:.3f})")

        # Create plots
        if plot:
            print("\nCreating plots...")
            self.plotter.plot_hydrograph_with_peaks(
                df=df_features,
                detected_peaks=detected_peaks,
                return_periods=return_periods,
                site_no=gage_number,
                show_probability=False  # Removed probability overlay
            )

            self.plotter.plot_annual_maxima(
                df=df,
                return_periods=return_periods,
                site_no=gage_number
            )

        # Save peaks
        if output_csv or len(detected_peaks) > 0:
            output_path = output_csv if output_csv else f'{gage_number}_detected_peaks.csv'

            output_df = detected_peaks[[
                'datetime_utc', 'discharge_cms', 'peak_probability'
            ]].copy()
            output_df.columns = ['peak_time_utc', 'peak_flow_cms', 'peak_probability']
            output_df['site_no'] = gage_number

            output_df.to_csv(output_path, index=False)
            print(f"\nDetected peaks saved to: {output_path}")

        print("\n" + "="*60)
        print("PROCESSING COMPLETE")
        print("="*60 + "\n")

        return detected_peaks

    def _filter_peaks_by_distance(
        self,
        peaks: pd.DataFrame,
        min_hours: int = 48
    ) -> pd.DataFrame:
        """Filter peaks to ensure minimum time distance between them"""
        if len(peaks) == 0:
            return peaks

        # Sort by probability (descending)
        peaks = peaks.sort_values('peak_probability', ascending=False).copy()

        selected_peaks = []
        selected_times = []

        for idx, row in peaks.iterrows():
            peak_time = row['datetime_utc']

            # Check distance to already selected peaks
            if len(selected_times) == 0:
                selected_peaks.append(idx)
                selected_times.append(peak_time)
            else:
                # Calculate minimum time distance to selected peaks
                min_distance = min([
                    abs((peak_time - t).total_seconds() / 3600)
                    for t in selected_times
                ])

                if min_distance >= min_hours:
                    selected_peaks.append(idx)
                    selected_times.append(peak_time)

        # Return selected peaks sorted by time
        result = peaks.loc[selected_peaks].sort_values('datetime_utc')
        return result

    def _compare_peaks(self, detected_peaks: pd.DataFrame, manual_peaks: pd.DataFrame,
                      tolerance_hours: int = 24):
        """Compare detected peaks with manual peaks"""
        if len(detected_peaks) == 0 or len(manual_peaks) == 0:
            return

        matched = 0
        for _, manual_peak in manual_peaks.iterrows():
            manual_time = manual_peak['peak_time_utc']

            # Check if any detected peak is within tolerance
            for _, detected_peak in detected_peaks.iterrows():
                detected_time = detected_peak['datetime_utc']
                time_diff_hours = abs((detected_time - manual_time).total_seconds() / 3600)

                if time_diff_hours <= tolerance_hours:
                    matched += 1
                    break

        precision = matched / len(detected_peaks) if len(detected_peaks) > 0 else 0
        recall = matched / len(manual_peaks) if len(manual_peaks) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print("\nPeak Detection Performance:")
        print(f"  Matched peaks: {matched}")
        print(f"  Precision: {precision:.2%}")
        print(f"  Recall: {recall:.2%}")
        print(f"  F1-Score: {f1:.2%}")


def main():
    """Main entry point for command-line usage"""
    parser = argparse.ArgumentParser(
        description='PeakPicker - Automated Peak Detection for Hydrographs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train a new model
  python peakpicker.py --train

  # Detect peaks for an existing gage
  python peakpicker.py --gage 03408500

  # Process a new gage file
  python peakpicker.py --file data.csv --gage-number 12345678

  # Detect peaks with custom threshold
  python peakpicker.py --gage 03408500 --threshold 0.7
        """
    )

    parser.add_argument('--train', action='store_true',
                       help='Train a new peak detection model')
    parser.add_argument('--model-type', type=str, default='random_forest',
                       choices=['random_forest', 'xgboost', 'lightgbm'],
                       help='Type of model to train (default: random_forest)')
    parser.add_argument('--model-path', type=str, default='peak_model.pkl',
                       help='Path to save/load model (default: peak_model.pkl)')

    parser.add_argument('--gage', type=str,
                       help='Gage number to process (from existing data)')
    parser.add_argument('--file', type=str,
                       help='Path to new gage CSV file to process')
    parser.add_argument('--gage-number', type=str,
                       help='Gage number for new file (required with --file)')

    parser.add_argument('--datetime-col', type=str, default='datetime',
                       help='Name of datetime column in new file (default: datetime)')
    parser.add_argument('--discharge-col', type=str, default='streamflow',
                       help='Name of discharge column in new file (default: streamflow)')

    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Peak probability threshold (0-1, default: 0.5)')
    parser.add_argument('--min-distance', type=int, default=48,
                       help='Minimum hours between peaks (default: 48)')

    parser.add_argument('--no-plot', action='store_true',
                       help='Disable plotting')
    parser.add_argument('--output', type=str,
                       help='Path to save detected peaks CSV')

    args = parser.parse_args()

    # Create PeakPicker instance
    picker = PeakPicker(model_path=args.model_path)

    # Train model
    if args.train:
        picker.train_model(model_type=args.model_type)
        return

    # Detect peaks for existing gage
    if args.gage:
        picker.detect_peaks_for_gage(
            site_no=args.gage,
            probability_threshold=args.threshold,
            min_peak_distance_hours=args.min_distance,
            plot=not args.no_plot,
            output_csv=args.output
        )
        return

    # Process new file
    if args.file:
        if not args.gage_number:
            parser.error("--gage-number is required when using --file")

        picker.process_new_gage_file(
            file_path=args.file,
            gage_number=args.gage_number,
            datetime_col=args.datetime_col,
            discharge_col=args.discharge_col,
            probability_threshold=args.threshold,
            min_peak_distance_hours=args.min_distance,
            plot=not args.no_plot,
            output_csv=args.output
        )
        return

    # If no action specified, show help
    parser.print_help()


if __name__ == "__main__":
    main()
