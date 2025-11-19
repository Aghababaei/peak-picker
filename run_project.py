#!/usr/bin/env python3
"""
Project-specific runner for PeakPicker
Processes gage files from project folder with custom return periods
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import existing PeakPicker modules
from feature_engineering import FeatureEngineer
from model_trainer import PeakDetectionModel
from plotter import HydrographPlotter


class ProjectPeakPicker:
    """
    Project-specific peak picker that uses custom folder structure
    """

    def __init__(self, project_dir: str = 'project'):
        """
        Initialize project runner

        Args:
            project_dir: Path to project directory
        """
        self.project_dir = Path(project_dir)
        self.gages_dir = self.project_dir / 'gages'
        self.data_dir = self.project_dir / 'data'
        self.plots_dir = self.project_dir / 'plots'
        self.results_dir = self.project_dir / 'results'

        # Create output directories if they don't exist
        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Initialize components
        self.engineer = FeatureEngineer()
        self.model = None
        self.plotter = HydrographPlotter()

        # Override plotter output directory
        self.plotter.output_dir = self.plots_dir

        # Load mapping data
        self.usgs_comid_map = self._load_usgs_comid_mapping()
        self.return_periods = self._load_return_periods()

        print(f"Project directory: {self.project_dir}")
        print(f"Loaded {len(self.usgs_comid_map)} USGS-COMID mappings")
        print(f"Loaded {len(self.return_periods)} return period records")

    def _normalize_usgs_id(self, usgs_id: str) -> str:
        """Normalize USGS ID to 8 digits with leading zeros"""
        usgs_id = str(usgs_id).strip()
        if usgs_id.isdigit():
            return usgs_id.zfill(8)
        return usgs_id

    def _load_usgs_comid_mapping(self) -> dict:
        """
        Load USGS ID to COMID mapping
        Returns dict with both formats of USGS ID as keys
        """
        mapping_file = self.data_dir / 'usgsid_comid.csv'

        if not mapping_file.exists():
            print(f"Warning: Mapping file not found: {mapping_file}")
            return {}

        df = pd.read_csv(mapping_file)

        # Create mapping dictionary with both normalized and original keys
        mapping = {}
        for _, row in df.iterrows():
            usgs_id = str(row['USGSID']).strip()
            comid = str(row['COMID']).strip()

            # Add with original format
            mapping[usgs_id] = comid

            # Add with normalized format (8 digits)
            normalized = self._normalize_usgs_id(usgs_id)
            mapping[normalized] = comid

            # Also add version without leading zeros
            mapping[usgs_id.lstrip('0')] = comid

        return mapping

    def _load_return_periods(self) -> dict:
        """
        Load return periods by COMID (feature_id)
        Returns dict mapping COMID -> return period dictionary
        """
        rp_file = self.data_dir / 'return_periods.csv'

        if not rp_file.exists():
            print(f"Warning: Return periods file not found: {rp_file}")
            return {}

        df = pd.read_csv(rp_file)

        # Create dictionary mapping COMID to return periods
        rp_dict = {}
        for _, row in df.iterrows():
            # Convert to int first to remove .0, then to string
            comid = str(int(float(row['feature_id']))).strip()
            rp_dict[comid] = {
                'return_period_2': float(row['return_period_2']),
                'return_period_5': float(row['return_period_5']),
                'return_period_10': float(row['return_period_10']),
                'return_period_25': float(row['return_period_25']),
                'return_period_50': float(row['return_period_50']),
                'return_period_100': float(row['return_period_100'])
            }

        return rp_dict

    def _get_return_periods_for_usgs(self, usgs_id: str) -> dict:
        """
        Get return periods for a USGS gage

        Args:
            usgs_id: USGS gage ID

        Returns:
            Dictionary of return periods or None
        """
        # Try to find COMID for this USGS ID
        comid = None

        # Try with original ID
        if usgs_id in self.usgs_comid_map:
            comid = self.usgs_comid_map[usgs_id]
        # Try normalized (8 digits)
        elif self._normalize_usgs_id(usgs_id) in self.usgs_comid_map:
            comid = self.usgs_comid_map[self._normalize_usgs_id(usgs_id)]
        # Try without leading zeros
        elif usgs_id.lstrip('0') in self.usgs_comid_map:
            comid = self.usgs_comid_map[usgs_id.lstrip('0')]

        if comid and comid in self.return_periods:
            return self.return_periods[comid]

        return None

    def load_model(self, model_path: str = 'peak_model.pkl'):
        """Load trained model"""
        model_file = Path(model_path)

        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                f"Please train a model first using: python peakpicker.py --train"
            )

        self.model = PeakDetectionModel.load_model(str(model_file))
        print(f"Model loaded from {model_path}")

    def load_gage_file(self, file_path: Path) -> pd.DataFrame:
        """
        Load a gage CSV file

        Args:
            file_path: Path to CSV file

        Returns:
            DataFrame with datetime_utc and discharge_cms columns
        """
        df = pd.read_csv(file_path)

        # Expected columns: datetime, discharge
        if 'datetime' not in df.columns or 'discharge' not in df.columns:
            raise ValueError(
                f"File {file_path.name} must have 'datetime' and 'discharge' columns. "
                f"Found: {df.columns.tolist()}"
            )

        # Rename to standard format
        df = df.rename(columns={
            'datetime': 'datetime_utc',
            'discharge': 'discharge_cms'
        })

        # Convert datetime
        df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

        # Clean discharge data - convert non-numeric values to NaN
        df['discharge_cms'] = pd.to_numeric(df['discharge_cms'], errors='coerce')

        # Count and report removed values
        n_missing = df['discharge_cms'].isna().sum()
        if n_missing > 0:
            print(f"  Removed {n_missing} non-numeric discharge values")

        # Remove rows with missing discharge
        df = df.dropna(subset=['discharge_cms'])

        df = df.sort_values('datetime_utc').reset_index(drop=True)

        return df

    def process_gage(self, gage_file: Path, threshold: float = 0.5,
                     min_distance_hours: int = 48) -> pd.DataFrame:
        """
        Process a single gage file

        Args:
            gage_file: Path to gage CSV file
            threshold: Probability threshold for peak detection
            min_distance_hours: Minimum hours between peaks

        Returns:
            DataFrame with detected peaks
        """
        # Extract USGS ID from filename (remove .csv extension)
        usgs_id = gage_file.stem

        print(f"\n{'='*60}")
        print(f"Processing: {gage_file.name} (USGS ID: {usgs_id})")
        print(f"{'='*60}")

        # Load gage data
        df = self.load_gage_file(gage_file)
        print(f"Loaded {len(df)} time steps")
        print(f"Date range: {df['datetime_utc'].min()} to {df['datetime_utc'].max()}")

        # Get return periods for this gage
        return_periods = self._get_return_periods_for_usgs(usgs_id)

        if return_periods:
            print(f"Found return periods for USGS {usgs_id}")
        else:
            print(f"No return periods found for USGS {usgs_id}, will use statistical estimates")

        # Engineer features
        print("Engineering features...")
        df_features = self.engineer.engineer_features(
            df,
            return_periods=return_periods,
            value_col='discharge_cms'
        )

        # Get feature columns
        feature_cols = self.engineer.get_feature_columns(df_features)

        # Predict
        print("Detecting peaks...")
        predictions, probabilities = self.model.predict(df_features[feature_cols])

        # Add predictions to dataframe
        df_features['is_peak_predicted'] = predictions
        df_features['peak_probability'] = probabilities

        # Filter by threshold
        potential_peaks = df_features[
            df_features['peak_probability'] >= threshold
        ].copy()

        print(f"Found {len(potential_peaks)} potential peaks (threshold: {threshold})")

        # Apply distance filtering
        if len(potential_peaks) > 0 and min_distance_hours > 0:
            detected_peaks = self._filter_peaks_by_distance(
                potential_peaks,
                min_distance_hours
            )
            print(f"After distance filtering: {len(detected_peaks)} peaks")
        else:
            detected_peaks = potential_peaks

        # Create output dataframe
        if len(detected_peaks) > 0:
            output_peaks = detected_peaks[[
                'datetime_utc', 'discharge_cms', 'peak_probability'
            ]].copy()
            output_peaks.columns = ['peak_time_utc', 'peak_flow_cms', 'peak_probability']
            output_peaks['site_no'] = usgs_id

            # Display peaks
            print("\nDetected Peaks:")
            print("-" * 60)
            for _, row in output_peaks.iterrows():
                print(f"  {row['peak_time_utc']}: {row['peak_flow_cms']:.2f} cms "
                      f"(probability: {row['peak_probability']:.3f})")
        else:
            print("\nNo peaks detected.")
            output_peaks = pd.DataFrame(columns=[
                'peak_time_utc', 'peak_flow_cms', 'peak_probability', 'site_no'
            ])

        # Create plots
        print("\nCreating plots...")
        self.plotter.plot_hydrograph_with_peaks(
            df=df_features,
            detected_peaks=detected_peaks if len(detected_peaks) > 0 else None,
            return_periods=return_periods,
            site_no=usgs_id,
            show_probability=False
        )

        self.plotter.plot_annual_maxima(
            df=df,
            return_periods=return_periods,
            site_no=usgs_id
        )

        # Save results
        output_file = self.results_dir / f'{usgs_id}_detected_peaks.csv'
        output_peaks.to_csv(output_file, index=False)
        print(f"Results saved to {output_file}")

        return output_peaks

    def _filter_peaks_by_distance(self, peaks_df: pd.DataFrame,
                                   min_distance_hours: int) -> pd.DataFrame:
        """
        Filter peaks to ensure minimum time distance between them
        Keep peaks with highest probability when conflicts arise
        """
        if len(peaks_df) == 0:
            return peaks_df

        # Sort by probability (highest first)
        peaks_sorted = peaks_df.sort_values(
            'peak_probability',
            ascending=False
        ).copy()

        # Initialize list of kept peaks
        kept_peaks = []

        for idx, peak in peaks_sorted.iterrows():
            peak_time = peak['datetime_utc']

            # Check if this peak is far enough from all kept peaks
            is_far_enough = True
            for kept_peak in kept_peaks:
                time_diff = abs((peak_time - kept_peak['datetime_utc']).total_seconds() / 3600)
                if time_diff < min_distance_hours:
                    is_far_enough = False
                    break

            if is_far_enough:
                kept_peaks.append(peak)

        # Convert back to DataFrame
        if kept_peaks:
            result_df = pd.DataFrame(kept_peaks)
            result_df = result_df.sort_values('datetime_utc').reset_index(drop=True)
            return result_df
        else:
            return pd.DataFrame(columns=peaks_df.columns)

    def process_all_gages(self, threshold: float = 0.5,
                          min_distance_hours: int = 48):
        """
        Process all gage files in the gages directory

        Args:
            threshold: Probability threshold for peak detection
            min_distance_hours: Minimum hours between peaks
        """
        # Get all CSV files in gages directory
        gage_files = sorted(self.gages_dir.glob('*.csv'))

        if len(gage_files) == 0:
            print(f"No CSV files found in {self.gages_dir}")
            return

        print(f"\n{'='*60}")
        print(f"PROCESSING {len(gage_files)} GAGE FILES")
        print(f"{'='*60}")

        all_results = []

        for gage_file in gage_files:
            try:
                results = self.process_gage(
                    gage_file,
                    threshold=threshold,
                    min_distance_hours=min_distance_hours
                )
                all_results.append(results)
            except Exception as e:
                print(f"\nError processing {gage_file.name}: {e}")
                import traceback
                traceback.print_exc()
                continue

        # Combine all results
        if all_results:
            combined_results = pd.concat(all_results, ignore_index=True)
            combined_file = self.results_dir / 'all_detected_peaks.csv'
            combined_results.to_csv(combined_file, index=False)
            print(f"\n{'='*60}")
            print(f"PROCESSING COMPLETE")
            print(f"{'='*60}")
            print(f"Total peaks detected: {len(combined_results)}")
            print(f"Combined results saved to {combined_file}")
        else:
            print("\nNo results to save.")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(
        description='Process gage files from project directory',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all gages in project folder
  python run_project.py

  # Process with custom threshold
  python run_project.py --threshold 0.6

  # Process single gage
  python run_project.py --gage 01200600

  # Use custom project directory
  python run_project.py --project-dir /path/to/project
        """
    )

    parser.add_argument('--project-dir', type=str, default='project',
                       help='Path to project directory (default: project)')
    parser.add_argument('--model-path', type=str, default='peak_model.pkl',
                       help='Path to trained model (default: peak_model.pkl)')
    parser.add_argument('--gage', type=str,
                       help='Process single gage by USGS ID (e.g., 01200600)')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Peak probability threshold (default: 0.5)')
    parser.add_argument('--min-distance', type=int, default=48,
                       help='Minimum hours between peaks (default: 48)')

    args = parser.parse_args()

    # Initialize project runner
    runner = ProjectPeakPicker(project_dir=args.project_dir)

    # Load model
    runner.load_model(model_path=args.model_path)

    # Process gages
    if args.gage:
        # Process single gage
        gage_file = runner.gages_dir / f'{args.gage}.csv'
        if not gage_file.exists():
            print(f"Error: Gage file not found: {gage_file}")
            return

        runner.process_gage(
            gage_file,
            threshold=args.threshold,
            min_distance_hours=args.min_distance
        )
    else:
        # Process all gages
        runner.process_all_gages(
            threshold=args.threshold,
            min_distance_hours=args.min_distance
        )


if __name__ == '__main__':
    main()
