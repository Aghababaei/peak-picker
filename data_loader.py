"""
Data loading and preprocessing module for peak picking
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class DataLoader:
    """Handles loading and preprocessing of gage data, peaks, and return periods"""

    def __init__(self, base_dir: str = '.'):
        self.base_dir = Path(base_dir)
        self.gages_dir = self.base_dir / 'gages'
        self.manual_peaks_file = self.base_dir / 'manual_added_peaks.csv'
        self.return_periods_file = self.base_dir / 'return_periods.csv'

        # Cache for loaded data
        self._return_periods_df = None
        self._manual_peaks_df = None
        self._comid_to_siteno_map = None

    def normalize_gage_number(self, gage_num: str) -> str:
        """Normalize gage number by removing leading zeros"""
        return str(gage_num).lstrip('0')

    def normalize_gage_number_with_variants(self, gage_num: str) -> List[str]:
        """Get all possible variants of a gage number (with/without leading zeros)"""
        gage_str = str(gage_num)
        variants = [
            gage_str,  # Original
            gage_str.lstrip('0'),  # Without leading zeros
            gage_str.zfill(8),  # 8-digit padded
        ]
        return list(set(variants))  # Remove duplicates

    def load_return_periods(self) -> pd.DataFrame:
        """Load return periods data"""
        if self._return_periods_df is None:
            self._return_periods_df = pd.read_csv(self.return_periods_file)
            print(f"Loaded {len(self._return_periods_df)} return period records")
        return self._return_periods_df

    def load_manual_peaks(self) -> pd.DataFrame:
        """Load manually selected peaks"""
        if self._manual_peaks_df is None:
            self._manual_peaks_df = pd.read_csv(self.manual_peaks_file)
            self._manual_peaks_df['peak_time_utc'] = pd.to_datetime(
                self._manual_peaks_df['peak_time_utc']
            )
            # Normalize site_no for easier matching
            self._manual_peaks_df['site_no_normalized'] = (
                self._manual_peaks_df['site_no'].astype(str).str.lstrip('0')
            )
            print(f"Loaded {len(self._manual_peaks_df)} manual peak records")
        return self._manual_peaks_df

    def build_comid_siteno_map(self) -> Dict[int, str]:
        """Build mapping from comid to site_no"""
        if self._comid_to_siteno_map is None:
            manual_peaks = self.load_manual_peaks()
            self._comid_to_siteno_map = (
                manual_peaks[['comid', 'site_no']]
                .drop_duplicates()
                .set_index('comid')['site_no']
                .to_dict()
            )
        return self._comid_to_siteno_map

    def get_return_periods_for_gage(self, site_no: str) -> Optional[Dict[str, float]]:
        """Get return periods for a specific gage"""
        return_periods = self.load_return_periods()
        manual_peaks = self.load_manual_peaks()

        # Normalize the site number
        site_no_normalized = str(site_no).lstrip('0')

        # Find comid for this site_no
        site_peaks = manual_peaks[
            manual_peaks['site_no_normalized'] == site_no_normalized
        ]

        if len(site_peaks) == 0:
            return None

        comid = site_peaks['comid'].iloc[0]

        # Get return periods for this comid
        rp_row = return_periods[return_periods['feature_id'] == comid]

        if len(rp_row) == 0:
            return None

        return {
            'return_period_2': rp_row['return_period_2'].iloc[0],
            'return_period_5': rp_row['return_period_5'].iloc[0],
            'return_period_10': rp_row['return_period_10'].iloc[0],
            'return_period_25': rp_row['return_period_25'].iloc[0],
            'return_period_50': rp_row['return_period_50'].iloc[0],
            'return_period_100': rp_row['return_period_100'].iloc[0],
        }

    def load_gage_data(self, site_no: str) -> Optional[pd.DataFrame]:
        """Load streamflow data for a specific gage"""
        # Try different variants of the gage number
        variants = self.normalize_gage_number_with_variants(site_no)

        for variant in variants:
            file_path = self.gages_dir / f"{variant}_Obs.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                df['datetime_utc'] = pd.to_datetime(df['datetime_utc'])

                # Clean discharge data - convert non-numeric values to NaN
                # Handle USGS quality codes: 'Ice', 'Eqp', 'Dis', 'Mnt', etc.
                if 'discharge_cms' in df.columns:
                    df['discharge_cms'] = pd.to_numeric(df['discharge_cms'], errors='coerce')
                    # Remove rows with missing discharge
                    df = df.dropna(subset=['discharge_cms'])

                df = df.sort_values('datetime_utc').reset_index(drop=True)
                df['site_no_normalized'] = str(site_no).lstrip('0')
                return df

        print(f"Warning: Could not find data file for gage {site_no}")
        return None

    def get_manual_peaks_for_gage(self, site_no: str) -> pd.DataFrame:
        """Get manually selected peaks for a specific gage"""
        manual_peaks = self.load_manual_peaks()
        site_no_normalized = str(site_no).lstrip('0')

        peaks = manual_peaks[
            manual_peaks['site_no_normalized'] == site_no_normalized
        ].copy()

        return peaks

    def get_all_available_gages(self) -> List[str]:
        """Get list of all available gage numbers from files"""
        gage_files = list(self.gages_dir.glob("*_Obs.csv"))
        gage_numbers = [f.stem.replace('_Obs', '') for f in gage_files]
        return sorted(gage_numbers)

    def get_gages_with_return_periods(self) -> List[str]:
        """Get list of gages that have return period data"""
        return_periods = self.load_return_periods()
        manual_peaks = self.load_manual_peaks()
        comid_to_siteno = self.build_comid_siteno_map()

        # Get all comids that have return periods
        available_comids = set(return_periods['feature_id'].unique())

        # Get site_nos for these comids
        gages_with_rp = []
        for comid in available_comids:
            if comid in comid_to_siteno:
                site_no = comid_to_siteno[comid]
                gages_with_rp.append(str(site_no))

        return sorted(gages_with_rp)

    def load_training_data(self) -> Tuple[List[pd.DataFrame], List[pd.DataFrame]]:
        """
        Load all training data (gages with return periods and manual peaks)

        Returns:
            Tuple of (gage_data_list, peaks_list)
        """
        gages_with_rp = self.get_gages_with_return_periods()

        gage_data_list = []
        peaks_list = []

        for site_no in gages_with_rp:
            gage_data = self.load_gage_data(site_no)
            if gage_data is not None:
                peaks = self.get_manual_peaks_for_gage(site_no)
                if len(peaks) > 0:
                    gage_data_list.append(gage_data)
                    peaks_list.append(peaks)

        print(f"Loaded training data for {len(gage_data_list)} gages")
        return gage_data_list, peaks_list


if __name__ == "__main__":
    # Test the data loader
    loader = DataLoader()

    print("\n=== Testing Data Loader ===\n")

    # Test loading return periods
    print("1. Loading return periods...")
    rp_df = loader.load_return_periods()
    print(f"   Shape: {rp_df.shape}")
    print(f"   Columns: {rp_df.columns.tolist()}\n")

    # Test loading manual peaks
    print("2. Loading manual peaks...")
    peaks_df = loader.load_manual_peaks()
    print(f"   Shape: {peaks_df.shape}")
    print(f"   Columns: {peaks_df.columns.tolist()}\n")

    # Test loading a specific gage
    print("3. Loading specific gage (03408500)...")
    gage_data = loader.load_gage_data('03408500')
    if gage_data is not None:
        print(f"   Shape: {gage_data.shape}")
        print(f"   Date range: {gage_data['datetime_utc'].min()} to {gage_data['datetime_utc'].max()}\n")

    # Test getting return periods for a gage
    print("4. Getting return periods for gage 03408500...")
    rp = loader.get_return_periods_for_gage('03408500')
    if rp:
        print(f"   Return periods: {rp}\n")

    # Test getting manual peaks for a gage
    print("5. Getting manual peaks for gage 03408500...")
    peaks = loader.get_manual_peaks_for_gage('03408500')
    print(f"   Number of peaks: {len(peaks)}\n")

    # Test getting gages with return periods
    print("6. Getting gages with return periods...")
    gages_with_rp = loader.get_gages_with_return_periods()
    print(f"   Number of gages: {len(gages_with_rp)}")
    print(f"   First 10 gages: {gages_with_rp[:10]}\n")
