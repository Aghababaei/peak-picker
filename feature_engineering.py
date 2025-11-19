"""
Feature engineering module for peak detection
Extracts relevant features from hydrograph time series
"""
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import skew, kurtosis
from typing import Dict, List, Optional


class FeatureEngineer:
    """Extract features from streamflow time series for peak detection"""

    def __init__(self):
        # Rolling window sizes (in time steps)
        self.windows = [4, 12, 24, 48, 96]  # 1hr, 3hr, 6hr, 12hr, 24hr for 15-min data

    def add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features (hour, day, month, etc.)"""
        df = df.copy()
        df['hour'] = df['datetime_utc'].dt.hour
        df['day_of_week'] = df['datetime_utc'].dt.dayofweek
        df['day_of_year'] = df['datetime_utc'].dt.dayofyear
        df['month'] = df['datetime_utc'].dt.month
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_fall'] = df['month'].isin([9, 10, 11]).astype(int)
        return df

    def add_rolling_statistics(self, df: pd.DataFrame,
                               value_col: str = 'discharge_cms') -> pd.DataFrame:
        """Add rolling window statistics"""
        df = df.copy()

        for window in self.windows:
            # Rolling mean
            df[f'rolling_mean_{window}'] = (
                df[value_col].rolling(window=window, center=True, min_periods=1).mean()
            )

            # Rolling std
            df[f'rolling_std_{window}'] = (
                df[value_col].rolling(window=window, center=True, min_periods=1).std()
            )

            # Rolling max
            df[f'rolling_max_{window}'] = (
                df[value_col].rolling(window=window, center=True, min_periods=1).max()
            )

            # Rolling min
            df[f'rolling_min_{window}'] = (
                df[value_col].rolling(window=window, center=True, min_periods=1).min()
            )

            # Ratio to rolling mean
            df[f'ratio_to_mean_{window}'] = (
                df[value_col] / (df[f'rolling_mean_{window}'] + 1e-6)
            )

            # Distance from rolling max
            df[f'dist_from_max_{window}'] = (
                df[f'rolling_max_{window}'] - df[value_col]
            )

        return df

    def add_derivatives(self, df: pd.DataFrame,
                       value_col: str = 'discharge_cms') -> pd.DataFrame:
        """Add first and second derivatives (rate of change)"""
        df = df.copy()

        # First derivative (velocity)
        df['first_derivative'] = df[value_col].diff()

        # Second derivative (acceleration)
        df['second_derivative'] = df['first_derivative'].diff()

        # Smoothed derivatives
        for window in [4, 12]:
            df[f'smoothed_derivative_{window}'] = (
                df[value_col]
                .rolling(window=window, center=True, min_periods=1)
                .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) > 1 else 0)
            )

        # Sign changes (potential peaks/valleys)
        df['derivative_sign'] = np.sign(df['first_derivative'])
        df['sign_change'] = (df['derivative_sign'].diff() != 0).astype(int)

        return df

    def add_local_extrema_features(self, df: pd.DataFrame,
                                   value_col: str = 'discharge_cms') -> pd.DataFrame:
        """Identify local maxima and minima"""
        df = df.copy()

        # Find local maxima with different window sizes
        for window in [12, 24, 48]:
            # Local maxima (peak if value is max in window)
            df[f'is_local_max_{window}'] = (
                df[value_col] == df[value_col].rolling(
                    window=window, center=True, min_periods=1
                ).max()
            ).astype(int)

            # Distance to nearest local maximum
            local_max_indices = df[df[f'is_local_max_{window}'] == 1].index
            df[f'dist_to_local_max_{window}'] = 0

            for idx in df.index:
                if idx in local_max_indices:
                    df.at[idx, f'dist_to_local_max_{window}'] = 0
                else:
                    distances = np.abs(local_max_indices - idx)
                    if len(distances) > 0:
                        df.at[idx, f'dist_to_local_max_{window}'] = distances.min()

        return df

    def add_percentile_features(self, df: pd.DataFrame,
                                value_col: str = 'discharge_cms') -> pd.DataFrame:
        """Add features based on percentiles"""
        df = df.copy()

        # Global percentiles
        percentiles = [50, 75, 90, 95, 99]
        for p in percentiles:
            threshold = np.percentile(df[value_col].dropna(), p)
            df[f'above_p{p}'] = (df[value_col] > threshold).astype(int)
            df[f'ratio_to_p{p}'] = df[value_col] / (threshold + 1e-6)

        # Rolling percentiles
        for window in [96, 192]:  # 24hr, 48hr
            df[f'rolling_p90_{window}'] = (
                df[value_col].rolling(window=window, center=True, min_periods=1)
                .quantile(0.90)
            )
            df[f'above_rolling_p90_{window}'] = (
                df[value_col] > df[f'rolling_p90_{window}']
            ).astype(int)

        return df

    def add_statistical_features(self, df: pd.DataFrame,
                                 value_col: str = 'discharge_cms') -> pd.DataFrame:
        """Add statistical features over rolling windows"""
        df = df.copy()

        for window in [48, 96]:
            # Skewness
            df[f'rolling_skew_{window}'] = (
                df[value_col].rolling(window=window, center=True, min_periods=1)
                .apply(lambda x: skew(x) if len(x) > 2 else 0)
            )

            # Kurtosis
            df[f'rolling_kurtosis_{window}'] = (
                df[value_col].rolling(window=window, center=True, min_periods=1)
                .apply(lambda x: kurtosis(x) if len(x) > 2 else 0)
            )

            # Range
            df[f'rolling_range_{window}'] = (
                df[f'rolling_max_{window}'] - df[f'rolling_min_{window}']
            )

            # Coefficient of variation
            df[f'rolling_cv_{window}'] = (
                df[f'rolling_std_{window}'] / (df[f'rolling_mean_{window}'] + 1e-6)
            )

        return df

    def add_peak_prominence_features(self, df: pd.DataFrame,
                                     value_col: str = 'discharge_cms') -> pd.DataFrame:
        """Add features related to peak prominence"""
        df = df.copy()

        # Find peaks using scipy
        peaks, properties = signal.find_peaks(
            df[value_col].values,
            prominence=df[value_col].std() * 0.5,
            width=4
        )

        # Create features
        df['is_scipy_peak'] = 0
        df.loc[peaks, 'is_scipy_peak'] = 1

        if 'prominences' in properties:
            df['peak_prominence'] = 0.0
            df.loc[peaks, 'peak_prominence'] = properties['prominences']

        if 'widths' in properties:
            df['peak_width'] = 0.0
            df.loc[peaks, 'peak_width'] = properties['widths']

        return df

    def add_return_period_features(self, df: pd.DataFrame,
                                   return_periods: Optional[Dict[str, float]] = None,
                                   value_col: str = 'discharge_cms') -> pd.DataFrame:
        """Add features based on return period thresholds"""
        df = df.copy()

        if return_periods is not None:
            for rp_name, rp_value in return_periods.items():
                df[f'above_{rp_name}'] = (df[value_col] > rp_value).astype(int)
                df[f'ratio_to_{rp_name}'] = df[value_col] / (rp_value + 1e-6)
        else:
            # If no return periods, use statistical approximations
            # Create features for ALL return periods to match training data
            mean_q = df[value_col].mean()
            std_q = df[value_col].std()

            # Estimate thresholds for all return periods
            # Using rough statistical approximations
            estimated_rp = {
                'return_period_2': mean_q + 1.0 * std_q,
                'return_period_5': mean_q + 1.5 * std_q,
                'return_period_10': mean_q + 2.0 * std_q,
                'return_period_25': mean_q + 2.5 * std_q,
                'return_period_50': mean_q + 3.0 * std_q,
                'return_period_100': mean_q + 3.5 * std_q,
            }

            # Create features with same naming convention as when return_periods is provided
            for rp_name, rp_value in estimated_rp.items():
                df[f'above_{rp_name}'] = (df[value_col] > rp_value).astype(int)
                df[f'ratio_to_{rp_name}'] = df[value_col] / (rp_value + 1e-6)

        return df

    def engineer_features(self, df: pd.DataFrame,
                         return_periods: Optional[Dict[str, float]] = None,
                         value_col: str = 'discharge_cms') -> pd.DataFrame:
        """
        Apply all feature engineering steps

        Args:
            df: DataFrame with datetime and discharge columns
            return_periods: Optional dictionary of return period values
            value_col: Name of the discharge column

        Returns:
            DataFrame with engineered features
        """
        print(f"Engineering features for {len(df)} time steps...")

        # Apply all feature engineering steps
        df = self.add_temporal_features(df)
        df = self.add_rolling_statistics(df, value_col)
        df = self.add_derivatives(df, value_col)
        df = self.add_local_extrema_features(df, value_col)
        df = self.add_percentile_features(df, value_col)
        df = self.add_statistical_features(df, value_col)
        df = self.add_peak_prominence_features(df, value_col)
        df = self.add_return_period_features(df, return_periods, value_col)

        # Fill NaN values
        df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

        print(f"Created {len(df.columns)} total columns")

        return df

    def get_feature_columns(self, df: pd.DataFrame) -> List[str]:
        """Get list of feature columns (excluding metadata)"""
        exclude_cols = [
            'datetime_utc', 'site_no', 'qual_cd', 'discharge_cms',
            'site_no_normalized', 'is_peak'
        ]
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        return feature_cols


if __name__ == "__main__":
    # Test feature engineering
    from data_loader import DataLoader

    print("\n=== Testing Feature Engineering ===\n")

    loader = DataLoader()
    engineer = FeatureEngineer()

    # Load a test gage
    gage_data = loader.load_gage_data('03408500')
    return_periods = loader.get_return_periods_for_gage('03408500')

    if gage_data is not None:
        print(f"Original data shape: {gage_data.shape}")

        # Engineer features
        gage_with_features = engineer.engineer_features(
            gage_data,
            return_periods=return_periods
        )

        print(f"Data with features shape: {gage_with_features.shape}")

        # Get feature columns
        feature_cols = engineer.get_feature_columns(gage_with_features)
        print(f"Number of feature columns: {len(feature_cols)}")
        print(f"\nFirst 20 features:")
        for i, col in enumerate(feature_cols[:20], 1):
            print(f"  {i}. {col}")
