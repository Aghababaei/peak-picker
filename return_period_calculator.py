"""
Return period calculator for estimating flood return periods from streamflow data
"""
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Optional


class ReturnPeriodCalculator:
    """Calculate return periods from streamflow time series"""

    def __init__(self):
        self.return_periods = [2, 5, 10, 25, 50, 100]

    def extract_annual_maxima(self, df: pd.DataFrame,
                              value_col: str = 'discharge_cms') -> pd.Series:
        """
        Extract annual maximum series from streamflow data

        Args:
            df: DataFrame with datetime_utc and discharge columns
            value_col: Name of discharge column

        Returns:
            Series of annual maxima
        """
        df = df.copy()
        df['year'] = df['datetime_utc'].dt.year

        # Get annual maxima
        annual_max = df.groupby('year')[value_col].max()

        return annual_max

    def fit_gumbel_distribution(self, annual_maxima: pd.Series) -> tuple:
        """
        Fit Gumbel (Extreme Value Type I) distribution to annual maxima

        Returns:
            tuple of (location, scale) parameters
        """
        # Fit Gumbel distribution using maximum likelihood
        params = stats.gumbel_r.fit(annual_maxima)
        return params

    def fit_log_pearson_type3(self, annual_maxima: pd.Series) -> tuple:
        """
        Fit Log-Pearson Type III distribution (commonly used in hydrology)

        Returns:
            Distribution parameters
        """
        # Log-transform the data
        log_data = np.log10(annual_maxima)

        # Fit normal distribution to log-transformed data
        shape, loc, scale = stats.skewnorm.fit(log_data)

        return shape, loc, scale

    def calculate_return_periods_gumbel(self, annual_maxima: pd.Series) -> Dict[str, float]:
        """
        Calculate return period values using Gumbel distribution

        Args:
            annual_maxima: Series of annual maximum flows

        Returns:
            Dictionary with return period values
        """
        # Fit Gumbel distribution
        loc, scale = self.fit_gumbel_distribution(annual_maxima)

        # Calculate discharge for each return period
        # Gumbel: Q = μ + σ * (-ln(-ln(1 - 1/T)))
        # Or using scipy: Q = gumbel_r.ppf(1 - 1/T, loc, scale)

        results = {}
        for T in self.return_periods:
            # Probability of non-exceedance
            p = 1 - 1/T

            # Calculate discharge for this return period
            Q = stats.gumbel_r.ppf(p, loc, scale)

            results[f'return_period_{T}'] = Q

        return results

    def calculate_return_periods_weibull(self, annual_maxima: pd.Series) -> Dict[str, float]:
        """
        Calculate return period values using Weibull plotting position formula
        (Non-parametric approach)

        Args:
            annual_maxima: Series of annual maximum flows

        Returns:
            Dictionary with return period values
        """
        # Sort annual maxima in descending order
        sorted_data = np.sort(annual_maxima)[::-1]
        n = len(sorted_data)

        # Calculate Weibull plotting position
        # Return period T = (n + 1) / m, where m is rank
        ranks = np.arange(1, n + 1)
        return_period_empirical = (n + 1) / ranks

        results = {}
        for T in self.return_periods:
            # Interpolate to find discharge for desired return period
            if T <= return_period_empirical.max():
                Q = np.interp(T, return_period_empirical[::-1], sorted_data[::-1])
            else:
                # Extrapolate using the last two points
                # Use log-linear extrapolation
                log_T = np.log(return_period_empirical[-2:])
                log_Q = np.log(sorted_data[-2:])
                slope = (log_Q[1] - log_Q[0]) / (log_T[1] - log_T[0])
                intercept = log_Q[1] - slope * log_T[1]
                Q = np.exp(slope * np.log(T) + intercept)

            results[f'return_period_{T}'] = Q

        return results

    def calculate_return_periods_mixed(self, annual_maxima: pd.Series) -> Dict[str, float]:
        """
        Calculate return periods using a mixed approach:
        - Weibull for lower return periods (observed data)
        - Gumbel for higher return periods (extrapolation)

        Args:
            annual_maxima: Series of annual maximum flows

        Returns:
            Dictionary with return period values
        """
        weibull_results = self.calculate_return_periods_weibull(annual_maxima)
        gumbel_results = self.calculate_return_periods_gumbel(annual_maxima)

        n_years = len(annual_maxima)

        results = {}
        for T in self.return_periods:
            # Use Weibull for return periods within observed range
            # Use Gumbel for extrapolation beyond observed data
            if T <= n_years:
                results[f'return_period_{T}'] = weibull_results[f'return_period_{T}']
            else:
                results[f'return_period_{T}'] = gumbel_results[f'return_period_{T}']

        return results

    def calculate_return_periods(self, df: pd.DataFrame,
                                 value_col: str = 'discharge_cms',
                                 method: str = 'mixed') -> Optional[Dict[str, float]]:
        """
        Calculate return periods from streamflow time series

        Args:
            df: DataFrame with datetime_utc and discharge columns
            value_col: Name of discharge column
            method: 'gumbel', 'weibull', or 'mixed' (default)

        Returns:
            Dictionary with return period values
        """
        # Extract annual maxima
        annual_maxima = self.extract_annual_maxima(df, value_col)

        if len(annual_maxima) < 5:
            print(f"Warning: Only {len(annual_maxima)} years of data. "
                  "Return period estimates may be unreliable.")
            return None

        # Calculate return periods based on method
        if method == 'gumbel':
            results = self.calculate_return_periods_gumbel(annual_maxima)
        elif method == 'weibull':
            results = self.calculate_return_periods_weibull(annual_maxima)
        elif method == 'mixed':
            results = self.calculate_return_periods_mixed(annual_maxima)
        else:
            raise ValueError(f"Unknown method: {method}")

        return results

    def get_return_period_for_flow(self, flow: float, annual_maxima: pd.Series) -> float:
        """
        Estimate return period for a given flow value

        Args:
            flow: Flow value to estimate return period for
            annual_maxima: Series of annual maximum flows

        Returns:
            Estimated return period in years
        """
        # Fit Gumbel distribution
        loc, scale = self.fit_gumbel_distribution(annual_maxima)

        # Calculate probability of non-exceedance
        p = stats.gumbel_r.cdf(flow, loc, scale)

        # Calculate return period: T = 1 / (1 - p)
        if p < 0.9999:  # Avoid division by very small numbers
            T = 1 / (1 - p)
        else:
            T = 10000  # Very high return period

        return T


if __name__ == "__main__":
    # Test return period calculator
    from data_loader import DataLoader

    print("\n=== Testing Return Period Calculator ===\n")

    loader = DataLoader()
    calculator = ReturnPeriodCalculator()

    # Load a test gage
    gage_data = loader.load_gage_data('03408500')

    if gage_data is not None:
        print(f"Calculating return periods for gage 03408500...")
        print(f"Data period: {gage_data['datetime_utc'].min()} to {gage_data['datetime_utc'].max()}")

        # Calculate return periods
        calculated_rp = calculator.calculate_return_periods(gage_data, method='mixed')

        if calculated_rp:
            print("\nCalculated return periods:")
            for key, value in calculated_rp.items():
                print(f"  {key}: {value:.2f} cms")

            # Compare with actual return periods (if available)
            actual_rp = loader.get_return_periods_for_gage('03408500')
            if actual_rp:
                print("\nActual return periods (from file):")
                for key, value in actual_rp.items():
                    print(f"  {key}: {value:.2f} cms")

                print("\nComparison (Calculated vs Actual):")
                for key in calculated_rp.keys():
                    calc_val = calculated_rp[key]
                    actual_val = actual_rp[key]
                    diff_pct = ((calc_val - actual_val) / actual_val) * 100
                    print(f"  {key}: {calc_val:.2f} vs {actual_val:.2f} ({diff_pct:+.1f}%)")

        # Test annual maxima extraction
        print("\nAnnual maxima:")
        annual_max = calculator.extract_annual_maxima(gage_data)
        print(annual_max)
