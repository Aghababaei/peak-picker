"""
Visualization module for hydrographs and peak detection
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
from pathlib import Path
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')


class HydrographPlotter:
    """Create visualizations for hydrographs and detected peaks"""

    def __init__(self, output_dir: str = 'plots'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

        # Set plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = {
            'discharge': '#1f77b4',
            'detected_peaks': '#d62728',
            'manual_peaks': '#2ca02c',
            'return_period': '#ff7f0e',
            'probability': '#9467bd'
        }

    def plot_hydrograph_with_peaks(
        self,
        df: pd.DataFrame,
        detected_peaks: Optional[pd.DataFrame] = None,
        manual_peaks: Optional[pd.DataFrame] = None,
        return_periods: Optional[Dict[str, float]] = None,
        site_no: str = 'Unknown',
        save_path: Optional[str] = None,
        show_probability: bool = False
    ):
        """
        Plot hydrograph with detected and manual peaks

        Args:
            df: DataFrame with datetime_utc, discharge_cms, and optionally peak_probability
            detected_peaks: DataFrame with detected peak information
            manual_peaks: DataFrame with manual peak information
            return_periods: Dictionary of return period values
            site_no: Gage site number
            save_path: Path to save the plot (if None, auto-generates)
            show_probability: Whether to show peak probability as secondary y-axis
        """
        fig, ax1 = plt.subplots(figsize=(18, 9))

        # Set white background
        ax1.set_facecolor('white')
        fig.patch.set_facecolor('white')

        # Get y-axis limits for proper zone plotting
        max_discharge = df['discharge_cms'].max()

        # Plot colored return period zones (from bottom to top)
        if return_periods:
            # Sort return periods
            rp_sorted = sorted(return_periods.items(), key=lambda x: x[1])

            # Define colors for zones (from your image)
            zone_colors = {
                0: '#FFFFFF',      # White - below RP2
                1: '#FFFACD',      # Light yellow - RP2 to RP5
                2: '#FFE4B5',      # Moccasin - RP5 to RP10
                3: '#FFCCCB',      # Light pink - RP10 to RP25
                4: '#FFB6C1',      # Pink - RP25 to RP50
                5: '#DDA0DD',      # Plum - RP50 to RP100
                6: '#D8BFD8',      # Thistle - above RP100
            }

            # Plot zones
            y_min = 0
            for i, (rp_name, rp_value) in enumerate(rp_sorted):
                color = zone_colors.get(i, '#E6E6FA')
                ax1.axhspan(y_min, rp_value, facecolor=color, alpha=0.6, zorder=0)
                y_min = rp_value

            # Add zone above highest return period
            ax1.axhspan(y_min, max_discharge * 1.2, facecolor=zone_colors[6], alpha=0.6, zorder=0)

            # Plot return period threshold lines (dashed red)
            for rp_name, rp_value in return_periods.items():
                rp_num = rp_name.split('_')[-1]
                ax1.axhline(
                    y=rp_value,
                    color='#DC143C',  # Crimson red
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.8,
                    zorder=2
                )

        # Plot discharge line (bold blue)
        ax1.plot(
            df['datetime_utc'],
            df['discharge_cms'],
            color='#1E90FF',  # Dodger blue
            linewidth=2.5,
            label='Discharge',
            zorder=3
        )

        # Plot detected peaks with labels
        if detected_peaks is not None and len(detected_peaks) > 0:
            ax1.scatter(
                detected_peaks['datetime_utc'],
                detected_peaks['discharge_cms'],
                color='#DC143C',  # Crimson red
                s=100,
                marker='o',
                edgecolors='white',
                linewidths=1.5,
                label=f'Detected Peaks (n={len(detected_peaks)})',
                zorder=5
            )

            # Add numbered labels to peaks
            for idx, (_, row) in enumerate(detected_peaks.iterrows(), 1):
                ax1.annotate(
                    str(idx),
                    xy=(row['datetime_utc'], row['discharge_cms']),
                    xytext=(0, 8),
                    textcoords='offset points',
                    ha='center',
                    fontsize=9,
                    fontweight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='black', linewidth=1),
                    zorder=6
                )

        # Plot manual peaks (if available) - as triangles
        if manual_peaks is not None and len(manual_peaks) > 0:
            ax1.scatter(
                manual_peaks['peak_time_utc'],
                manual_peaks['peak_flow_cms'],
                color='#32CD32',  # Lime green
                s=120,
                marker='^',
                edgecolors='white',
                linewidths=1.5,
                label=f'Manual Peaks (n={len(manual_peaks)})',
                zorder=5
            )

        # Styling
        ax1.set_xlabel('Date', fontsize=13, fontweight='bold')
        ax1.set_ylabel('Discharge (cms)', fontsize=13, fontweight='bold')
        ax1.tick_params(axis='both', labelsize=11)

        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.xticks(rotation=45, ha='right')

        # Set y-axis limits
        ax1.set_ylim([0, max_discharge * 1.15])

        # Title
        plt.title(
            f'Site {site_no}',
            fontsize=15,
            fontweight='bold',
            pad=15
        )

        # Legend
        ax1.legend(loc='upper right', fontsize=11, framealpha=0.95, edgecolor='black')

        # Add subtle grid
        ax1.grid(True, alpha=0.2, linestyle='-', linewidth=0.5, color='gray', zorder=1)

        # Tight layout
        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.output_dir / f'{site_no}_hydrograph.png'
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"Plot saved to {save_path}")

        plt.close()

    def plot_peak_comparison(
        self,
        detected_peaks: pd.DataFrame,
        manual_peaks: pd.DataFrame,
        site_no: str = 'Unknown',
        save_path: Optional[str] = None
    ):
        """
        Create comparison plot between detected and manual peaks

        Args:
            detected_peaks: DataFrame with detected peaks
            manual_peaks: DataFrame with manual peaks
            site_no: Gage site number
            save_path: Path to save the plot
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Peak Detection Analysis - Gage {site_no}', fontsize=14, fontweight='bold')

        # Plot 1: Time series comparison
        ax = axes[0, 0]
        if len(detected_peaks) > 0:
            ax.scatter(
                range(len(detected_peaks)),
                detected_peaks['discharge_cms'],
                color=self.colors['detected_peaks'],
                s=100,
                alpha=0.6,
                label='Detected Peaks'
            )
        if len(manual_peaks) > 0:
            ax.scatter(
                range(len(manual_peaks)),
                manual_peaks['peak_flow_cms'],
                color=self.colors['manual_peaks'],
                s=100,
                alpha=0.6,
                label='Manual Peaks'
            )
        ax.set_xlabel('Peak Index')
        ax.set_ylabel('Discharge (cms)')
        ax.set_title('Peak Magnitudes')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Distribution of peak magnitudes
        ax = axes[0, 1]
        if len(detected_peaks) > 0:
            ax.hist(
                detected_peaks['discharge_cms'],
                bins=20,
                color=self.colors['detected_peaks'],
                alpha=0.5,
                label='Detected Peaks'
            )
        if len(manual_peaks) > 0:
            ax.hist(
                manual_peaks['peak_flow_cms'],
                bins=20,
                color=self.colors['manual_peaks'],
                alpha=0.5,
                label='Manual Peaks'
            )
        ax.set_xlabel('Discharge (cms)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Peak Magnitudes')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Monthly distribution
        ax = axes[1, 0]
        if len(detected_peaks) > 0:
            detected_months = detected_peaks['datetime_utc'].dt.month.value_counts().sort_index()
            ax.bar(
                detected_months.index,
                detected_months.values,
                color=self.colors['detected_peaks'],
                alpha=0.5,
                label='Detected Peaks',
                width=0.4
            )
        if len(manual_peaks) > 0:
            manual_months = manual_peaks['peak_time_utc'].dt.month.value_counts().sort_index()
            ax.bar(
                [m + 0.4 for m in manual_months.index],
                manual_months.values,
                color=self.colors['manual_peaks'],
                alpha=0.5,
                label='Manual Peaks',
                width=0.4
            )
        ax.set_xlabel('Month')
        ax.set_ylabel('Number of Peaks')
        ax.set_title('Monthly Distribution of Peaks')
        ax.set_xticks(range(1, 13))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 4: Statistics summary
        ax = axes[1, 1]
        ax.axis('off')

        stats_text = "Peak Detection Statistics\n" + "="*30 + "\n\n"

        stats_text += f"Detected Peaks: {len(detected_peaks)}\n"
        stats_text += f"Manual Peaks: {len(manual_peaks)}\n\n"

        if len(detected_peaks) > 0:
            stats_text += "Detected Peaks:\n"
            stats_text += f"  Mean: {detected_peaks['discharge_cms'].mean():.2f} cms\n"
            stats_text += f"  Median: {detected_peaks['discharge_cms'].median():.2f} cms\n"
            stats_text += f"  Max: {detected_peaks['discharge_cms'].max():.2f} cms\n"
            stats_text += f"  Min: {detected_peaks['discharge_cms'].min():.2f} cms\n\n"

        if len(manual_peaks) > 0:
            stats_text += "Manual Peaks:\n"
            stats_text += f"  Mean: {manual_peaks['peak_flow_cms'].mean():.2f} cms\n"
            stats_text += f"  Median: {manual_peaks['peak_flow_cms'].median():.2f} cms\n"
            stats_text += f"  Max: {manual_peaks['peak_flow_cms'].max():.2f} cms\n"
            stats_text += f"  Min: {manual_peaks['peak_flow_cms'].min():.2f} cms\n"

        ax.text(0.1, 0.9, stats_text, transform=ax.transAxes,
               fontsize=11, verticalalignment='top', fontfamily='monospace')

        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.output_dir / f'{site_no}_peak_comparison.png'
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison plot saved to {save_path}")

        plt.close()

    def plot_annual_maxima(
        self,
        df: pd.DataFrame,
        return_periods: Optional[Dict[str, float]] = None,
        site_no: str = 'Unknown',
        save_path: Optional[str] = None
    ):
        """
        Plot annual maximum series

        Args:
            df: DataFrame with datetime_utc and discharge_cms
            return_periods: Dictionary of return period values
            site_no: Gage site number
            save_path: Path to save the plot
        """
        # Extract annual maxima
        df['year'] = df['datetime_utc'].dt.year
        annual_max = df.groupby('year')['discharge_cms'].max()

        fig, ax = plt.subplots(figsize=(12, 6))

        # Plot annual maxima
        ax.bar(annual_max.index, annual_max.values, color=self.colors['discharge'], alpha=0.7)

        # Plot return period thresholds
        if return_periods:
            for rp_name, rp_value in return_periods.items():
                rp_num = rp_name.split('_')[-1]
                ax.axhline(
                    y=rp_value,
                    color=self.colors['return_period'],
                    linestyle='--',
                    linewidth=2,
                    label=f'{rp_num}-year RP'
                )

        ax.set_xlabel('Year', fontsize=12, fontweight='bold')
        ax.set_ylabel('Annual Maximum Discharge (cms)', fontsize=12, fontweight='bold')
        ax.set_title(f'Annual Maximum Series - Gage {site_no}', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save plot
        if save_path is None:
            save_path = self.output_dir / f'{site_no}_annual_maxima.png'
        else:
            save_path = Path(save_path)

        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Annual maxima plot saved to {save_path}")

        plt.close()


if __name__ == "__main__":
    # Test the plotter
    from data_loader import DataLoader

    print("\n=== Testing Hydrograph Plotter ===\n")

    loader = DataLoader()
    plotter = HydrographPlotter(output_dir='plots')

    # Load test data
    site_no = '03408500'
    gage_data = loader.load_gage_data(site_no)
    manual_peaks = loader.get_manual_peaks_for_gage(site_no)
    return_periods = loader.get_return_periods_for_gage(site_no)

    if gage_data is not None:
        # Plot hydrograph with manual peaks
        plotter.plot_hydrograph_with_peaks(
            df=gage_data,
            manual_peaks=manual_peaks,
            return_periods=return_periods,
            site_no=site_no
        )

        # Plot annual maxima
        plotter.plot_annual_maxima(
            df=gage_data,
            return_periods=return_periods,
            site_no=site_no
        )

        print("\nPlots created successfully!")
