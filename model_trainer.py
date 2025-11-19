"""
ML model training for peak detection
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    classification_report, confusion_matrix, precision_recall_curve,
    f1_score, precision_score, recall_score, roc_auc_score
)
import joblib
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from data_loader import DataLoader
from feature_engineering import FeatureEngineer


class PeakDetectionModel:
    """Train and manage ML models for peak detection"""

    def __init__(self, model_type: str = 'random_forest'):
        """
        Initialize the model

        Args:
            model_type: Type of model ('random_forest', 'xgboost', 'lightgbm', 'gradient_boosting')
        """
        self.model_type = model_type
        self.model = None
        self.feature_columns = None
        self.threshold = 0.5  # Classification threshold

    def create_model(self):
        """Create the ML model based on model_type"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,  # Reduced from 200 for faster training (2-3x speedup)
                max_depth=12,      # Slightly reduced depth
                min_samples_split=20,  # Increased to prevent overfitting on huge dataset
                min_samples_leaf=10,   # Increased for stability
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                verbose=2  # Show progress during training
            )
        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                self.model = xgb.XGBClassifier(
                    n_estimators=200,
                    max_depth=10,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    scale_pos_weight=10,  # Handle class imbalance
                    random_state=42,
                    n_jobs=-1,
                    eval_metric='logloss'
                )
            except ImportError:
                raise ImportError(
                    "XGBoost is not available. Please install libomp:\n"
                    "  brew install libomp\n"
                    "Or use Random Forest instead:\n"
                    "  python peakpicker.py --train --model-type random_forest"
                )
        elif self.model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                self.model = lgb.LGBMClassifier(
                    n_estimators=200,
                    max_depth=10,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    class_weight='balanced',
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1
                )
            except ImportError:
                raise ImportError(
                    "LightGBM is not available. Please install libomp:\n"
                    "  brew install libomp\n"
                    "Or use Random Forest instead:\n"
                    "  python peakpicker.py --train --model-type random_forest"
                )
        elif self.model_type == 'gradient_boosting':
            self.model = GradientBoostingClassifier(
                n_estimators=200,
                max_depth=10,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def create_labels(self, df: pd.DataFrame, peaks: pd.DataFrame,
                     tolerance_minutes: int = 60) -> pd.DataFrame:
        """
        Create binary labels for peak detection

        Args:
            df: DataFrame with engineered features
            peaks: DataFrame with manual peak times
            tolerance_minutes: Time window around peak to mark as positive

        Returns:
            DataFrame with 'is_peak' column added
        """
        df = df.copy()
        df['is_peak'] = 0

        for _, peak_row in peaks.iterrows():
            peak_time = peak_row['peak_time_utc']

            # Find time steps within tolerance window
            time_diff = np.abs((df['datetime_utc'] - peak_time).dt.total_seconds() / 60)
            within_window = time_diff <= tolerance_minutes

            df.loc[within_window, 'is_peak'] = 1

        return df

    def prepare_training_data(self, loader: DataLoader, engineer: FeatureEngineer,
                             tolerance_minutes: int = 60) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare training data from all gages with return periods

        Args:
            loader: DataLoader instance
            engineer: FeatureEngineer instance
            tolerance_minutes: Time window for peak labeling

        Returns:
            Tuple of (X_train features, y_train labels)
        """
        print("Preparing training data...")

        gages_with_rp = loader.get_gages_with_return_periods()
        print(f"Found {len(gages_with_rp)} gages with return periods")

        all_data = []

        for site_no in tqdm(gages_with_rp, desc="Processing gages"):
            # Load gage data
            gage_data = loader.load_gage_data(site_no)
            if gage_data is None:
                continue

            # Get return periods and peaks
            return_periods = loader.get_return_periods_for_gage(site_no)
            peaks = loader.get_manual_peaks_for_gage(site_no)

            if len(peaks) == 0:
                continue

            # Engineer features
            gage_features = engineer.engineer_features(
                gage_data,
                return_periods=return_periods
            )

            # Create labels
            gage_labeled = self.create_labels(
                gage_features,
                peaks,
                tolerance_minutes=tolerance_minutes
            )

            all_data.append(gage_labeled)

        # Combine all data
        combined_data = pd.concat(all_data, ignore_index=True)

        print(f"Total samples: {len(combined_data)}")
        print(f"Positive samples (peaks): {combined_data['is_peak'].sum()}")
        print(f"Negative samples: {(combined_data['is_peak'] == 0).sum()}")
        print(f"Class balance: {combined_data['is_peak'].mean():.4f}")

        return combined_data

    def train(self, X_train: pd.DataFrame, y_train: pd.Series):
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training labels
        """
        print(f"\nTraining {self.model_type} model...")

        # Create model if not already created
        if self.model is None:
            self.create_model()

        # Store feature columns
        self.feature_columns = X_train.columns.tolist()

        # Train the model
        self.model.fit(X_train, y_train)

        print("Training complete!")

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """
        Evaluate the model

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary with evaluation metrics
        """
        print("\nEvaluating model...")

        # Predict
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        # Calculate metrics
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        try:
            auc = roc_auc_score(y_test, y_pred_proba)
        except:
            auc = 0.0

        metrics = {
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc
        }

        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))

        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))

        return metrics

    def get_feature_importance(self, top_n: int = 20) -> pd.DataFrame:
        """
        Get feature importance from the trained model

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature names and importance scores
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
        else:
            raise ValueError("Model does not support feature importance")

        # Create DataFrame
        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': importances
        }).sort_values('importance', ascending=False)

        return importance_df.head(top_n)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict on new data

        Args:
            X: Features

        Returns:
            Tuple of (predicted labels, predicted probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")

        # Ensure columns match training
        X = X[self.feature_columns]

        y_pred_proba = self.model.predict_proba(X)[:, 1]
        y_pred = (y_pred_proba >= self.threshold).astype(int)

        return y_pred, y_pred_proba

    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_columns': self.feature_columns,
            'threshold': self.threshold
        }
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str) -> 'PeakDetectionModel':
        """Load a trained model"""
        model_data = joblib.load(filepath)

        instance = cls(model_type=model_data['model_type'])
        instance.model = model_data['model']
        instance.feature_columns = model_data['feature_columns']
        instance.threshold = model_data['threshold']

        print(f"Model loaded from {filepath}")
        return instance


def train_peak_detection_model(model_type: str = 'random_forest',
                               test_size: float = 0.2,
                               save_path: str = 'peak_model.pkl') -> PeakDetectionModel:
    """
    Train a peak detection model on all available data

    Args:
        model_type: Type of model to train
        test_size: Fraction of data to use for testing
        save_path: Path to save the trained model

    Returns:
        Trained PeakDetectionModel instance
    """
    # Initialize components
    loader = DataLoader()
    engineer = FeatureEngineer()
    model = PeakDetectionModel(model_type=model_type)

    # Prepare training data
    labeled_data = model.prepare_training_data(loader, engineer, tolerance_minutes=60)

    # Get features and labels
    feature_cols = engineer.get_feature_columns(labeled_data)
    X = labeled_data[feature_cols]
    y = labeled_data['is_peak']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    print(f"\nTrain set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train model
    model.train(X_train, y_train)

    # Evaluate model
    metrics = model.evaluate(X_test, y_test)

    # Show feature importance
    print("\nTop 20 Most Important Features:")
    importance_df = model.get_feature_importance(top_n=20)
    for idx, row in importance_df.iterrows():
        print(f"  {row['feature']}: {row['importance']:.4f}")

    # Save model
    model.save_model(save_path)

    return model


if __name__ == "__main__":
    print("\n=== Training Peak Detection Model ===\n")

    # Train the model
    trained_model = train_peak_detection_model(
        model_type='random_forest',
        test_size=0.2,
        save_path='peak_model.pkl'
    )

    print("\n=== Training Complete ===")
