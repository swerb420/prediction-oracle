"""
Dataset builder for ML training.
Handles feature alignment, train/val/test splits, and time-series aware sampling.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from core.logging_utils import get_logger

logger = get_logger(__name__)


class TrainingDataset(BaseModel):
    """Training dataset with metadata."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    # Arrays
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray | None = None
    y_test: np.ndarray | None = None
    
    # Metadata
    feature_names: list[str]
    target_name: str
    
    # Time info
    train_start: datetime | None = None
    train_end: datetime | None = None
    val_start: datetime | None = None
    val_end: datetime | None = None
    test_start: datetime | None = None
    test_end: datetime | None = None
    
    # Statistics
    n_samples_train: int = 0
    n_samples_val: int = 0
    n_samples_test: int = 0
    n_features: int = 0
    class_balance_train: dict[int, float] = Field(default_factory=dict)
    
    @property
    def has_test(self) -> bool:
        return self.X_test is not None and len(self.X_test) > 0


class DatasetBuilder:
    """
    Builds datasets for ML training with time-series aware splitting.
    
    Features:
    - Proper temporal train/val/test splits (no data leakage)
    - Gap period between train and val to avoid look-ahead bias
    - Class balancing options
    - Feature normalization
    """
    
    def __init__(
        self,
        target_column: str = "target",
        timestamp_column: str = "timestamp",
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        gap_periods: int = 4,  # Gap between train and val (4 periods = 1 hour for 15m)
        normalize: bool = True
    ):
        self.target_column = target_column
        self.timestamp_column = timestamp_column
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.gap_periods = gap_periods
        self.normalize = normalize
        
        self._feature_means: dict[str, float] = {}
        self._feature_stds: dict[str, float] = {}
    
    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
        exclude_columns: list[str] | None = None
    ) -> TrainingDataset:
        """
        Build dataset from a pandas DataFrame.
        
        Args:
            df: DataFrame with features, target, and timestamp
            feature_columns: Specific columns to use as features
            exclude_columns: Columns to exclude from features
            
        Returns:
            TrainingDataset
        """
        df = df.copy()
        
        # Validate columns
        if self.target_column not in df.columns:
            raise ValueError(f"Target column '{self.target_column}' not found")
        
        if self.timestamp_column not in df.columns:
            raise ValueError(f"Timestamp column '{self.timestamp_column}' not found")
        
        # Sort by timestamp
        df = df.sort_values(self.timestamp_column).reset_index(drop=True)
        
        # Determine feature columns
        if feature_columns is None:
            exclude = {self.target_column, self.timestamp_column}
            if exclude_columns:
                exclude.update(exclude_columns)
            feature_columns = [c for c in df.columns if c not in exclude]
        
        # Remove any remaining non-numeric columns
        numeric_cols = df[feature_columns].select_dtypes(include=[np.number]).columns.tolist()
        feature_columns = numeric_cols
        
        logger.info(f"Using {len(feature_columns)} features")
        
        # Handle missing values
        df[feature_columns] = df[feature_columns].fillna(0)
        
        # Calculate split indices
        n = len(df)
        train_end = int(n * self.train_ratio)
        val_start = train_end + self.gap_periods
        val_end = val_start + int(n * self.val_ratio)
        test_start = val_end + self.gap_periods
        
        # Split data
        df_train = df.iloc[:train_end]
        df_val = df.iloc[val_start:val_end]
        df_test = df.iloc[test_start:] if test_start < n else pd.DataFrame()
        
        # Extract features and targets
        X_train = df_train[feature_columns].values.astype(np.float32)
        y_train = df_train[self.target_column].values.astype(np.int32)
        
        X_val = df_val[feature_columns].values.astype(np.float32)
        y_val = df_val[self.target_column].values.astype(np.int32)
        
        X_test = df_test[feature_columns].values.astype(np.float32) if len(df_test) > 0 else None
        y_test = df_test[self.target_column].values.astype(np.int32) if len(df_test) > 0 else None
        
        # Normalize features
        if self.normalize:
            X_train, X_val, X_test = self._normalize_features(
                X_train, X_val, X_test, feature_columns
            )
        
        # Calculate class balance
        unique, counts = np.unique(y_train, return_counts=True)
        class_balance = {int(u): float(c / len(y_train)) for u, c in zip(unique, counts)}
        
        # Extract timestamps
        train_start = df_train[self.timestamp_column].iloc[0] if len(df_train) > 0 else None
        train_end_ts = df_train[self.timestamp_column].iloc[-1] if len(df_train) > 0 else None
        val_start_ts = df_val[self.timestamp_column].iloc[0] if len(df_val) > 0 else None
        val_end_ts = df_val[self.timestamp_column].iloc[-1] if len(df_val) > 0 else None
        test_start_ts = df_test[self.timestamp_column].iloc[0] if len(df_test) > 0 else None
        test_end_ts = df_test[self.timestamp_column].iloc[-1] if len(df_test) > 0 else None
        
        return TrainingDataset(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_columns,
            target_name=self.target_column,
            train_start=train_start,
            train_end=train_end_ts,
            val_start=val_start_ts,
            val_end=val_end_ts,
            test_start=test_start_ts,
            test_end=test_end_ts,
            n_samples_train=len(X_train),
            n_samples_val=len(X_val),
            n_samples_test=len(X_test) if X_test is not None else 0,
            n_features=len(feature_columns),
            class_balance_train=class_balance,
        )
    
    def _normalize_features(
        self,
        X_train: np.ndarray,
        X_val: np.ndarray,
        X_test: np.ndarray | None,
        feature_names: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Z-score normalization using training set statistics.
        """
        # Compute stats from training set only
        means = np.mean(X_train, axis=0)
        stds = np.std(X_train, axis=0)
        stds = np.where(stds == 0, 1.0, stds)  # Avoid division by zero
        
        # Store for later use
        for i, name in enumerate(feature_names):
            self._feature_means[name] = float(means[i])
            self._feature_stds[name] = float(stds[i])
        
        # Normalize
        X_train_norm = (X_train - means) / stds
        X_val_norm = (X_val - means) / stds
        X_test_norm = (X_test - means) / stds if X_test is not None else None
        
        return X_train_norm, X_val_norm, X_test_norm
    
    def normalize_new_data(
        self,
        X: np.ndarray,
        feature_names: list[str]
    ) -> np.ndarray:
        """
        Normalize new data using stored statistics.
        """
        means = np.array([self._feature_means.get(n, 0.0) for n in feature_names])
        stds = np.array([self._feature_stds.get(n, 1.0) for n in feature_names])
        
        return (X - means) / stds
    
    def get_normalization_params(self) -> dict[str, dict[str, float]]:
        """Get normalization parameters for persistence."""
        return {
            "means": self._feature_means.copy(),
            "stds": self._feature_stds.copy()
        }
    
    def set_normalization_params(
        self,
        params: dict[str, dict[str, float]]
    ) -> None:
        """Restore normalization parameters."""
        self._feature_means = params.get("means", {})
        self._feature_stds = params.get("stds", {})


class SequenceDatasetBuilder(DatasetBuilder):
    """
    Builds sequence datasets for LSTM/temporal models.
    Creates sliding windows of historical data.
    """
    
    def __init__(
        self,
        sequence_length: int = 12,  # 12 periods = 3 hours for 15m
        **kwargs
    ):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
    
    def build_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: list[str]
    ) -> TrainingDataset:
        """
        Build sequence dataset with sliding windows.
        
        Returns:
            TrainingDataset with shape (n_samples, sequence_length, n_features)
        """
        df = df.copy().sort_values(self.timestamp_column).reset_index(drop=True)
        
        # Extract arrays
        X = df[feature_columns].values.astype(np.float32)
        y = df[self.target_column].values.astype(np.int32)
        timestamps = df[self.timestamp_column].values
        
        # Create sequences
        X_seq = []
        y_seq = []
        ts_seq = []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i - self.sequence_length:i])
            y_seq.append(y[i])
            ts_seq.append(timestamps[i])
        
        X_seq = np.array(X_seq)
        y_seq = np.array(y_seq)
        
        # Split temporally
        n = len(X_seq)
        train_end = int(n * self.train_ratio)
        val_start = train_end + self.gap_periods
        val_end = val_start + int(n * self.val_ratio)
        test_start = val_end + self.gap_periods
        
        return TrainingDataset(
            X_train=X_seq[:train_end],
            y_train=y_seq[:train_end],
            X_val=X_seq[val_start:val_end],
            y_val=y_seq[val_start:val_end],
            X_test=X_seq[test_start:] if test_start < n else None,
            y_test=y_seq[test_start:] if test_start < n else None,
            feature_names=feature_columns,
            target_name=self.target_column,
            n_samples_train=train_end,
            n_samples_val=val_end - val_start,
            n_samples_test=n - test_start if test_start < n else 0,
            n_features=len(feature_columns),
        )


def create_training_dataset(
    df: pd.DataFrame,
    target_column: str = "target",
    timestamp_column: str = "timestamp",
    feature_columns: list[str] | None = None,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    normalize: bool = True
) -> TrainingDataset:
    """Convenience function to create a training dataset."""
    builder = DatasetBuilder(
        target_column=target_column,
        timestamp_column=timestamp_column,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=1.0 - train_ratio - val_ratio,
        normalize=normalize
    )
    return builder.build_from_dataframe(df, feature_columns)
