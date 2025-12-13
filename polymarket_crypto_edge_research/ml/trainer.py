"""
Training orchestration for ML models.
Handles hyperparameter tuning, cross-validation, and training pipelines.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
from pydantic import BaseModel, Field

from core.logging_utils import get_logger
from ml.datasets import TrainingDataset
from ml.models_tabular import TabularModel, LightGBMClassifier, GradientBoostingClassifier
from ml.calibration import IsotonicCalibrator, CalibrationMetrics

logger = get_logger(__name__)


class TrainingConfig(BaseModel):
    """Configuration for model training."""
    
    # Model selection
    model_type: str = "lightgbm"  # lightgbm, gradient_boosting, lstm
    
    # Training params
    n_folds: int = 5  # For time-series CV
    early_stopping_rounds: int = 20
    random_state: int = 42
    
    # LightGBM params
    lgb_n_estimators: int = 200
    lgb_max_depth: int = 6
    lgb_learning_rate: float = 0.05
    lgb_min_child_samples: int = 20
    lgb_subsample: float = 0.8
    lgb_colsample_bytree: float = 0.8
    
    # Gradient Boosting params
    gb_n_estimators: int = 100
    gb_max_depth: int = 5
    gb_learning_rate: float = 0.1
    
    # Calibration
    calibrate: bool = True
    calibration_method: str = "isotonic"  # isotonic, platt
    
    # Thresholds
    confidence_threshold: float = 0.6
    min_accuracy_threshold: float = 0.52  # Minimum to be profitable


class TrainingResult(BaseModel):
    """Results from model training."""
    
    model_config = {"arbitrary_types_allowed": True}
    
    # Metrics
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float | None = None
    
    train_auc: float | None = None
    val_auc: float | None = None
    test_auc: float | None = None
    
    # Calibration
    calibration_metrics: CalibrationMetrics | None = None
    
    # Feature importance
    feature_importance: dict[str, float] = Field(default_factory=dict)
    top_features: list[str] = Field(default_factory=list)
    
    # CV results
    cv_scores: list[float] = Field(default_factory=list)
    cv_mean: float | None = None
    cv_std: float | None = None
    
    # Training metadata
    training_timestamp: datetime | None = None
    training_duration_seconds: float = 0.0
    n_samples_train: int = 0
    n_features: int = 0
    
    @property
    def is_profitable(self) -> bool:
        """Check if model accuracy suggests profitability."""
        return self.val_accuracy > 0.52


class Trainer:
    """
    Model training orchestrator.
    
    Handles:
    - Model instantiation
    - Time-series cross-validation
    - Hyperparameter tuning (optional)
    - Calibration
    - Result aggregation
    """
    
    def __init__(self, config: TrainingConfig | None = None):
        self.config = config or TrainingConfig()
        self._model: TabularModel | None = None
        self._calibrator: IsotonicCalibrator | None = None
    
    @property
    def model(self) -> TabularModel | None:
        return self._model
    
    @property
    def calibrator(self) -> IsotonicCalibrator | None:
        return self._calibrator
    
    def train(
        self,
        dataset: TrainingDataset,
        tune_hyperparams: bool = False
    ) -> TrainingResult:
        """
        Train model on dataset.
        
        Args:
            dataset: Training dataset
            tune_hyperparams: Whether to run hyperparameter tuning
            
        Returns:
            TrainingResult
        """
        import time
        from core.time_utils import now_utc
        
        start_time = time.time()
        training_timestamp = now_utc()
        
        logger.info(f"Training {self.config.model_type} model...")
        logger.info(f"Dataset: {dataset.n_samples_train} train, {dataset.n_samples_val} val")
        
        # Create model
        self._model = self._create_model()
        
        # Optionally tune hyperparameters
        if tune_hyperparams:
            best_params = self._tune_hyperparameters(dataset)
            logger.info(f"Best params: {best_params}")
            self._model = self._create_model(**best_params)
        
        # Train model
        metrics = self._model.fit(
            X_train=dataset.X_train,
            y_train=dataset.y_train,
            X_val=dataset.X_val,
            y_val=dataset.y_val,
            feature_names=dataset.feature_names
        )
        
        # Calculate AUC
        from sklearn.metrics import roc_auc_score
        
        train_proba = self._model.predict_proba(dataset.X_train)[:, 1]
        val_proba = self._model.predict_proba(dataset.X_val)[:, 1]
        
        try:
            train_auc = float(roc_auc_score(dataset.y_train, train_proba))
            val_auc = float(roc_auc_score(dataset.y_val, val_proba))
        except ValueError:
            train_auc = val_auc = 0.5
        
        # Test set evaluation
        test_accuracy = None
        test_auc = None
        if dataset.has_test:
            test_pred = self._model.predict(dataset.X_test)
            test_accuracy = float(np.mean(test_pred == dataset.y_test))
            
            test_proba = self._model.predict_proba(dataset.X_test)[:, 1]
            try:
                test_auc = float(roc_auc_score(dataset.y_test, test_proba))
            except ValueError:
                test_auc = 0.5
        
        # Calibration
        calibration_metrics = None
        if self.config.calibrate:
            self._calibrator = IsotonicCalibrator()
            calibration_metrics = self._calibrator.fit(val_proba, dataset.y_val)
        
        # Feature importance
        feature_importance = self._model.get_feature_importance()
        top_features = sorted(
            feature_importance.keys(),
            key=lambda k: feature_importance[k],
            reverse=True
        )[:10]
        
        # Time-series CV
        cv_scores = self._time_series_cv(dataset)
        
        duration = time.time() - start_time
        
        result = TrainingResult(
            train_accuracy=metrics.get("train_accuracy", 0.0),
            val_accuracy=metrics.get("val_accuracy", 0.0),
            test_accuracy=test_accuracy,
            train_auc=train_auc,
            val_auc=val_auc,
            test_auc=test_auc,
            calibration_metrics=calibration_metrics,
            feature_importance=feature_importance,
            top_features=top_features,
            cv_scores=cv_scores,
            cv_mean=float(np.mean(cv_scores)) if cv_scores else None,
            cv_std=float(np.std(cv_scores)) if cv_scores else None,
            training_timestamp=training_timestamp,
            training_duration_seconds=duration,
            n_samples_train=dataset.n_samples_train,
            n_features=dataset.n_features
        )
        
        logger.info(
            f"Training complete: val_acc={result.val_accuracy:.4f}, "
            f"val_auc={val_auc:.4f}, duration={duration:.1f}s"
        )
        
        return result
    
    def _create_model(self, **override_params) -> TabularModel:
        """Create model based on config."""
        if self.config.model_type == "lightgbm":
            params = {
                "n_estimators": self.config.lgb_n_estimators,
                "max_depth": self.config.lgb_max_depth,
                "learning_rate": self.config.lgb_learning_rate,
                "min_child_samples": self.config.lgb_min_child_samples,
                "subsample": self.config.lgb_subsample,
                "colsample_bytree": self.config.lgb_colsample_bytree,
                "early_stopping_rounds": self.config.early_stopping_rounds,
                "random_state": self.config.random_state,
                **override_params
            }
            return LightGBMClassifier(**params)
        
        elif self.config.model_type == "gradient_boosting":
            params = {
                "n_estimators": self.config.gb_n_estimators,
                "max_depth": self.config.gb_max_depth,
                "learning_rate": self.config.gb_learning_rate,
                "random_state": self.config.random_state,
                **override_params
            }
            return GradientBoostingClassifier(**params)
        
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")
    
    def _time_series_cv(
        self,
        dataset: TrainingDataset
    ) -> list[float]:
        """
        Time-series cross-validation.
        Uses expanding window approach.
        """
        X = np.vstack([dataset.X_train, dataset.X_val])
        y = np.concatenate([dataset.y_train, dataset.y_val])
        
        n = len(X)
        fold_size = n // (self.config.n_folds + 1)
        
        scores = []
        
        for i in range(self.config.n_folds):
            train_end = (i + 1) * fold_size
            val_start = train_end
            val_end = min(val_start + fold_size, n)
            
            if val_end <= val_start:
                continue
            
            X_train_cv = X[:train_end]
            y_train_cv = y[:train_end]
            X_val_cv = X[val_start:val_end]
            y_val_cv = y[val_start:val_end]
            
            # Train fold model
            model = self._create_model()
            model.fit(X_train_cv, y_train_cv)
            
            # Evaluate
            pred = model.predict(X_val_cv)
            accuracy = float(np.mean(pred == y_val_cv))
            scores.append(accuracy)
        
        return scores
    
    def _tune_hyperparameters(
        self,
        dataset: TrainingDataset
    ) -> dict[str, Any]:
        """
        Simple grid search for hyperparameter tuning.
        Returns best parameters.
        """
        if self.config.model_type != "lightgbm":
            return {}
        
        param_grid = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [100, 200, 300]
        }
        
        best_score = 0.0
        best_params = {}
        
        for max_depth in param_grid["max_depth"]:
            for lr in param_grid["learning_rate"]:
                for n_est in param_grid["n_estimators"]:
                    params = {
                        "max_depth": max_depth,
                        "learning_rate": lr,
                        "n_estimators": n_est
                    }
                    
                    model = self._create_model(**params)
                    model.fit(
                        dataset.X_train, dataset.y_train,
                        dataset.X_val, dataset.y_val
                    )
                    
                    val_pred = model.predict(dataset.X_val)
                    accuracy = float(np.mean(val_pred == dataset.y_val))
                    
                    if accuracy > best_score:
                        best_score = accuracy
                        best_params = params
        
        logger.info(f"Best hyperparams: {best_params} with accuracy {best_score:.4f}")
        return best_params
    
    def predict(
        self,
        features: dict[str, float],
        symbol: str
    ):
        """
        Make prediction with optional calibration.
        
        Returns:
            ModelPrediction
        """
        if self._model is None:
            raise ValueError("Model not trained")
        
        prediction = self._model.predict_single(
            features, symbol,
            confidence_threshold=self.config.confidence_threshold
        )
        
        # Apply calibration if available
        if self._calibrator is not None and self._calibrator.is_fitted:
            calibrated_up = float(self._calibrator.calibrate(
                np.array([prediction.probability_up])
            )[0])
            
            prediction.probability_up = calibrated_up
            prediction.probability_down = 1 - calibrated_up
            prediction.predicted_class = 1 if calibrated_up >= 0.5 else 0
            prediction.confidence = abs(calibrated_up - 0.5) * 2
            prediction.is_confident = prediction.confidence >= self.config.confidence_threshold
        
        return prediction
    
    def save(self, path: Path) -> None:
        """Save trainer state (model + calibrator)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        if self._model is not None:
            self._model.save(path / "model")
        
        if self._calibrator is not None:
            self._calibrator.save(path / "calibrator")
        
        # Save config
        import json
        with open(path / "config.json", "w") as f:
            json.dump(self.config.model_dump(), f, indent=2)
        
        logger.info(f"Trainer saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load trainer state."""
        import json
        
        path = Path(path)
        
        # Load config
        with open(path / "config.json") as f:
            self.config = TrainingConfig(**json.load(f))
        
        # Load model
        self._model = self._create_model()
        self._model.load(path / "model")
        
        # Load calibrator if exists
        if (path / "calibrator.joblib").exists():
            self._calibrator = IsotonicCalibrator()
            self._calibrator.load(path / "calibrator")
        
        logger.info(f"Trainer loaded from {path}")


def train_model(
    dataset: TrainingDataset,
    model_type: str = "lightgbm",
    calibrate: bool = True
) -> tuple[Trainer, TrainingResult]:
    """
    Convenience function to train a model.
    
    Args:
        dataset: Training dataset
        model_type: Type of model
        calibrate: Whether to calibrate probabilities
        
    Returns:
        (trainer, result)
    """
    config = TrainingConfig(
        model_type=model_type,
        calibrate=calibrate
    )
    
    trainer = Trainer(config)
    result = trainer.train(dataset)
    
    return trainer, result
