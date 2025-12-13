"""
Tabular ML models for price direction prediction.
Includes LightGBM and Gradient Boosting classifiers.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from core.logging_utils import get_logger

logger = get_logger(__name__)


class ModelPrediction(BaseModel):
    """Model prediction output."""
    
    timestamp: datetime
    symbol: str
    
    # Predictions
    predicted_class: int  # 0=down, 1=up
    probability_up: float = Field(ge=0.0, le=1.0)
    probability_down: float = Field(ge=0.0, le=1.0)
    
    # Confidence metrics
    confidence: float = Field(ge=0.0, le=1.0)
    is_confident: bool = False  # Above confidence threshold
    
    # Model info
    model_name: str = ""
    model_version: str = ""
    
    @property
    def direction(self) -> str:
        return "up" if self.predicted_class == 1 else "down"


class TabularModel(ABC):
    """Abstract base class for tabular ML models."""
    
    model_name: str = "base"
    
    def __init__(self):
        self._model: Any = None
        self._is_fitted: bool = False
        self._feature_names: list[str] = []
        self._training_timestamp: datetime | None = None
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None
    ) -> dict[str, float]:
        """Fit the model and return training metrics."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities."""
        pass
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)
    
    def predict_single(
        self,
        features: dict[str, float],
        symbol: str,
        confidence_threshold: float = 0.6
    ) -> ModelPrediction:
        """
        Make prediction for a single sample.
        
        Args:
            features: Feature dict
            symbol: Asset symbol
            confidence_threshold: Minimum confidence for is_confident flag
            
        Returns:
            ModelPrediction
        """
        from core.time_utils import now_utc
        
        # Build feature vector
        X = np.array([[features.get(name, 0.0) for name in self._feature_names]])
        
        # Get probabilities
        proba = self.predict_proba(X)[0]
        prob_down, prob_up = proba[0], proba[1]
        
        # Calculate confidence (distance from 0.5)
        confidence = abs(prob_up - 0.5) * 2
        
        predicted_class = 1 if prob_up >= 0.5 else 0
        
        return ModelPrediction(
            timestamp=now_utc(),
            symbol=symbol,
            predicted_class=predicted_class,
            probability_up=float(prob_up),
            probability_down=float(prob_down),
            confidence=confidence,
            is_confident=confidence >= confidence_threshold,
            model_name=self.model_name,
            model_version=self._training_timestamp.isoformat() if self._training_timestamp else "unknown"
        )
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        pass
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance scores."""
        return {}


class LightGBMClassifier(TabularModel):
    """
    LightGBM binary classifier for price direction.
    Optimized for speed and accuracy on tabular data.
    """
    
    model_name: str = "lightgbm"
    
    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        learning_rate: float = 0.05,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 0.1,
        early_stopping_rounds: int = 20,
        random_state: int = 42
    ):
        super().__init__()
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_child_samples": min_child_samples,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_alpha": reg_alpha,
            "reg_lambda": reg_lambda,
            "random_state": random_state,
            "objective": "binary",
            "metric": "binary_logloss",
            "verbose": -1,
            "n_jobs": -1
        }
        self.early_stopping_rounds = early_stopping_rounds
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None
    ) -> dict[str, float]:
        """Fit LightGBM model."""
        try:
            import lightgbm as lgb
        except ImportError:
            raise ImportError("LightGBM not installed. Run: pip install lightgbm")
        
        from core.time_utils import now_utc
        
        self._feature_names = feature_names or [f"f_{i}" for i in range(X_train.shape[1])]
        
        # Create datasets
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self._feature_names)
        
        callbacks = [lgb.log_evaluation(period=50)]
        valid_sets = [train_data]
        valid_names = ["train"]
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append("valid")
            callbacks.append(lgb.early_stopping(self.early_stopping_rounds))
        
        # Train model
        self._model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.params["n_estimators"],
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        self._is_fitted = True
        self._training_timestamp = now_utc()
        
        # Calculate metrics
        metrics = {}
        train_pred = self.predict(X_train)
        metrics["train_accuracy"] = float(np.mean(train_pred == y_train))
        
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            metrics["val_accuracy"] = float(np.mean(val_pred == y_val))
            
            # Calculate AUC
            from sklearn.metrics import roc_auc_score
            val_proba = self.predict_proba(X_val)[:, 1]
            try:
                metrics["val_auc"] = float(roc_auc_score(y_val, val_proba))
            except ValueError:
                metrics["val_auc"] = 0.5
        
        metrics["n_estimators"] = self._model.num_trees()
        
        logger.info(f"LightGBM trained: {metrics}")
        return metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        proba_up = self._model.predict(X)
        proba_down = 1 - proba_up
        
        return np.column_stack([proba_down, proba_up])
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from LightGBM."""
        if not self._is_fitted:
            return {}
        
        importance = self._model.feature_importance(importance_type="gain")
        total = importance.sum()
        
        if total == 0:
            return {}
        
        return {
            name: float(imp / total)
            for name, imp in zip(self._feature_names, importance)
        }
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        import json
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        self._model.save_model(str(path.with_suffix(".lgb")))
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "feature_names": self._feature_names,
            "training_timestamp": self._training_timestamp.isoformat() if self._training_timestamp else None,
            "params": self.params
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        import json
        import lightgbm as lgb
        
        path = Path(path)
        
        # Load model
        self._model = lgb.Booster(model_file=str(path.with_suffix(".lgb")))
        
        # Load metadata
        with open(path.with_suffix(".json")) as f:
            metadata = json.load(f)
        
        self._feature_names = metadata["feature_names"]
        if metadata.get("training_timestamp"):
            self._training_timestamp = datetime.fromisoformat(metadata["training_timestamp"])
        
        self._is_fitted = True
        logger.info(f"Model loaded from {path}")


class GradientBoostingClassifier(TabularModel):
    """
    Scikit-learn Gradient Boosting classifier.
    Fallback when LightGBM is unavailable.
    """
    
    model_name: str = "gradient_boosting"
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 5,
        learning_rate: float = 0.1,
        min_samples_leaf: int = 20,
        subsample: float = 0.8,
        random_state: int = 42
    ):
        super().__init__()
        self.params = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "min_samples_leaf": min_samples_leaf,
            "subsample": subsample,
            "random_state": random_state
        }
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
        feature_names: list[str] | None = None
    ) -> dict[str, float]:
        """Fit Gradient Boosting model."""
        from sklearn.ensemble import GradientBoostingClassifier as GBC
        from core.time_utils import now_utc
        
        self._feature_names = feature_names or [f"f_{i}" for i in range(X_train.shape[1])]
        
        self._model = GBC(**self.params)
        self._model.fit(X_train, y_train)
        
        self._is_fitted = True
        self._training_timestamp = now_utc()
        
        # Calculate metrics
        metrics = {}
        train_pred = self.predict(X_train)
        metrics["train_accuracy"] = float(np.mean(train_pred == y_train))
        
        if X_val is not None and y_val is not None:
            val_pred = self.predict(X_val)
            metrics["val_accuracy"] = float(np.mean(val_pred == y_val))
            
            from sklearn.metrics import roc_auc_score
            val_proba = self.predict_proba(X_val)[:, 1]
            try:
                metrics["val_auc"] = float(roc_auc_score(y_val, val_proba))
            except ValueError:
                metrics["val_auc"] = 0.5
        
        logger.info(f"GradientBoosting trained: {metrics}")
        return metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model.predict_proba(X)
    
    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance."""
        if not self._is_fitted:
            return {}
        
        importance = self._model.feature_importances_
        return {
            name: float(imp)
            for name, imp in zip(self._feature_names, importance)
        }
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        import json
        import joblib
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save model
        joblib.dump(self._model, path.with_suffix(".joblib"))
        
        # Save metadata
        metadata = {
            "model_name": self.model_name,
            "feature_names": self._feature_names,
            "training_timestamp": self._training_timestamp.isoformat() if self._training_timestamp else None,
            "params": self.params
        }
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Model saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        import json
        import joblib
        
        path = Path(path)
        
        self._model = joblib.load(path.with_suffix(".joblib"))
        
        with open(path.with_suffix(".json")) as f:
            metadata = json.load(f)
        
        self._feature_names = metadata["feature_names"]
        if metadata.get("training_timestamp"):
            self._training_timestamp = datetime.fromisoformat(metadata["training_timestamp"])
        
        self._is_fitted = True
        logger.info(f"Model loaded from {path}")


def create_model(model_type: str = "lightgbm", **kwargs) -> TabularModel:
    """Factory function to create models."""
    if model_type == "lightgbm":
        return LightGBMClassifier(**kwargs)
    elif model_type in ("gradient_boosting", "sklearn"):
        return GradientBoostingClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
