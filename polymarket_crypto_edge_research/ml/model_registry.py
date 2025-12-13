"""
Model registry for versioning and persistence.
Tracks model versions, performance, and enables A/B testing.
"""

from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from core.logging_utils import get_logger
from ml.models_tabular import TabularModel, LightGBMClassifier, GradientBoostingClassifier
from ml.calibration import IsotonicCalibrator

logger = get_logger(__name__)


class ModelMetadata(BaseModel):
    """Metadata for a registered model."""
    
    # Identification
    model_id: str
    model_name: str
    model_type: str
    version: str
    
    # Asset/market info
    symbol: str  # BTC, ETH, SOL, or market_id
    prediction_target: str  # "15m_direction", "last_seconds_scalp", etc.
    
    # Training info
    training_timestamp: datetime
    training_samples: int
    n_features: int
    feature_names: list[str] = Field(default_factory=list)
    
    # Performance metrics
    train_accuracy: float
    val_accuracy: float
    test_accuracy: float | None = None
    val_auc: float | None = None
    
    # Calibration
    is_calibrated: bool = False
    calibration_ece: float | None = None
    
    # Status
    is_active: bool = True
    is_champion: bool = False  # Currently deployed model
    
    # File paths
    model_path: str
    calibrator_path: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for storage."""
        data = self.model_dump()
        data["training_timestamp"] = self.training_timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        if isinstance(data.get("training_timestamp"), str):
            data["training_timestamp"] = datetime.fromisoformat(data["training_timestamp"])
        return cls(**data)


class ModelRegistry:
    """
    Central registry for trained models.
    
    Features:
    - Version tracking
    - Performance comparison
    - Champion model selection
    - A/B testing support
    """
    
    def __init__(self, registry_path: Path):
        self.registry_path = Path(registry_path)
        self.models_path = self.registry_path / "models"
        self.index_path = self.registry_path / "index.json"
        
        self._index: dict[str, ModelMetadata] = {}
        self._loaded_models: dict[str, TabularModel] = {}
        self._loaded_calibrators: dict[str, IsotonicCalibrator] = {}
        
        self._ensure_paths()
        self._load_index()
    
    def _ensure_paths(self) -> None:
        """Create registry directories."""
        self.registry_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
    
    def _load_index(self) -> None:
        """Load model index from disk."""
        if not self.index_path.exists():
            return
        
        import json
        
        try:
            with open(self.index_path) as f:
                data = json.load(f)
            
            for model_id, metadata_dict in data.items():
                self._index[model_id] = ModelMetadata.from_dict(metadata_dict)
            
            logger.info(f"Loaded {len(self._index)} models from registry")
        except Exception as e:
            logger.error(f"Failed to load registry index: {e}")
    
    def _save_index(self) -> None:
        """Save model index to disk."""
        import json
        
        data = {
            model_id: metadata.to_dict()
            for model_id, metadata in self._index.items()
        }
        
        with open(self.index_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def register(
        self,
        model: TabularModel,
        metadata: ModelMetadata,
        calibrator: IsotonicCalibrator | None = None
    ) -> str:
        """
        Register a trained model.
        
        Args:
            model: Trained model
            metadata: Model metadata
            calibrator: Optional calibrator
            
        Returns:
            model_id
        """
        # Save model
        model_dir = self.models_path / metadata.model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = model_dir / "model"
        model.save(model_path)
        metadata.model_path = str(model_path)
        
        # Save calibrator
        if calibrator is not None:
            calibrator_path = model_dir / "calibrator"
            calibrator.save(calibrator_path)
            metadata.calibrator_path = str(calibrator_path)
            metadata.is_calibrated = True
        
        # Update index
        self._index[metadata.model_id] = metadata
        self._save_index()
        
        logger.info(f"Registered model: {metadata.model_id}")
        return metadata.model_id
    
    def get_metadata(self, model_id: str) -> ModelMetadata | None:
        """Get model metadata."""
        return self._index.get(model_id)
    
    def list_models(
        self,
        symbol: str | None = None,
        prediction_target: str | None = None,
        active_only: bool = True
    ) -> list[ModelMetadata]:
        """
        List registered models with optional filters.
        """
        models = list(self._index.values())
        
        if symbol:
            models = [m for m in models if m.symbol == symbol]
        
        if prediction_target:
            models = [m for m in models if m.prediction_target == prediction_target]
        
        if active_only:
            models = [m for m in models if m.is_active]
        
        # Sort by validation accuracy
        models.sort(key=lambda m: m.val_accuracy, reverse=True)
        
        return models
    
    def get_champion(
        self,
        symbol: str,
        prediction_target: str = "15m_direction"
    ) -> ModelMetadata | None:
        """Get the champion (deployed) model for a symbol."""
        for metadata in self._index.values():
            if (metadata.symbol == symbol and 
                metadata.prediction_target == prediction_target and
                metadata.is_champion):
                return metadata
        return None
    
    def set_champion(
        self,
        model_id: str
    ) -> None:
        """
        Set a model as the champion for its symbol/target.
        Demotes previous champion.
        """
        if model_id not in self._index:
            raise ValueError(f"Model not found: {model_id}")
        
        metadata = self._index[model_id]
        
        # Demote previous champion
        for other in self._index.values():
            if (other.symbol == metadata.symbol and
                other.prediction_target == metadata.prediction_target and
                other.is_champion):
                other.is_champion = False
        
        # Promote new champion
        metadata.is_champion = True
        self._save_index()
        
        logger.info(f"Set champion model: {model_id} for {metadata.symbol}")
    
    def load_model(
        self,
        model_id: str,
        with_calibrator: bool = True
    ) -> tuple[TabularModel, IsotonicCalibrator | None]:
        """
        Load a model (and calibrator) from registry.
        Uses caching for repeated loads.
        """
        # Check cache
        if model_id in self._loaded_models:
            model = self._loaded_models[model_id]
            calibrator = self._loaded_calibrators.get(model_id) if with_calibrator else None
            return model, calibrator
        
        # Get metadata
        metadata = self._index.get(model_id)
        if metadata is None:
            raise ValueError(f"Model not found: {model_id}")
        
        # Create model instance
        if metadata.model_type == "lightgbm":
            model = LightGBMClassifier()
        elif metadata.model_type == "gradient_boosting":
            model = GradientBoostingClassifier()
        else:
            raise ValueError(f"Unknown model type: {metadata.model_type}")
        
        # Load model
        model.load(Path(metadata.model_path))
        self._loaded_models[model_id] = model
        
        # Load calibrator
        calibrator = None
        if with_calibrator and metadata.calibrator_path:
            calibrator = IsotonicCalibrator()
            calibrator.load(Path(metadata.calibrator_path))
            self._loaded_calibrators[model_id] = calibrator
        
        return model, calibrator
    
    def load_champion(
        self,
        symbol: str,
        prediction_target: str = "15m_direction"
    ) -> tuple[TabularModel, IsotonicCalibrator | None] | None:
        """
        Load the champion model for a symbol.
        
        Returns:
            (model, calibrator) or None if no champion exists
        """
        metadata = self.get_champion(symbol, prediction_target)
        if metadata is None:
            return None
        
        return self.load_model(metadata.model_id)
    
    def compare_models(
        self,
        symbol: str,
        prediction_target: str = "15m_direction",
        top_n: int = 5
    ) -> list[dict[str, Any]]:
        """
        Compare top models for a symbol.
        
        Returns:
            List of comparison dicts
        """
        models = self.list_models(symbol, prediction_target)[:top_n]
        
        comparison = []
        for m in models:
            comparison.append({
                "model_id": m.model_id,
                "version": m.version,
                "val_accuracy": m.val_accuracy,
                "val_auc": m.val_auc,
                "is_champion": m.is_champion,
                "training_samples": m.training_samples,
                "training_date": m.training_timestamp.strftime("%Y-%m-%d %H:%M"),
            })
        
        return comparison
    
    def deactivate(self, model_id: str) -> None:
        """Deactivate a model (keep in registry but don't use)."""
        if model_id in self._index:
            self._index[model_id].is_active = False
            self._index[model_id].is_champion = False
            self._save_index()
            logger.info(f"Deactivated model: {model_id}")
    
    def delete(self, model_id: str) -> None:
        """Delete a model from registry."""
        if model_id not in self._index:
            return
        
        metadata = self._index[model_id]
        
        # Remove files
        import shutil
        model_dir = self.models_path / model_id
        if model_dir.exists():
            shutil.rmtree(model_dir)
        
        # Remove from index
        del self._index[model_id]
        self._save_index()
        
        # Remove from cache
        self._loaded_models.pop(model_id, None)
        self._loaded_calibrators.pop(model_id, None)
        
        logger.info(f"Deleted model: {model_id}")
    
    def cleanup_old_models(
        self,
        symbol: str,
        keep_n: int = 5
    ) -> int:
        """
        Remove old underperforming models, keeping top N.
        Never removes the champion.
        
        Returns:
            Number of models removed
        """
        models = self.list_models(symbol)
        
        # Sort by val_accuracy (best first)
        models.sort(key=lambda m: m.val_accuracy, reverse=True)
        
        removed = 0
        for model in models[keep_n:]:
            if not model.is_champion:
                self.delete(model.model_id)
                removed += 1
        
        if removed > 0:
            logger.info(f"Cleaned up {removed} old models for {symbol}")
        
        return removed


def generate_model_id(symbol: str, model_type: str) -> str:
    """Generate unique model ID."""
    from core.time_utils import now_utc
    timestamp = now_utc().strftime("%Y%m%d_%H%M%S")
    return f"{symbol}_{model_type}_{timestamp}"


def save_model(
    model: TabularModel,
    path: Path,
    calibrator: IsotonicCalibrator | None = None
) -> None:
    """Convenience function to save model outside registry."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    model.save(path / "model")
    
    if calibrator is not None:
        calibrator.save(path / "calibrator")
    
    logger.info(f"Model saved to {path}")


def load_model(
    path: Path,
    model_type: str = "lightgbm"
) -> tuple[TabularModel, IsotonicCalibrator | None]:
    """Convenience function to load model outside registry."""
    path = Path(path)
    
    if model_type == "lightgbm":
        model = LightGBMClassifier()
    elif model_type == "gradient_boosting":
        model = GradientBoostingClassifier()
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.load(path / "model")
    
    calibrator = None
    if (path / "calibrator.joblib").exists():
        calibrator = IsotonicCalibrator()
        calibrator.load(path / "calibrator")
    
    return model, calibrator
