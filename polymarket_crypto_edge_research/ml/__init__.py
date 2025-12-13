"""ML module for models and training."""

from .datasets import (
    DatasetBuilder,
    TrainingDataset,
    create_training_dataset,
)
from .models_tabular import (
    TabularModel,
    LightGBMClassifier,
    GradientBoostingClassifier,
)
from .models_sequence import (
    SequenceModel,
    LSTMClassifier,
)
from .calibration import (
    IsotonicCalibrator,
    PlattCalibrator,
    calibrate_probabilities,
)
from .trainer import (
    Trainer,
    TrainingConfig,
    train_model,
)
from .model_registry import (
    ModelRegistry,
    ModelMetadata,
    load_model,
    save_model,
)

__all__ = [
    # Datasets
    "DatasetBuilder",
    "TrainingDataset",
    "create_training_dataset",
    # Tabular models
    "TabularModel",
    "LightGBMClassifier",
    "GradientBoostingClassifier",
    # Sequence models
    "SequenceModel",
    "LSTMClassifier",
    # Calibration
    "IsotonicCalibrator",
    "PlattCalibrator",
    "calibrate_probabilities",
    # Training
    "Trainer",
    "TrainingConfig",
    "train_model",
    # Registry
    "ModelRegistry",
    "ModelMetadata",
    "load_model",
    "save_model",
]
