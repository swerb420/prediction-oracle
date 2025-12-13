"""
Sequence models for temporal pattern learning.
Includes LSTM and simple RNN classifiers.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from core.logging_utils import get_logger
from ml.models_tabular import ModelPrediction

logger = get_logger(__name__)


class SequenceModel(ABC):
    """Abstract base class for sequence models."""
    
    model_name: str = "sequence_base"
    
    def __init__(self):
        self._model: Any = None
        self._is_fitted: bool = False
        self._feature_names: list[str] = []
        self._sequence_length: int = 12
        self._training_timestamp: datetime | None = None
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,  # Shape: (n_samples, sequence_length, n_features)
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None
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
        return (proba >= 0.5).astype(int).flatten()
    
    @abstractmethod
    def save(self, path: Path) -> None:
        """Save model to disk."""
        pass
    
    @abstractmethod
    def load(self, path: Path) -> None:
        """Load model from disk."""
        pass


class LSTMClassifier(SequenceModel):
    """
    LSTM classifier for sequence data.
    Uses TensorFlow/Keras for implementation.
    
    Falls back to simple dense network if TensorFlow unavailable.
    """
    
    model_name: str = "lstm"
    
    def __init__(
        self,
        sequence_length: int = 12,
        lstm_units: int = 64,
        dense_units: int = 32,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 50,
        batch_size: int = 32,
        early_stopping_patience: int = 5
    ):
        super().__init__()
        self._sequence_length = sequence_length
        self.lstm_units = lstm_units
        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.early_stopping_patience = early_stopping_patience
        
        self._scaler_mean: np.ndarray | None = None
        self._scaler_std: np.ndarray | None = None
    
    def _build_model(self, n_features: int):
        """Build LSTM model architecture."""
        try:
            import tensorflow as tf
            from tensorflow import keras
            from tensorflow.keras import layers
        except ImportError:
            raise ImportError(
                "TensorFlow not installed. Run: pip install tensorflow\n"
                "Or use LightGBM for tabular prediction instead."
            )
        
        model = keras.Sequential([
            layers.LSTM(
                self.lstm_units,
                input_shape=(self._sequence_length, n_features),
                return_sequences=True,
                dropout=self.dropout
            ),
            layers.LSTM(
                self.lstm_units // 2,
                return_sequences=False,
                dropout=self.dropout
            ),
            layers.Dense(self.dense_units, activation="relu"),
            layers.Dropout(self.dropout),
            layers.Dense(1, activation="sigmoid")
        ])
        
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None
    ) -> dict[str, float]:
        """Fit LSTM model."""
        try:
            import tensorflow as tf
            from tensorflow import keras
        except ImportError:
            logger.warning("TensorFlow not available, using fallback")
            return self._fit_fallback(X_train, y_train, X_val, y_val)
        
        from core.time_utils import now_utc
        
        # Validate input shape
        if len(X_train.shape) != 3:
            raise ValueError(f"Expected 3D input, got shape {X_train.shape}")
        
        n_samples, seq_len, n_features = X_train.shape
        self._sequence_length = seq_len
        
        # Normalize data
        self._scaler_mean = X_train.mean(axis=(0, 1))
        self._scaler_std = X_train.std(axis=(0, 1))
        self._scaler_std = np.where(self._scaler_std == 0, 1.0, self._scaler_std)
        
        X_train_norm = (X_train - self._scaler_mean) / self._scaler_std
        if X_val is not None:
            X_val_norm = (X_val - self._scaler_mean) / self._scaler_std
        
        # Build model
        self._model = self._build_model(n_features)
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=self.early_stopping_patience,
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=3,
                min_lr=1e-6
            )
        ]
        
        # Train
        validation_data = (X_val_norm, y_val) if X_val is not None else None
        
        history = self._model.fit(
            X_train_norm, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=1
        )
        
        self._is_fitted = True
        self._training_timestamp = now_utc()
        
        # Calculate metrics
        metrics = {
            "train_loss": float(history.history["loss"][-1]),
            "train_accuracy": float(history.history["accuracy"][-1]),
            "epochs_trained": len(history.history["loss"])
        }
        
        if X_val is not None:
            metrics["val_loss"] = float(history.history["val_loss"][-1])
            metrics["val_accuracy"] = float(history.history["val_accuracy"][-1])
        
        logger.info(f"LSTM trained: {metrics}")
        return metrics
    
    def _fit_fallback(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None,
        y_val: np.ndarray | None
    ) -> dict[str, float]:
        """
        Fallback using sklearn when TensorFlow unavailable.
        Flattens sequences and uses MLP.
        """
        from sklearn.neural_network import MLPClassifier
        from core.time_utils import now_utc
        
        # Flatten sequences
        X_train_flat = X_train.reshape(X_train.shape[0], -1)
        if X_val is not None:
            X_val_flat = X_val.reshape(X_val.shape[0], -1)
        
        self._model = MLPClassifier(
            hidden_layer_sizes=(self.lstm_units, self.dense_units),
            max_iter=self.epochs * 10,
            early_stopping=X_val is not None,
            validation_fraction=0.1 if X_val is None else 0.0,
            random_state=42
        )
        
        self._model.fit(X_train_flat, y_train)
        
        self._is_fitted = True
        self._training_timestamp = now_utc()
        self._is_fallback = True
        
        metrics = {"train_accuracy": float(self._model.score(X_train_flat, y_train))}
        
        if X_val is not None:
            metrics["val_accuracy"] = float(self._model.score(X_val_flat, y_val))
        
        logger.info(f"MLP fallback trained: {metrics}")
        return metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        
        # Handle fallback model
        if hasattr(self, "_is_fallback") and self._is_fallback:
            X_flat = X.reshape(X.shape[0], -1)
            return self._model.predict_proba(X_flat)[:, 1]
        
        # Normalize
        if self._scaler_mean is not None:
            X = (X - self._scaler_mean) / self._scaler_std
        
        return self._model.predict(X).flatten()
    
    def predict_single(
        self,
        sequence: np.ndarray,  # Shape: (sequence_length, n_features)
        symbol: str,
        confidence_threshold: float = 0.6
    ) -> ModelPrediction:
        """
        Make prediction for a single sequence.
        
        Args:
            sequence: Feature sequence array
            symbol: Asset symbol
            confidence_threshold: Minimum confidence threshold
            
        Returns:
            ModelPrediction
        """
        from core.time_utils import now_utc
        
        # Add batch dimension
        X = sequence.reshape(1, *sequence.shape)
        
        # Get probability
        prob_up = float(self.predict_proba(X)[0])
        prob_down = 1 - prob_up
        
        confidence = abs(prob_up - 0.5) * 2
        predicted_class = 1 if prob_up >= 0.5 else 0
        
        return ModelPrediction(
            timestamp=now_utc(),
            symbol=symbol,
            predicted_class=predicted_class,
            probability_up=prob_up,
            probability_down=prob_down,
            confidence=confidence,
            is_confident=confidence >= confidence_threshold,
            model_name=self.model_name,
            model_version=self._training_timestamp.isoformat() if self._training_timestamp else "unknown"
        )
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        import json
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Handle fallback model
        if hasattr(self, "_is_fallback") and self._is_fallback:
            import joblib
            joblib.dump(self._model, path.with_suffix(".joblib"))
        else:
            self._model.save(path.with_suffix(".keras"))
        
        # Save metadata and normalization params
        metadata = {
            "model_name": self.model_name,
            "sequence_length": self._sequence_length,
            "lstm_units": self.lstm_units,
            "dense_units": self.dense_units,
            "training_timestamp": self._training_timestamp.isoformat() if self._training_timestamp else None,
            "is_fallback": getattr(self, "_is_fallback", False),
            "scaler_mean": self._scaler_mean.tolist() if self._scaler_mean is not None else None,
            "scaler_std": self._scaler_std.tolist() if self._scaler_std is not None else None
        }
        
        with open(path.with_suffix(".json"), "w") as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"LSTM model saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        import json
        
        path = Path(path)
        
        # Load metadata
        with open(path.with_suffix(".json")) as f:
            metadata = json.load(f)
        
        self._sequence_length = metadata["sequence_length"]
        self.lstm_units = metadata.get("lstm_units", 64)
        self.dense_units = metadata.get("dense_units", 32)
        
        if metadata.get("training_timestamp"):
            self._training_timestamp = datetime.fromisoformat(metadata["training_timestamp"])
        
        if metadata.get("scaler_mean"):
            self._scaler_mean = np.array(metadata["scaler_mean"])
            self._scaler_std = np.array(metadata["scaler_std"])
        
        # Load model
        if metadata.get("is_fallback", False):
            import joblib
            self._model = joblib.load(path.with_suffix(".joblib"))
            self._is_fallback = True
        else:
            from tensorflow import keras
            self._model = keras.models.load_model(path.with_suffix(".keras"))
        
        self._is_fitted = True
        logger.info(f"LSTM model loaded from {path}")


class SimpleRNNClassifier(SequenceModel):
    """
    Simple RNN classifier as a lighter alternative to LSTM.
    """
    
    model_name: str = "simple_rnn"
    
    def __init__(
        self,
        sequence_length: int = 12,
        rnn_units: int = 32,
        dense_units: int = 16,
        dropout: float = 0.2,
        learning_rate: float = 0.001,
        epochs: int = 30,
        batch_size: int = 32
    ):
        super().__init__()
        self._sequence_length = sequence_length
        self.rnn_units = rnn_units
        self.dense_units = dense_units
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
    
    def _build_model(self, n_features: int):
        """Build Simple RNN model."""
        from tensorflow import keras
        from tensorflow.keras import layers
        
        model = keras.Sequential([
            layers.SimpleRNN(
                self.rnn_units,
                input_shape=(self._sequence_length, n_features),
                return_sequences=False,
                dropout=self.dropout
            ),
            layers.Dense(self.dense_units, activation="relu"),
            layers.Dropout(self.dropout),
            layers.Dense(1, activation="sigmoid")
        ])
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"]
        )
        
        return model
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None
    ) -> dict[str, float]:
        """Fit Simple RNN model."""
        from tensorflow import keras
        from core.time_utils import now_utc
        
        n_samples, seq_len, n_features = X_train.shape
        self._sequence_length = seq_len
        
        self._model = self._build_model(n_features)
        
        validation_data = (X_val, y_val) if X_val is not None else None
        
        history = self._model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=validation_data,
            verbose=1
        )
        
        self._is_fitted = True
        self._training_timestamp = now_utc()
        
        metrics = {
            "train_loss": float(history.history["loss"][-1]),
            "train_accuracy": float(history.history["accuracy"][-1])
        }
        
        if X_val is not None:
            metrics["val_loss"] = float(history.history["val_loss"][-1])
            metrics["val_accuracy"] = float(history.history["val_accuracy"][-1])
        
        return metrics
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities."""
        if not self._is_fitted:
            raise ValueError("Model not fitted")
        return self._model.predict(X).flatten()
    
    def save(self, path: Path) -> None:
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._model.save(path.with_suffix(".keras"))
        logger.info(f"SimpleRNN saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load model from disk."""
        from tensorflow import keras
        path = Path(path)
        self._model = keras.models.load_model(path.with_suffix(".keras"))
        self._is_fitted = True
        logger.info(f"SimpleRNN loaded from {path}")
