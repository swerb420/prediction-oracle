"""
Learning ML Predictor - Learns ONLY from real data.

NO SYNTHETIC DATA. NO FAKE FEATURES.

This predictor:
1. Starts conservative (50% confidence) when no data
2. Learns from real outcomes as they come in
3. Increases confidence as it gains more training data
4. Uses proper cross-validation to avoid overfitting
5. Tracks calibration (are 80% confident predictions correct 80% of time?)

Cold Start Strategy:
- First 50 examples: Return 50% confidence, don't recommend betting
- 50-200 examples: Use simple model, low confidence
- 200+ examples: Full model, confidence based on validation

Training data comes from:
- real_data_store.py labeled examples (snapshots + outcomes)
"""

import json
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler

from real_data_store import get_store, RealDataStore

logger = logging.getLogger(__name__)

CryptoSymbol = Literal["BTC", "ETH", "SOL", "XRP"]


@dataclass
class MLPrediction:
    """A prediction from the ML model."""
    symbol: CryptoSymbol
    direction: str  # "UP" or "DOWN"
    confidence: float  # 0.0 to 1.0
    should_bet: bool  # Whether we have enough confidence to bet
    model_version: str
    training_examples: int
    calibration_score: float
    reasoning: str


@dataclass
class ModelState:
    """State of the ML model."""
    model: Optional[HistGradientBoostingClassifier] = None
    scaler: Optional[StandardScaler] = None
    training_examples: int = 0
    last_trained: Optional[str] = None
    validation_accuracy: float = 0.0
    calibration_score: float = 0.0  # How well calibrated are probabilities
    feature_names: list[str] = field(default_factory=list)


class LearningMLPredictor:
    """
    ML predictor that learns from real data only.
    
    Cold start: Conservative until we have enough data.
    Learning: Retrains periodically as new labeled data arrives.
    Calibration: Tracks if confidence matches actual accuracy.
    """
    
    # Minimum examples before we start betting
    MIN_EXAMPLES_TO_BET = 50
    MIN_EXAMPLES_FULL_MODEL = 200
    
    # Confidence thresholds
    MIN_CONFIDENCE_TO_BET = 0.60
    
    def __init__(
        self, 
        store: Optional[RealDataStore] = None,
        model_dir: str = "./models",
    ):
        self.store = store or get_store()
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Per-symbol models
        self.models: dict[str, ModelState] = {}
        
        # Load any saved models
        self._load_models()
    
    def _model_path(self, symbol: str) -> Path:
        return self.model_dir / f"ml_model_{symbol.lower()}.pkl"
    
    def _load_models(self):
        """Load saved models from disk."""
        for symbol in ["BTC", "ETH", "SOL", "XRP"]:
            path = self._model_path(symbol)
            if path.exists():
                try:
                    with open(path, "rb") as f:
                        self.models[symbol] = pickle.load(f)
                    logger.info(
                        f"Loaded model for {symbol}: "
                        f"{self.models[symbol].training_examples} examples"
                    )
                except Exception as e:
                    logger.warning(f"Failed to load model for {symbol}: {e}")
                    self.models[symbol] = ModelState()
            else:
                self.models[symbol] = ModelState()
    
    def _save_model(self, symbol: str):
        """Save model to disk."""
        path = self._model_path(symbol)
        try:
            with open(path, "wb") as f:
                pickle.dump(self.models[symbol], f)
            logger.info(f"Saved model for {symbol}")
        except Exception as e:
            logger.error(f"Failed to save model for {symbol}: {e}")
    
    # ─────────────────────────────────────────────────────────────────────────
    # Feature Engineering (from real data only)
    # ─────────────────────────────────────────────────────────────────────────
    
    def _extract_features(self, data: dict) -> dict[str, float]:
        """
        Extract features from a data point.
        
        All features come from real market data:
        - yes_price: Current YES price (market implied probability)
        - no_price: Current NO price
        - market_direction: What the market implies (encoded)
        - volume: Market volume
        - liquidity: Market liquidity
        - price_deviation: How far from 50/50
        - hour_of_day: Time feature
        - day_of_week: Time feature
        """
        features = {}
        
        # Price features
        yes_price = float(data.get("yes_price", 0.5))
        no_price = float(data.get("no_price", 0.5))
        
        features["yes_price"] = yes_price
        features["no_price"] = no_price
        features["price_deviation"] = abs(yes_price - 0.5)
        features["price_spread"] = abs(yes_price - no_price) - 1.0  # Should be ~0
        
        # Volume and liquidity
        features["volume"] = float(data.get("volume", 0))
        features["liquidity"] = float(data.get("liquidity", 0))
        features["log_volume"] = np.log1p(features["volume"])
        features["log_liquidity"] = np.log1p(features["liquidity"])
        
        # Market direction encoded
        market_dir = data.get("market_direction", "")
        features["market_is_up"] = 1.0 if market_dir == "UP" else 0.0
        
        # Time features (if we have timestamp)
        timestamp = data.get("timestamp", "")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                features["hour"] = dt.hour
                features["minute"] = dt.minute
                features["day_of_week"] = dt.weekday()
                features["is_weekend"] = 1.0 if dt.weekday() >= 5 else 0.0
            except:
                features["hour"] = 12
                features["minute"] = 0
                features["day_of_week"] = 0
                features["is_weekend"] = 0.0
        
        return features
    
    def _features_to_array(
        self, 
        features: dict[str, float],
        feature_names: list[str],
    ) -> np.ndarray:
        """Convert features dict to array in consistent order."""
        return np.array([features.get(name, 0.0) for name in feature_names])
    
    # ─────────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────────
    
    def train(self, symbol: Optional[str] = None, force: bool = False):
        """
        Train model(s) on real labeled data.
        
        Args:
            symbol: Train specific symbol, or all if None
            force: Force retrain even if recently trained
        """
        symbols = [symbol] if symbol else ["BTC", "ETH", "SOL", "XRP"]
        
        for sym in symbols:
            self._train_symbol(sym, force)
    
    def _train_symbol(self, symbol: str, force: bool = False):
        """Train model for a specific symbol."""
        # Get labeled data
        all_data = self.store.get_labeled_data(limit=10000)
        
        # Filter for this symbol
        data = [d for d in all_data if d.get("symbol") == symbol]
        
        n_examples = len(data)
        logger.info(f"{symbol}: {n_examples} labeled examples")
        
        state = self.models.get(symbol, ModelState())
        
        # Check if we need to retrain
        if not force and state.training_examples == n_examples:
            logger.info(f"{symbol}: Already trained on {n_examples} examples")
            return
        
        if n_examples < 10:
            logger.info(f"{symbol}: Not enough data to train ({n_examples} < 10)")
            state.training_examples = n_examples
            self.models[symbol] = state
            return
        
        # Extract features and labels
        feature_dicts = [self._extract_features(d) for d in data]
        
        # Get consistent feature names
        feature_names = sorted(feature_dicts[0].keys())
        state.feature_names = feature_names
        
        X = np.array([
            self._features_to_array(fd, feature_names) 
            for fd in feature_dicts
        ])
        y = np.array([1 if d["actual_outcome"] == "UP" else 0 for d in data])
        
        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        state.scaler = scaler
        
        # Train model with cross-validation
        if n_examples < self.MIN_EXAMPLES_FULL_MODEL:
            # Simple model for small data
            model = HistGradientBoostingClassifier(
                max_depth=3,
                max_iter=50,
                learning_rate=0.1,
                min_samples_leaf=5,
            )
        else:
            # Full model for larger data
            model = HistGradientBoostingClassifier(
                max_depth=5,
                max_iter=200,
                learning_rate=0.05,
                min_samples_leaf=10,
            )
        
        # Cross-validation
        if n_examples >= 20:
            cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
            state.validation_accuracy = cv_scores.mean()
            logger.info(
                f"{symbol}: CV accuracy = {cv_scores.mean():.3f} "
                f"(+/- {cv_scores.std():.3f})"
            )
        
        # Fit on all data
        model.fit(X_scaled, y)
        
        # Calibrate probabilities (if enough data)
        if n_examples >= 50:
            try:
                calibrated = CalibratedClassifierCV(model, method='isotonic', cv=3)
                calibrated.fit(X_scaled, y)
                state.model = calibrated
            except:
                state.model = model
        else:
            state.model = model
        
        state.training_examples = n_examples
        state.last_trained = datetime.now(timezone.utc).isoformat()
        
        # Calculate calibration score
        state.calibration_score = self._calculate_calibration(state, X_scaled, y)
        
        self.models[symbol] = state
        self._save_model(symbol)
        
        logger.info(
            f"{symbol}: Trained on {n_examples} examples, "
            f"val_acc={state.validation_accuracy:.3f}, "
            f"calibration={state.calibration_score:.3f}"
        )
    
    def _calculate_calibration(
        self, 
        state: ModelState, 
        X: np.ndarray, 
        y: np.ndarray,
    ) -> float:
        """
        Calculate calibration score.
        
        Good calibration: When we predict 80% confident, we're right 80% of time.
        """
        if state.model is None:
            return 0.0
        
        try:
            probs = state.model.predict_proba(X)[:, 1]
            
            # Bucket predictions by confidence
            buckets = [(0.5, 0.6), (0.6, 0.7), (0.7, 0.8), (0.8, 0.9), (0.9, 1.0)]
            
            total_error = 0.0
            n_buckets = 0
            
            for low, high in buckets:
                mask = (probs >= low) & (probs < high)
                if mask.sum() > 0:
                    expected_acc = (low + high) / 2
                    actual_acc = y[mask].mean()
                    total_error += abs(expected_acc - actual_acc)
                    n_buckets += 1
            
            if n_buckets == 0:
                return 0.5
            
            # Return 1 - average error (1.0 is perfect calibration)
            return 1.0 - (total_error / n_buckets)
            
        except Exception as e:
            logger.error(f"Calibration calculation error: {e}")
            return 0.5
    
    # ─────────────────────────────────────────────────────────────────────────
    # Prediction
    # ─────────────────────────────────────────────────────────────────────────
    
    def predict(
        self, 
        symbol: CryptoSymbol,
        current_data: dict,
    ) -> MLPrediction:
        """
        Make a prediction for a symbol.
        
        Returns conservative prediction if not enough training data.
        """
        state = self.models.get(symbol, ModelState())
        
        # Check if we have enough data
        if state.training_examples < 10 or state.model is None:
            return MLPrediction(
                symbol=symbol,
                direction="UP",
                confidence=0.50,
                should_bet=False,
                model_version="no_model",
                training_examples=state.training_examples,
                calibration_score=0.0,
                reasoning=f"Not enough training data ({state.training_examples} examples). "
                         f"Need at least 10 labeled examples to make predictions.",
            )
        
        # Extract features
        features = self._extract_features(current_data)
        
        if not state.feature_names:
            state.feature_names = sorted(features.keys())
        
        X = self._features_to_array(features, state.feature_names).reshape(1, -1)
        
        # Scale
        if state.scaler:
            X = state.scaler.transform(X)
        
        # Predict
        try:
            prob_up = state.model.predict_proba(X)[0, 1]
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return MLPrediction(
                symbol=symbol,
                direction="UP",
                confidence=0.50,
                should_bet=False,
                model_version="error",
                training_examples=state.training_examples,
                calibration_score=0.0,
                reasoning=f"Prediction error: {e}",
            )
        
        # Determine direction and confidence
        if prob_up >= 0.5:
            direction = "UP"
            confidence = prob_up
        else:
            direction = "DOWN"
            confidence = 1.0 - prob_up
        
        # Adjust confidence based on training data quality
        if state.training_examples < self.MIN_EXAMPLES_TO_BET:
            # Reduce confidence when we have little data
            confidence = 0.5 + (confidence - 0.5) * 0.5
            should_bet = False
            reasoning = (
                f"Limited training data ({state.training_examples}/{self.MIN_EXAMPLES_TO_BET}). "
                f"Raw confidence: {confidence:.1%}. Recommending no bet."
            )
        elif state.training_examples < self.MIN_EXAMPLES_FULL_MODEL:
            # Moderate confidence with moderate data
            confidence = 0.5 + (confidence - 0.5) * 0.7
            should_bet = confidence >= self.MIN_CONFIDENCE_TO_BET
            reasoning = (
                f"Moderate training data ({state.training_examples} examples). "
                f"Validation accuracy: {state.validation_accuracy:.1%}. "
                f"Calibration: {state.calibration_score:.1%}."
            )
        else:
            # Full confidence with enough data
            should_bet = confidence >= self.MIN_CONFIDENCE_TO_BET
            reasoning = (
                f"Full model ({state.training_examples} examples). "
                f"Validation accuracy: {state.validation_accuracy:.1%}. "
                f"Calibration: {state.calibration_score:.1%}."
            )
        
        return MLPrediction(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            should_bet=should_bet,
            model_version=f"v{state.training_examples}",
            training_examples=state.training_examples,
            calibration_score=state.calibration_score,
            reasoning=reasoning,
        )
    
    def get_model_status(self) -> dict:
        """Get status of all models."""
        status = {}
        for symbol in ["BTC", "ETH", "SOL", "XRP"]:
            state = self.models.get(symbol, ModelState())
            status[symbol] = {
                "training_examples": state.training_examples,
                "validation_accuracy": state.validation_accuracy,
                "calibration_score": state.calibration_score,
                "last_trained": state.last_trained,
                "ready_to_bet": state.training_examples >= self.MIN_EXAMPLES_TO_BET,
            }
        return status


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    predictor = LearningMLPredictor()
    
    # Try to train
    predictor.train()
    
    # Get status
    print("\nModel Status:")
    status = predictor.get_model_status()
    for symbol, s in status.items():
        print(f"  {symbol}: {s['training_examples']} examples, "
              f"acc={s['validation_accuracy']:.1%}, ready={s['ready_to_bet']}")
    
    # Make a test prediction
    test_data = {
        "yes_price": 0.55,
        "no_price": 0.45,
        "market_direction": "UP",
        "volume": 1000,
        "liquidity": 5000,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    pred = predictor.predict("BTC", test_data)
    print(f"\nTest Prediction: {pred.direction} ({pred.confidence:.1%})")
    print(f"Should bet: {pred.should_bet}")
    print(f"Reasoning: {pred.reasoning}")
