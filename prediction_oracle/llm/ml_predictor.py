"""
ML-based price direction predictor for 15-minute crypto trading.
Uses LightGBM/Gradient Boosting for fast, accurate predictions.
"""

import asyncio
import logging
import pickle
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from .crypto_data import CandleData, CryptoSymbol, fetch_crypto_data
from .feature_engineering import FeatureSet, extract_features

logger = logging.getLogger(__name__)

Direction = Literal["UP", "DOWN", "NEUTRAL"]


class MLPrediction(BaseModel):
    """ML model prediction output."""
    symbol: CryptoSymbol
    direction: Direction
    confidence: float  # 0.5 to 1.0
    probability_up: float  # Raw probability of price going up
    features_summary: dict[str, float]  # Top contributing features
    timestamp: datetime


class GradientBoostingPredictor:
    """
    Gradient Boosting classifier for price direction.
    Uses scikit-learn's HistGradientBoosting for speed.
    Can self-train on historical data.
    """
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.models: dict[str, object] = {}  # Symbol -> trained model
        self.feature_names = FeatureSet.feature_names()
        self._sklearn_available = False
        self._check_sklearn()
    
    def _check_sklearn(self):
        """Check if sklearn is available."""
        try:
            from sklearn.ensemble import HistGradientBoostingClassifier
            self._sklearn_available = True
        except ImportError:
            logger.warning("scikit-learn not installed. Using heuristic model.")
            self._sklearn_available = False
    
    def _create_model(self):
        """Create a new untrained model."""
        if self._sklearn_available:
            from sklearn.ensemble import HistGradientBoostingClassifier
            return HistGradientBoostingClassifier(
                max_iter=100,
                learning_rate=0.1,
                max_depth=5,
                min_samples_leaf=20,
                validation_fraction=0.1,
                early_stopping=True,
                n_iter_no_change=10,
                random_state=42
            )
        return None
    
    async def train_on_history(
        self,
        symbol: CryptoSymbol,
        candle_data: CandleData,
        lookahead_candles: int = 1
    ) -> dict:
        """
        Train model on historical data.
        
        Labels: 1 if price goes UP in next `lookahead_candles`, else 0.
        """
        if not self._sklearn_available:
            logger.info(f"Sklearn not available, using heuristic model for {symbol}")
            return {"status": "heuristic", "samples": 0}
        
        candles = candle_data.candles
        if len(candles) < 50:
            logger.warning(f"Not enough data to train for {symbol}")
            return {"status": "insufficient_data", "samples": len(candles)}
        
        # Generate training samples
        X = []
        y = []
        
        # Need at least 30 candles for features + lookahead for label
        for i in range(30, len(candles) - lookahead_candles):
            # Create subset of candles up to point i
            subset = CandleData(
                symbol=symbol,
                interval=candle_data.interval,
                candles=candles[:i+1]
            )
            
            features = extract_features(subset)
            X.append(features.to_array())
            
            # Label: did price go up?
            future_price = candles[i + lookahead_candles].close
            current_price = candles[i].close
            label = 1 if future_price > current_price else 0
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Training {symbol} model on {len(X)} samples")
        
        model = self._create_model()
        model.fit(X, y)
        
        # Save model
        model_path = self.model_dir / f"{symbol}_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        self.models[symbol] = model
        
        # Calculate training accuracy
        train_preds = model.predict(X)
        accuracy = np.mean(train_preds == y)
        
        return {
            "status": "trained",
            "samples": len(X),
            "train_accuracy": float(accuracy),
            "up_ratio": float(np.mean(y))
        }
    
    def load_model(self, symbol: CryptoSymbol) -> bool:
        """Load a pre-trained model from disk."""
        model_path = self.model_dir / f"{symbol}_model.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.models[symbol] = pickle.load(f)
            logger.info(f"Loaded model for {symbol}")
            return True
        return False
    
    def predict(self, features: FeatureSet) -> MLPrediction:
        """
        Make a prediction for the given features.
        Returns direction and confidence.
        """
        symbol = features.symbol
        
        # Try ML model first
        if symbol in self.models and self._sklearn_available:
            model = self.models[symbol]
            X = features.to_array().reshape(1, -1)
            
            prob_up = model.predict_proba(X)[0][1]
            
            if prob_up > 0.55:
                direction = "UP"
                confidence = prob_up
            elif prob_up < 0.45:
                direction = "DOWN"
                confidence = 1 - prob_up
            else:
                direction = "NEUTRAL"
                confidence = 0.5
            
            # Get feature importances
            importances = dict(zip(
                self.feature_names,
                model.feature_importances_
            ))
            top_features = dict(sorted(
                importances.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:5])
            
        else:
            # Heuristic model based on technical indicators
            prob_up = self._heuristic_predict(features)
            
            if prob_up > 0.55:
                direction = "UP"
                confidence = prob_up
            elif prob_up < 0.45:
                direction = "DOWN"
                confidence = 1 - prob_up
            else:
                direction = "NEUTRAL"
                confidence = 0.5
            
            top_features = {
                "rsi_14": features.rsi_14,
                "macd_signal": features.macd_signal,
                "bb_position": features.bb_position,
                "price_change_3": features.price_change_3,
                "volume_ratio_6": features.volume_ratio_6,
            }
        
        return MLPrediction(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            probability_up=prob_up,
            features_summary=top_features,
            timestamp=datetime.utcnow()
        )
    
    def _heuristic_predict(self, f: FeatureSet) -> float:
        """
        Simple heuristic model when sklearn is not available.
        Combines multiple signals into a probability.
        """
        signals = []
        weights = []
        
        # RSI signal (oversold = bullish, overbought = bearish)
        if f.rsi_14 < 30:
            signals.append(0.7)  # Oversold - likely to bounce up
        elif f.rsi_14 > 70:
            signals.append(0.3)  # Overbought - likely to drop
        else:
            signals.append(0.5)
        weights.append(1.5)
        
        # MACD signal
        if f.macd_signal > 0.001:
            signals.append(0.65)  # Bullish
        elif f.macd_signal < -0.001:
            signals.append(0.35)  # Bearish
        else:
            signals.append(0.5)
        weights.append(1.2)
        
        # Bollinger position
        if f.bb_position < -0.8:
            signals.append(0.65)  # Near lower band
        elif f.bb_position > 0.8:
            signals.append(0.35)  # Near upper band
        else:
            signals.append(0.5)
        weights.append(1.0)
        
        # Price momentum
        if f.price_change_3 > 0.01:
            signals.append(0.55)  # Upward momentum
        elif f.price_change_3 < -0.01:
            signals.append(0.45)  # Downward momentum
        else:
            signals.append(0.5)
        weights.append(0.8)
        
        # Volume confirmation
        if f.volume_ratio_6 > 1.5 and f.price_change_1 > 0:
            signals.append(0.6)  # High volume + up = bullish
        elif f.volume_ratio_6 > 1.5 and f.price_change_1 < 0:
            signals.append(0.4)  # High volume + down = bearish
        else:
            signals.append(0.5)
        weights.append(0.7)
        
        # Weighted average
        prob = np.average(signals, weights=weights)
        
        # Clamp to reasonable range
        return float(np.clip(prob, 0.3, 0.7))


class CryptoMLPredictor:
    """
    High-level interface for crypto price predictions.
    Combines data fetching, feature engineering, and ML prediction.
    """
    
    def __init__(self, model_dir: str = "./models"):
        self.predictor = GradientBoostingPredictor(model_dir)
        self._initialized = False
    
    async def initialize(self, retrain: bool = False):
        """
        Initialize the predictor with trained models.
        Loads existing models or trains new ones.
        """
        symbols: list[CryptoSymbol] = ["BTC", "ETH", "SOL", "XRP"]
        
        for symbol in symbols:
            if not retrain and self.predictor.load_model(symbol):
                continue
            
            # Fetch historical data and train
            logger.info(f"Training model for {symbol}...")
            try:
                from .crypto_data import get_fetcher
                fetcher = get_fetcher()
                # Get 500 candles (~5 days of 15m data)
                candle_data = await fetcher.get_candles(symbol, "15m", 500)
                result = await self.predictor.train_on_history(symbol, candle_data)
                logger.info(f"{symbol} training result: {result}")
            except Exception as e:
                logger.warning(f"Failed to train {symbol}: {e}")
        
        self._initialized = True
    
    async def predict_all(self) -> dict[CryptoSymbol, MLPrediction]:
        """
        Get predictions for all symbols.
        Fetches latest data and runs prediction.
        """
        if not self._initialized:
            await self.initialize()
        
        # Fetch latest data
        candle_data = await fetch_crypto_data(limit=50)
        
        predictions = {}
        for symbol, data in candle_data.items():
            features = extract_features(data)
            pred = self.predictor.predict(features)
            predictions[symbol] = pred
        
        return predictions
    
    async def predict_symbol(self, symbol: CryptoSymbol) -> MLPrediction:
        """Get prediction for a single symbol."""
        if not self._initialized:
            await self.initialize()
        
        from .crypto_data import get_fetcher
        fetcher = get_fetcher()
        
        candle_data = await fetcher.get_candles(symbol, "15m", 50)
        features = extract_features(candle_data)
        return self.predictor.predict(features)


# Singleton instance
_predictor: CryptoMLPredictor | None = None


def get_predictor() -> CryptoMLPredictor:
    """Get or create global predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = CryptoMLPredictor()
    return _predictor


async def predict_crypto_direction(
    symbol: CryptoSymbol | None = None
) -> dict[CryptoSymbol, MLPrediction] | MLPrediction:
    """
    Convenience function for predictions.
    
    Args:
        symbol: Specific symbol to predict, or None for all
        
    Returns:
        Dict of predictions or single prediction
    """
    predictor = get_predictor()
    
    if symbol:
        return await predictor.predict_symbol(symbol)
    else:
        return await predictor.predict_all()
