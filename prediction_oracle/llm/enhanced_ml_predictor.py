"""
Enhanced ML Predictor with Multi-Venue and Whale Features
=========================================================

Extends base ML predictor with:
1. Enhanced feature set (31 features vs 19)
2. Whale/venue signal integration
3. Confidence calibration using clean_score
4. Quality-gated predictions

Target: 63%+ win rate through additional alpha sources.
"""

import asyncio
import logging
import pickle
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel

from .crypto_data import CandleData, CryptoSymbol, fetch_crypto_data
from .enhanced_features import (
    EnhancedFeatureSet,
    FeatureQualityFilter,
    extract_enhanced_features,
    extract_enhanced_features_batch,
)
from .feature_engineering import FeatureSet, extract_features
from .multi_venue_client import CrossVenueFeatures, get_cross_venue_snapshot
from .poly_whale_client import WhaleConsensus, get_whale_signals

logger = logging.getLogger(__name__)

Direction = Literal["UP", "DOWN", "NEUTRAL"]


class EnhancedMLPrediction(BaseModel):
    """Enhanced ML prediction with whale/venue context."""
    symbol: CryptoSymbol
    direction: Direction
    confidence: float
    probability_up: float
    
    # Feature contributions
    features_summary: dict[str, float]
    
    # Enhanced signals
    whale_consensus: float
    venue_consensus: float
    signal_alignment: float
    clean_score: float
    
    # Quality gate
    quality_gate_passed: bool
    quality_gate_reasons: list[str]
    
    timestamp: datetime


class EnhancedGradientBoostingPredictor:
    """
    Enhanced Gradient Boosting classifier using 31 features.
    
    Features:
    - 19 base technical features
    - 5 cross-venue features
    - 5 whale features
    - 2 combined features
    """
    
    def __init__(self, model_dir: str = "./models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.models: dict[str, object] = {}
        self.feature_names = EnhancedFeatureSet.feature_names()
        self._sklearn_available = False
        self._check_sklearn()
        self.quality_filter = FeatureQualityFilter()
    
    def _check_sklearn(self):
        """Check sklearn availability."""
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
                max_iter=150,
                learning_rate=0.08,
                max_depth=6,
                min_samples_leaf=15,
                validation_fraction=0.15,
                early_stopping=True,
                n_iter_no_change=15,
                random_state=42,
                l2_regularization=0.1,  # Prevent overfitting
            )
        return None
    
    async def train_on_history(
        self,
        symbol: CryptoSymbol,
        candle_data: CandleData,
        lookahead_candles: int = 1,
    ) -> dict:
        """
        Train model on historical data with enhanced features.
        
        Note: Historical training uses base features only since
        we don't have historical whale/venue data. At inference,
        we use the full enhanced features but set whale/venue to defaults
        for training samples.
        """
        if not self._sklearn_available:
            return {"status": "heuristic", "samples": 0}
        
        candles = candle_data.candles
        if len(candles) < 100:
            return {"status": "insufficient_data", "samples": len(candles)}
        
        X = []
        y = []
        
        for i in range(50, len(candles) - lookahead_candles):
            subset = CandleData(
                symbol=symbol,
                interval=candle_data.interval,
                candles=candles[:i+1]
            )
            
            # Base features
            base = extract_features(subset)
            
            # Create enhanced features with neutral whale/venue
            enhanced = EnhancedFeatureSet.from_components(
                base=base,
                venue=None,  # No historical venue data
                whale=None,  # No historical whale data
            )
            
            X.append(enhanced.to_array())
            
            # Label
            future_price = candles[i + lookahead_candles].close
            current_price = candles[i].close
            label = 1 if future_price > current_price else 0
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        logger.info(f"Training enhanced {symbol} model on {len(X)} samples")
        
        model = self._create_model()
        model.fit(X, y)
        
        # Save model
        model_path = self.model_dir / f"{symbol}_enhanced_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        self.models[symbol] = model
        
        # Calculate metrics
        train_preds = model.predict(X)
        accuracy = np.mean(train_preds == y)
        
        return {
            "status": "trained",
            "samples": len(X),
            "features": len(self.feature_names),
            "train_accuracy": float(accuracy),
            "up_ratio": float(np.mean(y))
        }
    
    def load_model(self, symbol: CryptoSymbol) -> bool:
        """Load pre-trained enhanced model."""
        model_path = self.model_dir / f"{symbol}_enhanced_model.pkl"
        if model_path.exists():
            with open(model_path, 'rb') as f:
                self.models[symbol] = pickle.load(f)
            logger.info(f"Loaded enhanced model for {symbol}")
            return True
        return False
    
    def predict(self, features: EnhancedFeatureSet) -> EnhancedMLPrediction:
        """Make prediction with enhanced features."""
        symbol = features.symbol
        
        if symbol in self.models and self._sklearn_available:
            model = self.models[symbol]
            X = features.to_array().reshape(1, -1)
            
            prob_up = float(model.predict_proba(X)[0][1])
            
            if prob_up > 0.55:
                direction: Direction = "UP"
                confidence = prob_up
            elif prob_up < 0.45:
                direction = "DOWN"
                confidence = 1 - prob_up
            else:
                direction = "NEUTRAL"
                confidence = 0.5
            
            # Get feature importances (HistGradientBoosting doesn't have feature_importances_)
            # Use feature values as summary instead
            top_features = {
                "rsi_14": features.rsi_14,
                "macd_signal": features.macd_signal,
                "bb_position": features.bb_position,
                "whale_consensus": features.whale_consensus,
                "venue_consensus": features.venue_consensus,
                "signal_alignment": features.signal_alignment,
                "clean_score": features.clean_score,
                "prob_up": prob_up,
            }
            
        else:
            # Enhanced heuristic
            prob_up = self._enhanced_heuristic(features)
            
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
                "whale_consensus": features.whale_consensus,
                "venue_consensus": features.venue_consensus,
                "signal_alignment": features.signal_alignment,
            }
        
        # Quality gate
        gate_passed, gate_reasons = self.quality_filter.should_trade(
            features, direction
        )
        
        # Adjust confidence based on quality
        if gate_passed:
            confidence = self.quality_filter.get_confidence_adjustment(
                features, confidence
            )
        else:
            confidence = min(confidence, 0.55)  # Cap if gate fails
        
        return EnhancedMLPrediction(
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            probability_up=prob_up,
            features_summary=top_features,
            whale_consensus=features.whale_consensus,
            venue_consensus=features.venue_consensus,
            signal_alignment=features.signal_alignment,
            clean_score=features.clean_score,
            quality_gate_passed=gate_passed,
            quality_gate_reasons=gate_reasons,
            timestamp=datetime.now(timezone.utc),
        )
    
    def _enhanced_heuristic(self, f: EnhancedFeatureSet) -> float:
        """
        Enhanced heuristic using whale and venue signals.
        """
        signals = []
        weights = []
        
        # Base technical signals
        if f.rsi_14 < 30:
            signals.append(0.7)
        elif f.rsi_14 > 70:
            signals.append(0.3)
        else:
            signals.append(0.5)
        weights.append(1.5)
        
        if f.macd_signal > 0.001:
            signals.append(0.65)
        elif f.macd_signal < -0.001:
            signals.append(0.35)
        else:
            signals.append(0.5)
        weights.append(1.2)
        
        if f.bb_position < -0.8:
            signals.append(0.65)
        elif f.bb_position > 0.8:
            signals.append(0.35)
        else:
            signals.append(0.5)
        weights.append(1.0)
        
        # Whale consensus (high weight if participation is good)
        whale_weight = 2.0 if f.whale_participation > 0.3 else 0.5
        whale_signal = 0.5 + (f.whale_consensus * 0.25)  # Map [-1,1] to [0.25, 0.75]
        signals.append(whale_signal)
        weights.append(whale_weight)
        
        # Venue consensus
        venue_weight = 1.5 if f.venue_count >= 3 else 0.5
        venue_signal = 0.5 + (f.venue_consensus * 0.2)
        signals.append(venue_signal)
        weights.append(venue_weight)
        
        # Signal alignment (strong factor)
        align_signal = 0.5 + (f.signal_alignment * 0.2)
        signals.append(align_signal)
        weights.append(1.8)
        
        # Weighted average
        prob = float(np.average(signals, weights=weights))
        return float(np.clip(prob, 0.3, 0.7))


class EnhancedCryptoMLPredictor:
    """
    High-level interface for enhanced crypto predictions.
    Integrates all data sources and ML prediction.
    """
    
    def __init__(self, model_dir: str = "./models"):
        self.predictor = EnhancedGradientBoostingPredictor(model_dir)
        self._initialized = False
    
    async def initialize(self, retrain: bool = False):
        """Initialize with trained models."""
        symbols: list[CryptoSymbol] = ["BTC", "ETH", "SOL", "XRP"]
        
        for symbol in symbols:
            if not retrain and self.predictor.load_model(symbol):
                continue
            
            logger.info(f"Training enhanced model for {symbol}...")
            try:
                from .crypto_data import get_fetcher
                fetcher = get_fetcher()
                candle_data = await fetcher.get_candles(symbol, "15m", 500)
                result = await self.predictor.train_on_history(symbol, candle_data)
                logger.info(f"{symbol} enhanced training: {result}")
            except Exception as e:
                logger.warning(f"Failed to train {symbol}: {e}")
        
        self._initialized = True
    
    async def predict_symbol(
        self,
        symbol: CryptoSymbol,
        include_whale: bool = True,
        include_venue: bool = True,
    ) -> EnhancedMLPrediction:
        """
        Get enhanced prediction for a symbol.
        
        Args:
            symbol: Crypto symbol
            include_whale: Fetch whale signals (adds latency)
            include_venue: Fetch venue data (adds latency)
        """
        if not self._initialized:
            await self.initialize()
        
        from .crypto_data import get_fetcher
        fetcher = get_fetcher()
        
        # Fetch candle data
        candle_data = await fetcher.get_candles(symbol, "15m", 50)
        
        # Extract enhanced features
        features = await extract_enhanced_features(
            candle_data,
            include_venue=include_venue,
            include_whale=include_whale,
        )
        
        return self.predictor.predict(features)
    
    async def predict_all(
        self,
        include_whale: bool = True,
        include_venue: bool = True,
    ) -> dict[CryptoSymbol, EnhancedMLPrediction]:
        """Get predictions for all symbols."""
        if not self._initialized:
            await self.initialize()
        
        candle_data = await fetch_crypto_data(limit=50)
        
        # Extract features for all
        features = await extract_enhanced_features_batch(
            candle_data,
            include_venue=include_venue,
            include_whale=include_whale,
        )
        
        predictions = {}
        for symbol, feat in features.items():
            predictions[symbol] = self.predictor.predict(feat)
        
        return predictions


# Singleton
_enhanced_predictor: EnhancedCryptoMLPredictor | None = None


def get_enhanced_predictor() -> EnhancedCryptoMLPredictor:
    """Get or create global enhanced predictor."""
    global _enhanced_predictor
    if _enhanced_predictor is None:
        _enhanced_predictor = EnhancedCryptoMLPredictor()
    return _enhanced_predictor


async def predict_with_whale_signals(
    symbol: CryptoSymbol | None = None,
) -> dict[CryptoSymbol, EnhancedMLPrediction] | EnhancedMLPrediction:
    """
    Convenience function for enhanced predictions.
    
    Args:
        symbol: Specific symbol or None for all
    """
    predictor = get_enhanced_predictor()
    
    if symbol:
        return await predictor.predict_symbol(symbol)
    else:
        return await predictor.predict_all()
