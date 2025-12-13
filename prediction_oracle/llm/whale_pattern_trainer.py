"""
Whale Pattern ML Trainer - Train ML models on top trader behavior.

Uses data from top_trader_downloader to learn:
1. When whales are bullish/bearish
2. How whale consensus correlates with market direction
3. Timing patterns (when do whales trade before moves)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Literal

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

logger = logging.getLogger(__name__)

CryptoSymbol = Literal["BTC", "ETH", "SOL", "XRP"]


@dataclass
class WhaleFeatures:
    """Features extracted from whale trading patterns."""

    asset: CryptoSymbol
    timestamp: datetime

    # Whale consensus
    bullish_volume: float  # Total $ volume on bullish bets
    bearish_volume: float  # Total $ volume on bearish bets
    bullish_count: int  # Number of bullish trades
    bearish_count: int  # Number of bearish trades
    consensus_pct: float  # % agreement among whales

    # Timing
    hour: int
    day_of_week: int
    minutes_to_resolution: float

    # Trade characteristics
    avg_trade_size: float
    max_trade_size: float
    price_paid: float  # Average price paid for Yes tokens

    # Derived
    whale_direction: str  # "UP" or "DOWN"
    whale_confidence: float  # 0-1


@dataclass
class TrainingExample:
    """A single training example for the whale ML model."""

    features: WhaleFeatures
    actual_outcome: str  # "UP" or "DOWN" (what actually happened)
    was_correct: bool  # Did whales predict correctly?


class WhalePatternTrainer:
    """
    Trains ML models to predict based on whale behavior patterns.

    Two model types:
    1. Direction model: Predict UP/DOWN based on whale consensus
    2. Quality model: Predict whether whale consensus is reliable
    """

    def __init__(self, data_dir: Path | str = "./trader_data"):
        self.data_dir = Path(data_dir)

        # Models
        self.direction_model: Optional[HistGradientBoostingClassifier] = None
        self.quality_model: Optional[HistGradientBoostingClassifier] = None

        # Training data
        self.examples: list[TrainingExample] = []

        # Feature stats for normalization
        self._volume_mean: float = 0
        self._volume_std: float = 1

    def load_trade_data(self, filepath: Optional[Path] = None) -> list[dict]:
        """Load trade data from JSON file."""
        if filepath is None:
            filepath = self.data_dir / "latest_trades.json"

        if not filepath.exists():
            logger.warning(f"No trade data found at {filepath}")
            return []

        with open(filepath) as f:
            return json.load(f)

    def extract_features_from_trades(
        self,
        trades: list[dict],
        window_minutes: int = 15,
    ) -> list[WhaleFeatures]:
        """
        Extract whale features from raw trade data.

        Groups trades by asset and time window, calculates aggregate features.
        """
        # Group by asset and time window
        grouped: dict[tuple[str, int], list[dict]] = {}

        for trade in trades:
            asset = trade.get("asset_ticker")
            if not asset:
                continue

            ts_str = trade.get("timestamp")
            if not ts_str:
                continue

            try:
                ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except:
                continue

            # Create window key (asset + 15-min window)
            window_key = int(ts.timestamp() // (window_minutes * 60))
            key = (asset, window_key)

            if key not in grouped:
                grouped[key] = []
            grouped[key].append(trade)

        # Extract features for each group
        features = []
        for (asset, window), group_trades in grouped.items():
            if len(group_trades) < 2:
                continue

            bullish_vol = 0.0
            bearish_vol = 0.0
            bullish_count = 0
            bearish_count = 0
            prices = []
            sizes = []

            for t in group_trades:
                size = t.get("size", 0)
                price = t.get("price", 0.5)
                outcome = t.get("outcome", "").lower()
                side = t.get("side", "buy").lower()

                sizes.append(size)
                prices.append(price)

                # Determine if bullish or bearish
                is_bullish = (outcome == "yes" and side == "buy") or (
                    outcome == "no" and side == "sell"
                )

                if is_bullish:
                    bullish_vol += size
                    bullish_count += 1
                else:
                    bearish_vol += size
                    bearish_count += 1

            total_vol = bullish_vol + bearish_vol
            if total_vol == 0:
                continue

            # Determine consensus
            bullish_pct = bullish_vol / total_vol
            whale_direction = "UP" if bullish_pct > 0.5 else "DOWN"
            whale_confidence = abs(bullish_pct - 0.5) * 2

            # Get timestamp from first trade
            first_ts = datetime.fromisoformat(
                group_trades[0]["timestamp"].replace("Z", "+00:00")
            )

            feat = WhaleFeatures(
                asset=asset,
                timestamp=first_ts,
                bullish_volume=bullish_vol,
                bearish_volume=bearish_vol,
                bullish_count=bullish_count,
                bearish_count=bearish_count,
                consensus_pct=max(bullish_pct, 1 - bullish_pct),
                hour=first_ts.hour,
                day_of_week=first_ts.weekday(),
                minutes_to_resolution=15.0,  # Placeholder
                avg_trade_size=np.mean(sizes) if sizes else 0,
                max_trade_size=max(sizes) if sizes else 0,
                price_paid=np.mean(prices) if prices else 0.5,
                whale_direction=whale_direction,
                whale_confidence=whale_confidence,
            )
            features.append(feat)

        return features

    def features_to_vector(self, feat: WhaleFeatures) -> np.ndarray:
        """Convert WhaleFeatures to feature vector for ML."""
        total_vol = feat.bullish_volume + feat.bearish_volume

        return np.array(
            [
                # Volume features
                feat.bullish_volume / (total_vol + 1),
                feat.bearish_volume / (total_vol + 1),
                np.log1p(total_vol),
                # Count features
                feat.bullish_count / (feat.bullish_count + feat.bearish_count + 1),
                feat.bullish_count + feat.bearish_count,
                # Consensus
                feat.consensus_pct,
                feat.whale_confidence,
                # Timing
                feat.hour / 24,
                feat.day_of_week / 7,
                # Trade characteristics
                np.log1p(feat.avg_trade_size),
                np.log1p(feat.max_trade_size),
                feat.price_paid,
                # Direction encoding
                1.0 if feat.whale_direction == "UP" else 0.0,
            ]
        )

    def create_training_examples(
        self,
        features: list[WhaleFeatures],
        outcomes: Optional[dict[tuple[str, datetime], str]] = None,
    ) -> list[TrainingExample]:
        """
        Create training examples from features.

        If outcomes is None, uses simulated outcomes based on whale consensus
        (for demo purposes - real training needs actual market outcomes).
        """
        examples = []

        for feat in features:
            if outcomes:
                key = (feat.asset, feat.timestamp)
                actual = outcomes.get(key)
                if not actual:
                    continue
            else:
                # Simulate: whales are right ~55% of the time
                confidence_bonus = feat.whale_confidence * 0.1
                correct_prob = 0.55 + confidence_bonus
                actual = (
                    feat.whale_direction
                    if np.random.random() < correct_prob
                    else ("DOWN" if feat.whale_direction == "UP" else "UP")
                )

            was_correct = actual == feat.whale_direction

            examples.append(
                TrainingExample(
                    features=feat,
                    actual_outcome=actual,
                    was_correct=was_correct,
                )
            )

        return examples

    def train_direction_model(
        self,
        examples: Optional[list[TrainingExample]] = None,
    ) -> dict:
        """
        Train the direction prediction model.

        Predicts UP/DOWN based on whale features.
        """
        examples = examples or self.examples
        if len(examples) < 10:
            logger.warning(f"Not enough examples ({len(examples)}) for training")
            return {"error": "Not enough data"}

        # Create feature matrix and labels
        X = np.array([self.features_to_vector(e.features) for e in examples])
        y = np.array([1 if e.actual_outcome == "UP" else 0 for e in examples])

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train
        self.direction_model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
        )
        self.direction_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.direction_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Direction model trained: {accuracy:.1%} accuracy")

        return {
            "accuracy": accuracy,
            "train_size": len(X_train),
            "test_size": len(X_test),
            "report": classification_report(y_test, y_pred, output_dict=True),
        }

    def train_quality_model(
        self,
        examples: Optional[list[TrainingExample]] = None,
    ) -> dict:
        """
        Train the quality prediction model.

        Predicts whether whale consensus will be correct.
        """
        examples = examples or self.examples
        if len(examples) < 10:
            logger.warning(f"Not enough examples for training")
            return {"error": "Not enough data"}

        # Create feature matrix and labels
        X = np.array([self.features_to_vector(e.features) for e in examples])
        y = np.array([1 if e.was_correct else 0 for e in examples])

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train
        self.quality_model = HistGradientBoostingClassifier(
            max_iter=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42,
        )
        self.quality_model.fit(X_train, y_train)

        # Evaluate
        y_pred = self.quality_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        logger.info(f"Quality model trained: {accuracy:.1%} accuracy")

        return {
            "accuracy": accuracy,
            "whale_correct_rate": np.mean(y),
            "train_size": len(X_train),
            "test_size": len(X_test),
        }

    def predict(
        self,
        features: WhaleFeatures,
    ) -> dict:
        """
        Make prediction using trained models.

        Returns:
            dict with direction, confidence, quality_score
        """
        if self.direction_model is None:
            return {"error": "Model not trained"}

        X = self.features_to_vector(features).reshape(1, -1)

        # Direction prediction
        direction_proba = self.direction_model.predict_proba(X)[0]
        direction = "UP" if direction_proba[1] > 0.5 else "DOWN"
        direction_confidence = max(direction_proba)

        # Quality prediction (if available)
        quality_score = 0.5
        if self.quality_model is not None:
            quality_proba = self.quality_model.predict_proba(X)[0]
            quality_score = quality_proba[1]  # Probability of being correct

        return {
            "direction": direction,
            "confidence": direction_confidence,
            "quality_score": quality_score,
            "whale_consensus": features.whale_direction,
            "whale_confidence": features.whale_confidence,
            "agreement": direction == features.whale_direction,
        }

    def save_models(self, filepath: Optional[Path] = None):
        """Save trained models to disk."""
        import pickle

        filepath = filepath or self.data_dir / "whale_models.pkl"
        with open(filepath, "wb") as f:
            pickle.dump(
                {
                    "direction_model": self.direction_model,
                    "quality_model": self.quality_model,
                    "volume_mean": self._volume_mean,
                    "volume_std": self._volume_std,
                },
                f,
            )
        logger.info(f"Models saved to {filepath}")

    def load_models(self, filepath: Optional[Path] = None):
        """Load trained models from disk."""
        import pickle

        filepath = filepath or self.data_dir / "whale_models.pkl"
        if not filepath.exists():
            logger.warning(f"No models found at {filepath}")
            return False

        with open(filepath, "rb") as f:
            data = pickle.load(f)
            self.direction_model = data.get("direction_model")
            self.quality_model = data.get("quality_model")
            self._volume_mean = data.get("volume_mean", 0)
            self._volume_std = data.get("volume_std", 1)

        logger.info(f"Models loaded from {filepath}")
        return True


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────


def demo():
    """Demo the whale pattern trainer."""
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("WHALE PATTERN ML TRAINER")
    print("=" * 60)

    trainer = WhalePatternTrainer()

    # Load trade data
    trades = trainer.load_trade_data()
    print(f"\nLoaded {len(trades)} trades from disk")

    if not trades:
        print("Generating synthetic training data for demo...")
        # Generate synthetic data
        import random

        trades = []
        for i in range(500):
            asset = random.choice(["BTC", "ETH", "SOL", "XRP"])
            ts = datetime.now(timezone.utc) - timedelta(hours=random.randint(0, 72))
            trades.append(
                {
                    "asset_ticker": asset,
                    "timestamp": ts.isoformat(),
                    "size": random.uniform(10, 1000),
                    "price": random.uniform(0.3, 0.7),
                    "outcome": random.choice(["Yes", "No"]),
                    "side": random.choice(["buy", "sell"]),
                }
            )

    # Extract features
    features = trainer.extract_features_from_trades(trades)
    print(f"Extracted {len(features)} feature sets")

    # Create training examples
    examples = trainer.create_training_examples(features)
    trainer.examples = examples
    print(f"Created {len(examples)} training examples")

    # Train models
    print("\n" + "-" * 60)
    print("TRAINING DIRECTION MODEL")
    print("-" * 60)
    direction_results = trainer.train_direction_model()
    print(f"Accuracy: {direction_results.get('accuracy', 0):.1%}")

    print("\n" + "-" * 60)
    print("TRAINING QUALITY MODEL")
    print("-" * 60)
    quality_results = trainer.train_quality_model()
    print(f"Accuracy: {quality_results.get('accuracy', 0):.1%}")
    print(f"Whale correct rate: {quality_results.get('whale_correct_rate', 0):.1%}")

    # Test prediction
    if features:
        print("\n" + "-" * 60)
        print("SAMPLE PREDICTION")
        print("-" * 60)
        pred = trainer.predict(features[0])
        print(f"Direction: {pred.get('direction')} ({pred.get('confidence', 0):.1%})")
        print(f"Quality: {pred.get('quality_score', 0):.1%}")
        print(f"Whale says: {pred.get('whale_consensus')}")
        print(f"Agreement: {pred.get('agreement')}")

    print("\n✅ Whale Pattern Trainer working!")


if __name__ == "__main__":
    demo()
