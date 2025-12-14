"""
Expected value modeling utilities for binary markets and hedge combinations.

This module provides helpers to:
- Prepare feature matrices from recent markets and resolved outcomes.
- Split training and test sets along a time axis.
- Train a simple logistic model and evaluate Brier score plus profit metrics.
- Estimate EV for hedge combinations and gate trades behind EV/liquidity thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable, List, Sequence, Tuple

import numpy as np


@dataclass
class MarketSample:
    """Simple container for an individual market observation."""

    market_id: str
    timestamp: datetime
    implied_prob: float
    liquidity: float
    price: float
    outcome: int


@dataclass
class HedgeLeg:
    """A leg in a hedge combination."""

    market_id: str
    predicted_prob: float
    price: float
    payout: float = 1.0
    liquidity: float = 0.0

    def expected_value(self) -> float:
        return self.predicted_prob * self.payout - self.price


def _as_datetime(raw: datetime | str | float) -> datetime:
    if isinstance(raw, datetime):
        return raw
    if isinstance(raw, (int, float)):
        return datetime.fromtimestamp(raw)
    return datetime.fromisoformat(raw)


def prepare_feature_matrix(
    markets: Iterable[dict],
    outcomes: dict,
    *,
    default_liquidity: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a feature matrix from recent market snapshots and known outcomes.

    Parameters
    ----------
    markets: iterable of mapping
        Each item should at minimum include an ``id`` and ``timestamp``. Optional
        numeric keys that are used when present: ``implied_prob``, ``price``,
        ``liquidity``, ``recent_prices`` (list/tuple of floats), and ``opened_at``.
    outcomes: mapping
        Mapping from market id to resolved binary outcome (1 for yes, 0 for no).
    default_liquidity: float, default 0.0
        Fallback liquidity when the field is missing.

    Returns
    -------
    X: np.ndarray
        Feature matrix.
    y: np.ndarray
        Binary labels matching ``outcomes``.
    timestamps: np.ndarray
        Timestamps aligned with rows of ``X``/``y`` for time-based splits.
    """

    feature_rows: List[List[float]] = []
    labels: List[int] = []
    timestamps: List[datetime] = []

    for market in markets:
        market_id = market.get("id") or market.get("market_id")
        if market_id not in outcomes:
            continue

        resolved = outcomes[market_id]
        if resolved not in (0, 1):
            continue

        ts = _as_datetime(market.get("timestamp", datetime.utcnow()))
        opened_at = market.get("opened_at")
        opened_at_dt = _as_datetime(opened_at) if opened_at else ts

        implied_prob = float(market.get("implied_prob", market.get("price", 0.5)))
        price = float(market.get("price", implied_prob))
        liquidity = float(market.get("liquidity", default_liquidity))

        recent_prices = market.get("recent_prices") or []
        if recent_prices:
            momentum = recent_prices[-1] - recent_prices[0]
            volatility = float(np.std(recent_prices))
        else:
            momentum = 0.0
            volatility = 0.0

        age_hours = max((ts - opened_at_dt).total_seconds() / 3600, 0)

        feature_rows.append(
            [
                implied_prob,
                price,
                liquidity,
                momentum,
                volatility,
                age_hours,
            ]
        )
        labels.append(int(resolved))
        timestamps.append(ts)

    return np.array(feature_rows, dtype=float), np.array(labels, dtype=int), np.array(timestamps)


def time_based_split(
    X: np.ndarray,
    y: np.ndarray,
    timestamps: Sequence[datetime],
    *,
    test_fraction: float = 0.2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split arrays into train/test sets using chronological order."""

    if len(X) == 0:
        return X, X, y, y

    order = np.argsort(np.array(timestamps))
    cutoff = int(len(order) * (1 - test_fraction))
    train_idx, test_idx = order[:cutoff], order[cutoff:]

    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


def train_logistic_model(
    X: np.ndarray, y: np.ndarray, *, lr: float = 0.05, epochs: int = 2000
) -> Tuple[np.ndarray, float]:
    """Train a simple logistic regression via gradient descent."""

    if X.size == 0:
        return np.zeros(X.shape[1]), 0.0

    weights = np.zeros(X.shape[1])
    bias = 0.0
    n = len(X)

    for _ in range(epochs):
        logits = X @ weights + bias
        preds = _sigmoid(logits)
        error = preds - y
        weights -= lr * (X.T @ error) / n
        bias -= lr * error.mean()

    return weights, bias


def predict_probabilities(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
    if X.size == 0:
        return np.array([])
    return _sigmoid(X @ weights + bias)


def brier_score(predictions: np.ndarray, labels: np.ndarray) -> float:
    if len(predictions) == 0:
        return float("nan")
    return float(np.mean((predictions - labels) ** 2))


def profit_metrics(predictions: np.ndarray, labels: np.ndarray, prices: np.ndarray) -> dict:
    """Compute simple profit-based metrics for binary yes contracts."""

    if len(predictions) == 0:
        return {"predicted_ev": float("nan"), "realized_profit": float("nan")}

    predicted_ev = float(np.mean(predictions - prices))
    realized_profit = float(np.mean(labels - prices))
    hit_rate = float(np.mean((predictions >= 0.5) == labels))

    return {
        "predicted_ev": predicted_ev,
        "realized_profit": realized_profit,
        "hit_rate": hit_rate,
    }


def estimate_combo_ev(legs: Sequence[HedgeLeg]) -> dict:
    """Aggregate expected value and liquidity across a hedge combination."""

    if not legs:
        return {"combo_ev": 0.0, "min_liquidity": 0.0, "worst_case_loss": 0.0}

    combo_ev = sum(leg.expected_value() for leg in legs)
    min_liquidity = min((leg.liquidity for leg in legs), default=0.0)
    total_spend = sum(leg.price for leg in legs)

    # In a binary hedge, at least one leg should pay out.
    max_payout = sum(leg.payout for leg in legs)
    worst_case_loss = total_spend - max_payout

    return {
        "combo_ev": combo_ev,
        "min_liquidity": min_liquidity,
        "worst_case_loss": worst_case_loss,
    }


def should_execute_combo(
    legs: Sequence[HedgeLeg], *, min_ev: float = 0.02, min_liquidity: float = 500.0
) -> bool:
    """Gate live trades behind both expected value and liquidity thresholds."""

    combo_stats = estimate_combo_ev(legs)
    return combo_stats["combo_ev"] >= min_ev and combo_stats["min_liquidity"] >= min_liquidity


def evaluate_pipeline(
    markets: Iterable[dict],
    outcomes: dict,
    *,
    test_fraction: float = 0.2,
    lr: float = 0.05,
    epochs: int = 2000,
) -> dict:
    """
    End-to-end training and evaluation helper for quick experiments.

    Returns dictionaries with Brier score, profit metrics, and a reference to the
    learned weights for later EV estimation.
    """

    X, y, timestamps = prepare_feature_matrix(markets, outcomes)
    X_train, X_test, y_train, y_test = time_based_split(X, y, timestamps, test_fraction=test_fraction)

    weights, bias = train_logistic_model(X_train, y_train, lr=lr, epochs=epochs)
    test_probs = predict_probabilities(X_test, weights, bias)

    metrics = {
        "brier": brier_score(test_probs, y_test),
        "weights": weights,
        "bias": bias,
    }

    if len(test_probs) == len(y_test):
        # Use the mid-price as the trading cost proxy.
        prices = X_test[:, 1] if X_test.size else np.array([])
        metrics.update(profit_metrics(test_probs, y_test, prices))

    return metrics
