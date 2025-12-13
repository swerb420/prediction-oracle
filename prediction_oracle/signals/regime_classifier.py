"""Regime classifier for routing strategy behavior."""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, Iterable, List

from ..markets import Market
from ..storage import MarketSnapshot, get_session


@dataclass
class RegimeContext:
    label: str
    volatility: float
    liquidity: float
    crowd_skew: float
    timestamp: datetime

    def risk_multiplier(self) -> float:
        """Return a multiplier for sizing based on the regime."""
        if self.label == "stress":
            return 0.35
        if self.label == "high_vol":
            return 0.6
        if self.label == "illiquid":
            return 0.5
        return 1.0


class RegimeClassifier:
    """Simple regime classifier using recent market snapshots."""

    def __init__(self, lookback_minutes: int = 60, liquidity_floor: float = 500.0):
        self.lookback = timedelta(minutes=lookback_minutes)
        self.liquidity_floor = liquidity_floor
        self._latest_context: RegimeContext | None = None

    async def classify(self, markets: Iterable[Market]) -> RegimeContext:
        now = datetime.utcnow()
        volatility_values: List[float] = []
        liquidity_values: List[float] = []
        crowd_skews: List[float] = []

        async with get_session() as session:
            for market in markets:
                snapshot: MarketSnapshot | None = await session.get(
                    MarketSnapshot, getattr(market, "id", None)
                )
                if snapshot and snapshot.snapshot_time < now - self.lookback:
                    continue

                prices = [o.price for o in market.outcomes]
                if prices:
                    implied = statistics.mean(prices)
                    dispersion = statistics.pvariance(prices)
                    volatility_values.append(dispersion)
                    crowd_skews.append(abs(0.5 - implied))

                volume_fields = [getattr(o, "volume_24h", 0.0) or 0.0 for o in market.outcomes]
                if volume_fields:
                    liquidity_values.append(sum(volume_fields))

        volatility = statistics.mean(volatility_values) if volatility_values else 0.0
        liquidity = statistics.mean(liquidity_values) if liquidity_values else 0.0
        crowd_skew = statistics.mean(crowd_skews) if crowd_skews else 0.0

        label = "normal"
        if volatility > 0.02 and liquidity < self.liquidity_floor:
            label = "stress"
        elif volatility > 0.02:
            label = "high_vol"
        elif liquidity < self.liquidity_floor:
            label = "illiquid"
        elif crowd_skew > 0.2:
            label = "one_sided"

        self._latest_context = RegimeContext(
            label=label,
            volatility=volatility,
            liquidity=liquidity,
            crowd_skew=crowd_skew,
            timestamp=now,
        )
        return self._latest_context

    @property
    def latest(self) -> RegimeContext | None:
        return self._latest_context

