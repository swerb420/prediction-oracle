"""Simplified microstructure feature extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

from ..markets import Market


@dataclass
class MicrostructureSnapshot:
    imbalance: float
    depth: float
    sweep_risk: float


class MicrostructureAnalyzer:
    """Derive quick microstructure heuristics from market data."""

    def __init__(self, default_depth: float = 1_000.0):
        self.default_depth = default_depth

    def analyze(self, markets: Iterable[Market]) -> Dict[str, MicrostructureSnapshot]:
        snapshots: Dict[str, MicrostructureSnapshot] = {}
        for market in markets:
            bids = [o.price for o in market.outcomes if o.price <= 0.5]
            asks = [o.price for o in market.outcomes if o.price > 0.5]
            bid_depth = sum(getattr(o, "volume_24h", 0.0) or 0.0 for o in market.outcomes if o.price <= 0.5)
            ask_depth = sum(getattr(o, "volume_24h", 0.0) or 0.0 for o in market.outcomes if o.price > 0.5)
            total_depth = bid_depth + ask_depth or self.default_depth

            imbalance = 0.0
            if bids or asks:
                imbalance = (sum(bids) - sum(asks)) / max(len(bids + asks), 1)

            sweep_risk = 1 - min(1.0, total_depth / (self.default_depth * 2))
            snapshots[market.market_id] = MicrostructureSnapshot(
                imbalance=imbalance,
                depth=total_depth,
                sweep_risk=sweep_risk,
            )
        return snapshots

