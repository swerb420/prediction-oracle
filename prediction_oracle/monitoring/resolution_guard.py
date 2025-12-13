"""Outcome resolution guardrails to avoid bad fills."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Iterable, List

from ..markets import Market


class ResolutionGuard:
    def __init__(self, freshness_minutes: int = 5):
        self.freshness = timedelta(minutes=freshness_minutes)

    def filter_markets(self, markets: Iterable[Market]) -> List[Market]:
        guarded: List[Market] = []
        now = datetime.utcnow()
        for market in markets:
            if market.close_time and market.close_time < now:
                continue
            if market.close_time and market.close_time - now < self.freshness:
                continue
            if "ambiguous" in (market.rules or "").lower():
                continue
            guarded.append(market)
        return guarded

