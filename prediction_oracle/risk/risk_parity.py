"""Risk parity allocator across categories."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from ..strategies import TradeDecision


class RiskParityAllocator:
    def __init__(self, max_category_share: float = 0.3):
        self.max_category_share = max_category_share

    def rebalance(self, decisions: Iterable[TradeDecision]) -> list[TradeDecision]:
        buckets = defaultdict(list)
        for decision in decisions:
            category = getattr(decision, "category", "uncategorized")
            buckets[category].append(decision)

        total_size = sum(d.size_usd for d in decisions) or 1.0
        balanced: list[TradeDecision] = []

        for category, items in buckets.items():
            category_cap = total_size * self.max_category_share
            per_trade = category_cap / max(len(items), 1)
            for decision in items:
                balanced.append(decision.copy(update={"size_usd": min(decision.size_usd, per_trade)}))

        return balanced

