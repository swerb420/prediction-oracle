"""Lightweight online RL policy placeholder for execution adjustments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from ..strategies import TradeDecision


@dataclass
class ExecutionAction:
    price_offset: float
    urgency: float


class RLExecutionPolicy:
    """Simplified policy that nudges limit prices based on fill feedback."""

    def __init__(self, exploration: float = 0.05):
        self.exploration = exploration
        self.recent_fills: list[float] = []

    def act(self, decisions: Iterable[TradeDecision]) -> list[TradeDecision]:
        adjusted: list[TradeDecision] = []
        fill_score = self._fill_score()
        for decision in decisions:
            offset = (self.exploration * (1 - fill_score)) * (0.5 - decision.implied_p)
            new_price = max(0.01, min(0.99, decision.implied_p + offset))
            adjusted.append(decision.copy(update={"implied_p": new_price, "rationale": f"{decision.rationale} | rl_offset={offset:.4f}"}))
        return adjusted

    def feedback(self, fill_slippage: float) -> None:
        self.recent_fills.append(fill_slippage)
        if len(self.recent_fills) > 100:
            self.recent_fills.pop(0)

    def _fill_score(self) -> float:
        if not self.recent_fills:
            return 0.5
        avg_slip = sum(self.recent_fills) / len(self.recent_fills)
        return max(0.0, min(1.0, 1 - abs(avg_slip)))

