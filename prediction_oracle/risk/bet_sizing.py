"""Adaptive bet sizing using fractional Kelly with drawdown guards."""

from __future__ import annotations

import math
from typing import Iterable

from .bankroll import BankrollManager
from ..strategies import TradeDecision


class AdaptiveKellySizer:
    def __init__(self, bankroll: BankrollManager, max_fraction: float = 0.02):
        self.bankroll = bankroll
        self.max_fraction = max_fraction
        self.drawdown_cap = 0.25

    def resize(self, decisions: Iterable[TradeDecision], risk_multiplier: float = 1.0) -> list[TradeDecision]:
        sized: list[TradeDecision] = []
        state = self.bankroll.get_state()
        drawdown_ratio = max(0.0, (self.bankroll.initial - state.total) / self.bankroll.initial)
        drawdown_penalty = max(0.25, 1.0 - drawdown_ratio)

        for decision in decisions:
            edge = max(1e-6, decision.edge)
            p = min(0.999, max(0.001, decision.p_true))
            q = 1 - p
            kelly_fraction = (p * (1 + edge) - q) / (1 + edge)
            kelly_fraction = max(0.0, kelly_fraction)
            position_fraction = min(self.max_fraction, kelly_fraction * drawdown_penalty * risk_multiplier)
            target_size = state.total * position_fraction
            sized.append(decision.copy(update={"size_usd": target_size}))
        return sized

