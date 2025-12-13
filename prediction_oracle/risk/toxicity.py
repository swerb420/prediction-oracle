"""Toxicity-aware trade filter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

from ..strategies import TradeDecision


@dataclass
class ToxicityScore:
    venue: str
    score: float


class ToxicityModel:
    def __init__(self, venue_penalties: dict[str, float] | None = None):
        self.venue_penalties = venue_penalties or {}

    def adjust(self, decisions: Iterable[TradeDecision]) -> list[TradeDecision]:
        adjusted: List[TradeDecision] = []
        for decision in decisions:
            venue_penalty = self.venue_penalties.get(decision.venue.value, 0.0)
            toxic_penalty = max(0.0, min(0.25, venue_penalty))
            adjusted_prob = max(0.0, decision.p_true - toxic_penalty)
            adjusted.append(
                decision.copy(
                    update={
                        "p_true": adjusted_prob,
                        "edge": adjusted_prob - decision.implied_p,
                        "rationale": f"{decision.rationale} | toxicity_penalty={toxic_penalty:.3f}",
                    }
                )
            )
        return adjusted

