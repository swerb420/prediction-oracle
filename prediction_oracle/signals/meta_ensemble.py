"""Meta-learning ensemble to fuse model and heuristic signals."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List

from ..strategies import TradeDecision


@dataclass
class MetaSignal:
    source: str
    probability: float
    weight: float


class MetaEnsemble:
    """Stack lightweight learners to stabilize probability estimates."""

    def __init__(self, min_weight: float = 0.05):
        self.min_weight = min_weight

    def fuse(self, decisions: Iterable[TradeDecision], regime_label: str | None = None) -> list[TradeDecision]:
        fused: list[TradeDecision] = []
        for decision in decisions:
            signals = self._build_signals(decision, regime_label)
            prob = self._weighted_blend(signals)
            edge = prob - decision.implied_p
            fused.append(
                decision.copy(
                    update={
                        "p_true": prob,
                        "edge": edge,
                        "confidence": min(1.0, sum(s.weight for s in signals)),
                        "inter_model_disagreement": self._disagreement(signals),
                        "models_used": list(set(decision.models_used + [s.source for s in signals])),
                    }
                )
            )
        return fused

    def _build_signals(self, decision: TradeDecision, regime_label: str | None) -> List[MetaSignal]:
        signals = [
            MetaSignal(source="llm", probability=decision.p_true, weight=0.5),
            MetaSignal(source="market", probability=decision.implied_p, weight=0.2),
        ]
        if decision.edge > 0:
            signals.append(MetaSignal(source="edge_boost", probability=min(1.0, decision.p_true + decision.edge), weight=0.15))
        if regime_label in {"stress", "high_vol"}:
            signals.append(MetaSignal(source="regime_penalty", probability=decision.p_true * 0.9, weight=0.15))
        else:
            signals.append(MetaSignal(source="regime", probability=decision.p_true, weight=0.15))
        return [s for s in signals if s.weight >= self.min_weight]

    def _weighted_blend(self, signals: List[MetaSignal]) -> float:
        weight_sum = sum(s.weight for s in signals)
        if weight_sum == 0:
            return 0.0
        return sum(s.weight * s.probability for s in signals) / weight_sum

    def _disagreement(self, signals: List[MetaSignal]) -> float:
        probabilities = [s.probability for s in signals]
        if len(probabilities) < 2:
            return 0.0
        mean = sum(probabilities) / len(probabilities)
        variance = sum((p - mean) ** 2 for p in probabilities) / len(probabilities)
        return math.sqrt(variance)

