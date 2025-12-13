"""Strategy implementing serious alpha edge heuristics."""

from __future__ import annotations

import itertools
from typing import Iterable, List

from .base_strategy import BaseStrategy, TradeDecision
from ..markets import Market, Venue


class AlphaEdgesStrategy(BaseStrategy):
    def __init__(self, config: dict, regime_label: str = "normal"):
        super().__init__("alpha_edges", config)
        self.regime_label = regime_label

    async def select_markets(self, all_markets: list[Market]) -> list[Market]:
        return [m for m in all_markets if m.volume_24h and m.volume_24h > 1_000]

    async def evaluate(self, markets: list[Market], oracle_results: dict | None) -> list[TradeDecision]:
        decisions: List[TradeDecision] = []
        dislocation_threshold = self.config.get("dislocation_threshold", 0.05)

        grouped = self._group_by_question(markets)
        for _, venue_markets in grouped.items():
            if len(venue_markets) < 2:
                continue
            for m1, m2 in itertools.permutations(venue_markets, 2):
                if not m1.outcomes or not m2.outcomes:
                    continue
                avg1 = sum(o.price for o in m1.outcomes) / len(m1.outcomes)
                avg2 = sum(o.price for o in m2.outcomes) / len(m2.outcomes)
                if avg1 - avg2 > dislocation_threshold:
                    top_outcome = max(m1.outcomes, key=lambda o: o.price)
                    decisions.append(
                        self._build_decision(
                            cheap_market=m2,
                            rich_market=m1,
                            cheap_outcome=top_outcome,
                            premium=avg1 - avg2,
                            rationale="cross-venue dislocation harvest",
                        )
                    )

        for market in markets:
            if market.volume_24h and market.volume_24h > 5_000:
                best = max(market.outcomes, key=lambda o: o.price)
                decisions.append(
                    self._build_decision(
                        cheap_market=market,
                        rich_market=market,
                        cheap_outcome=best,
                        premium=0.02,
                        rationale="event-based scalping / queue control",
                    )
                )
        return decisions

    def _group_by_question(self, markets: Iterable[Market]):
        grouped: dict[str, list[Market]] = {}
        for market in markets:
            grouped.setdefault(market.question.lower(), []).append(market)
        return grouped

    def _build_decision(
        self,
        cheap_market: Market,
        rich_market: Market,
        cheap_outcome,
        premium: float,
        rationale: str,
    ) -> TradeDecision:
        direction = "BUY"
        implied_p = cheap_outcome.price
        p_true = min(0.99, implied_p + premium)
        return TradeDecision(
            venue=cheap_market.venue,
            market_id=cheap_market.market_id,
            outcome_id=cheap_outcome.id,
            direction=direction,
            size_usd=self.config.get("notional", 100.0),
            p_true=p_true,
            implied_p=implied_p,
            edge=p_true - implied_p,
            confidence=0.6,
            inter_model_disagreement=0.05,
            rule_risks=[],
            strategy_name=self.name,
            rationale=rationale,
            models_used=["alpha_edges"],
        )

