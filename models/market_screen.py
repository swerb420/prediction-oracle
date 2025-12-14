"""Market screening utilities for spreads, tail odds, liquidity, and hedges."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import mean
from typing import Iterable, List, Sequence


@dataclass
class OutcomeQuote:
    """Quote information for a single market outcome."""

    name: str
    bid: float
    ask: float
    liquidity: float

    def spread_width(self) -> float:
        """Return the bid/ask spread width."""

        return max(0.0, self.ask - self.bid)

    def mid_price(self) -> float:
        """Return the midpoint probability."""

        return (self.ask + self.bid) / 2


@dataclass
class HedgeScenario:
    """Parameters describing a two-sided hedge position."""

    payoff_up: float
    payoff_down: float
    fees: float = 0.0
    slippage: float = 0.0


@dataclass
class MarketQuote:
    """Market-level context used for screening and hedging simulations."""

    market_id: str
    question: str
    outcomes: List[OutcomeQuote]
    prior_probability: float = 0.5
    alignment_bias: float = 0.0
    hedge: HedgeScenario | None = None


@dataclass
class MarketScreenResult:
    """Computed ranking and hedge diagnostics for a market."""

    market_id: str
    question: str
    average_spread: float
    tail_odds: float
    total_liquidity: float
    ranking_score: float
    hedge_ev: float | None = None
    hedge_positive: bool = False
    alignment_boosted: bool = False
    adjusted_probability: float = 0.5
    notes: list[str] = field(default_factory=list)


def _clamp_probability(value: float) -> float:
    return max(0.0, min(1.0, value))


def _tail_odds(outcomes: Iterable[OutcomeQuote]) -> float:
    """Return the smallest tail probability across outcomes."""

    tail_values = [min(quote.mid_price(), 1 - quote.mid_price()) for quote in outcomes]
    if not tail_values:
        raise ValueError("Cannot compute tail odds without outcomes")
    return min(tail_values)


def simulate_two_sided_hedge(
    probability_up: float,
    payoff_up: float,
    payoff_down: float,
    fees: float,
    slippage: float,
) -> float:
    """Simulate the expected value of a two-sided X/X hedge."""

    probability_up = _clamp_probability(probability_up)
    return (probability_up * payoff_up) + ((1 - probability_up) * payoff_down) - fees - slippage


def _ranking_score(average_spread: float, tail_odds: float, total_liquidity: float) -> float:
    """Combine spread, tail odds, and liquidity into a sortable score."""

    spread_score = 1 / (1 + average_spread)
    tail_score = 1 - tail_odds
    liquidity_score = math.log1p(total_liquidity) / 10
    return spread_score + tail_score + liquidity_score


def _adjust_probability(prior: float, alignment_bias: float) -> tuple[float, bool]:
    """Adjust a probability using a trader alignment bias."""

    adjusted = _clamp_probability(prior + (alignment_bias * 0.1))
    return adjusted, not math.isclose(adjusted, prior)


def rank_markets(markets: Sequence[MarketQuote]) -> list[MarketScreenResult]:
    """Rank markets by spread width, tail odds, and liquidity."""

    results = screen_markets(markets)
    return sorted(results, key=lambda result: result.ranking_score, reverse=True)


def screen_markets(markets: Sequence[MarketQuote]) -> list[MarketScreenResult]:
    """Compute ranking metrics and hedge diagnostics for markets."""

    ranked: list[MarketScreenResult] = []
    for market in markets:
        if not market.outcomes:
            raise ValueError(f"Market {market.market_id} has no outcomes")

        spreads = [quote.spread_width() for quote in market.outcomes]
        average_spread = mean(spreads)
        tail_odds = _tail_odds(market.outcomes)
        total_liquidity = sum(quote.liquidity for quote in market.outcomes)

        adjusted_probability, alignment_boosted = _adjust_probability(
            market.prior_probability, market.alignment_bias
        )

        hedge_ev = None
        hedge_positive = False
        notes: list[str] = []

        if market.hedge:
            hedge_ev = simulate_two_sided_hedge(
                adjusted_probability,
                market.hedge.payoff_up,
                market.hedge.payoff_down,
                market.hedge.fees,
                market.hedge.slippage,
            )
            hedge_positive = hedge_ev > 0
            if hedge_positive:
                notes.append("Positive hedge EV")

        if alignment_boosted:
            notes.append("Alignment boosted priors")

        ranking_score = _ranking_score(average_spread, tail_odds, total_liquidity)

        ranked.append(
            MarketScreenResult(
                market_id=market.market_id,
                question=market.question,
                average_spread=average_spread,
                tail_odds=tail_odds,
                total_liquidity=total_liquidity,
                ranking_score=ranking_score,
                hedge_ev=hedge_ev,
                hedge_positive=hedge_positive,
                alignment_boosted=alignment_boosted,
                adjusted_probability=adjusted_probability,
                notes=notes,
            )
        )

    return ranked
