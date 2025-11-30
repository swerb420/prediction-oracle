"""Longshot '$2-5 RR monster' strategy."""

import logging
from datetime import datetime, timezone

from ..markets import Market
from .base_strategy import BaseStrategy, TradeDecision

logger = logging.getLogger(__name__)


class LongshotStrategy(BaseStrategy):
    """
    Longshot strategy targeting high-upside, low-probability outcomes.
    
    Looks for outcomes priced 1-10% that LLMs believe are significantly underpriced.
    Uses small, fixed bet sizes for asymmetric risk/reward.
    """

    def __init__(self, config: dict):
        super().__init__("longshot", config)

    async def select_markets(self, all_markets: list[Market]) -> list[Market]:
        """Filter to markets with longshot outcomes."""
        price_range = self.config.get("price_range", [0.01, 0.10])
        
        selected = []
        now = datetime.now(timezone.utc)
        
        for market in all_markets:
            # Check time to close (want enough time for outcome to materialize)
            hours_to_close = (market.close_time - now).total_seconds() / 3600
            if hours_to_close < 48:  # At least 2 days
                continue
            
            # Check if any outcome is in longshot price range
            for outcome in market.outcomes:
                if price_range[0] <= outcome.price <= price_range[1]:
                    selected.append(market)
                    break  # Only add market once
        
        logger.info(
            f"Longshot strategy selected {len(selected)} / {len(all_markets)} markets"
        )
        return selected

    async def evaluate(
        self,
        markets: list[Market],
        oracle_results: dict,
    ) -> list[TradeDecision]:
        """Generate longshot trade decisions."""
        min_edge = self.config.get("min_edge", 0.10)
        max_disagreement = self.config.get("max_disagreement", 0.12)
        min_confidence = self.config.get("min_confidence", 0.60)
        fixed_bet_usd = self.config.get("fixed_bet_usd", 3.0)
        max_bet_usd = self.config.get("max_bet_usd", 5.0)
        price_range = self.config.get("price_range", [0.01, 0.10])
        
        decisions = []
        
        for market in markets:
            market_results = oracle_results.get(market.market_id, [])
            
            for result in market_results:
                # Must be a longshot price
                if not (price_range[0] <= result.implied_p <= price_range[1]):
                    continue
                
                # Check edge requirement (stricter for longshots)
                if result.edge < min_edge:
                    continue
                
                # Check confidence
                if result.avg_confidence < min_confidence:
                    logger.debug(
                        f"Skipping longshot {market.market_id}: "
                        f"confidence {result.avg_confidence:.3f} < {min_confidence}"
                    )
                    continue
                
                # Check model disagreement
                if result.inter_model_disagreement > max_disagreement:
                    logger.debug(
                        f"Skipping longshot {market.market_id}: "
                        f"disagreement {result.inter_model_disagreement:.3f}"
                    )
                    continue
                
                # No critical rule risks
                critical_keywords = ["ambiguous", "unclear", "undefined", "missing"]
                has_critical_risk = any(
                    any(kw in risk.lower() for kw in critical_keywords)
                    for risk in result.rule_risks
                )
                if has_critical_risk:
                    logger.debug(
                        f"Skipping longshot {market.market_id}: critical rule risks"
                    )
                    continue
                
                # Use fixed small bet size
                size_usd = min(fixed_bet_usd, max_bet_usd)
                
                # Always BUY for longshots (betting the unlikely event will happen)
                decision = TradeDecision(
                    venue=market.venue,
                    market_id=market.market_id,
                    outcome_id=result.outcome_id,
                    direction="BUY",
                    size_usd=size_usd,
                    p_true=result.mean_p_true,
                    implied_p=result.implied_p,
                    edge=result.edge,
                    confidence=result.avg_confidence,
                    inter_model_disagreement=result.inter_model_disagreement,
                    rule_risks=result.rule_risks,
                    strategy_name=self.name,
                    rationale=(
                        f"Longshot: {result.implied_p:.1%} → {result.mean_p_true:.1%} "
                        f"({result.edge / result.implied_p:.1f}x edge)"
                    ),
                    models_used=result.models_used,
                )
                
                decisions.append(decision)
        
        logger.info(f"Longshot strategy generated {len(decisions)} trade decisions")
        return decisions
