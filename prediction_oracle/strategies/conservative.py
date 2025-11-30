"""Conservative 'edge harvester' strategy."""

import logging
from datetime import datetime, timezone

from ..markets import Market
from .base_strategy import BaseStrategy, TradeDecision

logger = logging.getLogger(__name__)


class ConservativeStrategy(BaseStrategy):
    """
    Conservative strategy targeting high-probability, low-risk edges.
    
    Focuses on liquid markets with narrow spreads and high model consensus.
    """

    def __init__(self, config: dict):
        super().__init__("conservative", config)

    async def select_markets(self, all_markets: list[Market]) -> list[Market]:
        """Filter to liquid, mid-probability markets."""
        min_liquidity = self.config.get("min_liquidity_usd", 500)
        max_spread = self.config.get("max_spread", 0.05)
        prob_range = self.config.get("implied_prob_range", [0.20, 0.75])
        min_hours_to_close = self.config.get("min_time_to_close_hours", 24)
        
        selected = []
        now = datetime.now(timezone.utc)
        
        for market in all_markets:
            # Check time to close
            hours_to_close = (market.close_time - now).total_seconds() / 3600
            if hours_to_close < min_hours_to_close:
                continue
            
            # Check each outcome
            for outcome in market.outcomes:
                # Liquidity check
                if outcome.liquidity and outcome.liquidity < min_liquidity:
                    continue
                
                # Probability range check
                if outcome.price < prob_range[0] or outcome.price > prob_range[1]:
                    continue
                
                # Spread check (if we have volume data)
                # For now, accept if liquidity exists
                selected.append(market)
                break  # Only add market once
        
        logger.info(
            f"Conservative strategy selected {len(selected)} / {len(all_markets)} markets"
        )
        return selected

    async def evaluate(
        self,
        markets: list[Market],
        oracle_results: dict,
    ) -> list[TradeDecision]:
        """Generate conservative trade decisions."""
        min_edge = self.config.get("min_edge", 0.04)
        max_disagreement = self.config.get("max_disagreement", 0.08)
        max_rule_risk_count = self.config.get("max_rule_risk_count", 0)
        position_size_pct = self.config.get("position_size_pct", 0.01)
        kelly_fraction = self.config.get("kelly_fraction", 0.25)
        
        decisions = []
        
        for market in markets:
            market_results = oracle_results.get(market.market_id, [])
            
            for result in market_results:
                # Check edge requirement
                if abs(result.edge) < min_edge:
                    continue
                
                # Check model disagreement
                if result.inter_model_disagreement > max_disagreement:
                    logger.debug(
                        f"Skipping {market.market_id}: disagreement "
                        f"{result.inter_model_disagreement:.3f} > {max_disagreement}"
                    )
                    continue
                
                # Check rule risks
                if len(result.rule_risks) > max_rule_risk_count:
                    logger.debug(
                        f"Skipping {market.market_id}: {len(result.rule_risks)} rule risks"
                    )
                    continue
                
                # Determine direction
                direction = "BUY" if result.edge > 0 else "SELL"
                
                # Calculate position size using fractional Kelly
                # Kelly = edge / odds
                # For now, use fixed percentage
                size_usd = 100.0 * position_size_pct  # Assuming $100 bankroll for now
                
                decision = TradeDecision(
                    venue=market.venue,
                    market_id=market.market_id,
                    outcome_id=result.outcome_id,
                    direction=direction,
                    size_usd=size_usd,
                    p_true=result.mean_p_true,
                    implied_p=result.implied_p,
                    edge=result.edge,
                    confidence=result.avg_confidence,
                    inter_model_disagreement=result.inter_model_disagreement,
                    rule_risks=result.rule_risks,
                    strategy_name=self.name,
                    rationale=f"Edge {result.edge:+.3f}, disagreement {result.inter_model_disagreement:.3f}",
                    models_used=result.models_used,
                )
                
                decisions.append(decision)
        
        logger.info(f"Conservative strategy generated {len(decisions)} trade decisions")
        return decisions
