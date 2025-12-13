"""
Cross-market arbitrage policy.
Exploits price discrepancies between related markets.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from core.logging_utils import get_logger

logger = get_logger(__name__)


class ArbLeg(BaseModel):
    """One leg of an arbitrage trade."""
    
    market_id: str
    question: str
    outcome: str  # "yes" or "no"
    action: str  # "buy" or "sell"
    price: float = Field(ge=0.0, le=1.0)
    size_usd: float = 0.0


class ArbSignal(BaseModel):
    """Arbitrage trading signal."""
    
    timestamp: datetime
    arb_type: str  # "sum_arb", "semantic_arb", "cross_market"
    
    # Legs
    legs: list[ArbLeg] = Field(default_factory=list)
    
    # Expected profit
    gross_profit_pct: float = 0.0  # Before costs
    net_profit_pct: float = 0.0  # After estimated costs
    net_profit_usd: float = 0.0
    
    # Confidence
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    semantic_similarity: float = Field(default=0.0, ge=0.0, le=1.0)
    
    # Risk assessment
    execution_risk: float = Field(default=0.5, ge=0.0, le=1.0)  # Risk of partial fill
    timing_risk: float = Field(default=0.5, ge=0.0, le=1.0)  # Risk of price movement
    resolution_risk: float = Field(default=0.0, ge=0.0, le=1.0)  # Risk of different resolutions
    
    # Trade sizing
    recommended_size_usd: float = 0.0
    max_size_usd: float = 0.0
    
    # Flags
    avoid_trade: bool = False
    reason: str = ""
    
    @property
    def is_tradeable(self) -> bool:
        """Check if arb is actionable."""
        return (
            not self.avoid_trade and
            self.net_profit_pct > 0.01 and  # 1% minimum
            len(self.legs) >= 2 and
            self.confidence >= 0.6
        )
    
    @property
    def risk_adjusted_return(self) -> float:
        """Risk-adjusted expected return."""
        total_risk = (self.execution_risk + self.timing_risk + self.resolution_risk) / 3
        return self.net_profit_pct * (1 - total_risk)


class ArbPolicy:
    """
    Cross-market arbitrage policy.
    
    Types of arbitrage:
    1. Sum Arbitrage: Outcome prices don't sum to 1.0
    2. Semantic Arbitrage: Similar markets with different prices
    3. Cross-Market: Related markets across crypto and predictions
    
    Key features:
    - Uses Grok semantic clustering to find related markets
    - Calculates execution costs and slippage
    - Risk-adjusted position sizing
    """
    
    def __init__(
        self,
        min_profit_pct: float = 0.01,  # 1% minimum profit
        max_position_usd: float = 500.0,
        execution_cost_pct: float = 0.005,  # 0.5% per leg
        max_slippage_pct: float = 0.02,  # 2% max slippage
        min_semantic_similarity: float = 0.9,
        max_resolution_gap_hours: float = 24.0
    ):
        self.min_profit_pct = min_profit_pct
        self.max_position_usd = max_position_usd
        self.execution_cost_pct = execution_cost_pct
        self.max_slippage_pct = max_slippage_pct
        self.min_semantic_similarity = min_semantic_similarity
        self.max_resolution_gap_hours = max_resolution_gap_hours
    
    def generate_sum_arb_signal(
        self,
        market_id: str,
        question: str,
        outcomes: list[tuple[str, float]],  # [(outcome_name, price), ...]
        liquidity: float = 10000.0
    ) -> ArbSignal | None:
        """
        Generate signal for sum arbitrage (outcomes don't sum to 1).
        
        Args:
            market_id: Market ID
            question: Market question
            outcomes: List of (outcome_name, price) tuples
            liquidity: Market liquidity in USD
            
        Returns:
            ArbSignal or None if no arb exists
        """
        from core.time_utils import now_utc
        
        if len(outcomes) < 2:
            return None
        
        # Calculate outcome sum
        outcome_sum = sum(price for _, price in outcomes)
        
        # Check for arb opportunity
        if abs(outcome_sum - 1.0) < self.min_profit_pct:
            return None  # No meaningful arb
        
        timestamp = now_utc()
        legs = []
        
        if outcome_sum < 1.0:
            # Buy all outcomes for less than 1.0
            arb_type = "sum_arb_under"
            for outcome_name, price in outcomes:
                legs.append(ArbLeg(
                    market_id=market_id,
                    question=question,
                    outcome=outcome_name,
                    action="buy",
                    price=price
                ))
            
            gross_profit = 1.0 - outcome_sum
            
        else:
            # Can sell all outcomes for more than 1.0
            # This is rare on Polymarket but possible
            arb_type = "sum_arb_over"
            for outcome_name, price in outcomes:
                legs.append(ArbLeg(
                    market_id=market_id,
                    question=question,
                    outcome=outcome_name,
                    action="sell",
                    price=price
                ))
            
            gross_profit = outcome_sum - 1.0
        
        # Calculate costs
        n_legs = len(legs)
        total_cost = n_legs * self.execution_cost_pct
        net_profit = gross_profit - total_cost
        
        # Check profitability
        if net_profit < self.min_profit_pct:
            return None
        
        # Calculate position size
        # Size limited by smallest outcome liquidity
        max_size = min(self.max_position_usd, liquidity * 0.1)
        
        # Recommended size based on confidence
        confidence = min(1.0, gross_profit / 0.05)  # Higher profit = higher confidence
        recommended_size = max_size * confidence * 0.5  # Conservative
        
        return ArbSignal(
            timestamp=timestamp,
            arb_type=arb_type,
            legs=legs,
            gross_profit_pct=gross_profit,
            net_profit_pct=net_profit,
            net_profit_usd=net_profit * recommended_size,
            confidence=confidence,
            execution_risk=0.2,  # Low for sum arb (same market)
            timing_risk=0.1,
            resolution_risk=0.0,  # Same market = same resolution
            recommended_size_usd=recommended_size,
            max_size_usd=max_size
        )
    
    def generate_semantic_arb_signal(
        self,
        market_1: dict[str, Any],
        market_2: dict[str, Any],
        semantic_similarity: float
    ) -> ArbSignal | None:
        """
        Generate signal for semantic arbitrage (similar markets, different prices).
        
        Args:
            market_1: First market dict
            market_2: Second market dict
            semantic_similarity: Similarity score from Grok
            
        Returns:
            ArbSignal or None
        """
        from core.time_utils import now_utc
        
        if semantic_similarity < self.min_semantic_similarity:
            return None
        
        price_1 = market_1.get("yes_price", 0.5)
        price_2 = market_2.get("yes_price", 0.5)
        
        price_diff = abs(price_1 - price_2)
        
        # Need meaningful price difference
        if price_diff < self.min_profit_pct * 2:
            return None
        
        timestamp = now_utc()
        
        # Determine direction: buy cheap, sell expensive
        if price_1 < price_2:
            buy_market = market_1
            sell_market = market_2
            buy_price = price_1
            sell_price = price_2
        else:
            buy_market = market_2
            sell_market = market_1
            buy_price = price_2
            sell_price = price_1
        
        legs = [
            ArbLeg(
                market_id=buy_market.get("market_id", ""),
                question=buy_market.get("question", ""),
                outcome="yes",
                action="buy",
                price=buy_price
            ),
            ArbLeg(
                market_id=sell_market.get("market_id", ""),
                question=sell_market.get("question", ""),
                outcome="yes",
                action="sell",
                price=sell_price
            )
        ]
        
        # Calculate profit
        gross_profit = sell_price - buy_price
        total_cost = 2 * self.execution_cost_pct
        net_profit = gross_profit - total_cost
        
        if net_profit < self.min_profit_pct:
            return None
        
        # Risk assessment
        # Semantic arb has resolution risk - markets might resolve differently
        resolution_risk = 1.0 - semantic_similarity
        
        # Check resolution time difference
        res_1 = market_1.get("resolution_time")
        res_2 = market_2.get("resolution_time")
        timing_risk = 0.3  # Default moderate timing risk
        
        if res_1 and res_2:
            time_diff = abs((res_1 - res_2).total_seconds() / 3600)
            if time_diff > self.max_resolution_gap_hours:
                return ArbSignal(
                    timestamp=timestamp,
                    arb_type="semantic_arb",
                    legs=legs,
                    avoid_trade=True,
                    reason=f"Resolution time gap too large ({time_diff:.1f}h)"
                )
            timing_risk = min(0.5, time_diff / 24)
        
        # Position sizing
        min_liquidity = min(
            market_1.get("liquidity", 10000),
            market_2.get("liquidity", 10000)
        )
        max_size = min(self.max_position_usd, min_liquidity * 0.05)
        
        # Confidence based on similarity and profit
        confidence = semantic_similarity * min(1.0, net_profit / 0.05)
        
        # Reduce size for high-risk arbs
        risk_factor = 1 - (resolution_risk + timing_risk) / 2
        recommended_size = max_size * confidence * risk_factor
        
        return ArbSignal(
            timestamp=timestamp,
            arb_type="semantic_arb",
            legs=legs,
            gross_profit_pct=gross_profit,
            net_profit_pct=net_profit,
            net_profit_usd=net_profit * recommended_size,
            confidence=confidence,
            semantic_similarity=semantic_similarity,
            execution_risk=0.3,  # Moderate - different orderbooks
            timing_risk=timing_risk,
            resolution_risk=resolution_risk,
            recommended_size_usd=recommended_size,
            max_size_usd=max_size
        )
    
    def scan_for_sum_arbs(
        self,
        markets: list[dict[str, Any]]
    ) -> list[ArbSignal]:
        """
        Scan markets for sum arbitrage opportunities.
        
        Args:
            markets: List of market dicts with outcomes
            
        Returns:
            List of arb signals, sorted by profit
        """
        signals = []
        
        for market in markets:
            market_id = market.get("market_id", "")
            question = market.get("question", "")
            outcomes = market.get("outcomes", [])
            liquidity = market.get("liquidity", 10000.0)
            
            if not outcomes:
                continue
            
            # Format outcomes as (name, price) tuples
            outcome_tuples = [
                (o.get("name", ""), o.get("price", 0.5))
                for o in outcomes
            ]
            
            signal = self.generate_sum_arb_signal(
                market_id=market_id,
                question=question,
                outcomes=outcome_tuples,
                liquidity=liquidity
            )
            
            if signal and signal.is_tradeable:
                signals.append(signal)
        
        # Sort by net profit
        signals.sort(key=lambda s: s.net_profit_pct, reverse=True)
        
        logger.info(f"Found {len(signals)} sum arb opportunities")
        
        return signals
    
    def scan_for_semantic_arbs(
        self,
        clusters: list[dict[str, Any]],
        market_data: dict[str, dict[str, Any]]
    ) -> list[ArbSignal]:
        """
        Scan semantic clusters for arbitrage opportunities.
        
        Args:
            clusters: Clusters from Grok semantic analysis
            market_data: Market data by market_id
            
        Returns:
            List of arb signals
        """
        signals = []
        
        for cluster in clusters:
            market_ids = cluster.get("market_ids", [])
            similarity = cluster.get("similarity_score", 0.0)
            
            if len(market_ids) < 2 or similarity < self.min_semantic_similarity:
                continue
            
            # Find best arb pair within cluster
            markets = [market_data.get(mid) for mid in market_ids if mid in market_data]
            markets = [m for m in markets if m is not None]
            
            if len(markets) < 2:
                continue
            
            # Check all pairs
            for i, m1 in enumerate(markets):
                for m2 in markets[i+1:]:
                    signal = self.generate_semantic_arb_signal(m1, m2, similarity)
                    if signal and signal.is_tradeable:
                        signals.append(signal)
        
        # Sort by risk-adjusted return
        signals.sort(key=lambda s: s.risk_adjusted_return, reverse=True)
        
        logger.info(f"Found {len(signals)} semantic arb opportunities")
        
        return signals
    
    def evaluate_trade(
        self,
        signal: ArbSignal,
        leg_outcomes: list[tuple[str, bool, float]]  # [(market_id, filled, fill_price), ...]
    ) -> dict[str, Any]:
        """
        Evaluate a completed arb trade.
        
        Args:
            signal: Original signal
            leg_outcomes: Outcome of each leg
            
        Returns:
            Evaluation dict
        """
        total_cost = 0.0
        total_revenue = 0.0
        legs_filled = 0
        
        for market_id, filled, fill_price in leg_outcomes:
            if not filled:
                continue
            
            legs_filled += 1
            
            # Find corresponding leg
            leg = next((l for l in signal.legs if l.market_id == market_id), None)
            if not leg:
                continue
            
            if leg.action == "buy":
                total_cost += fill_price * leg.size_usd
            else:
                total_revenue += fill_price * leg.size_usd
        
        # For sum arb, revenue is guaranteed 1.0 per unit if all outcomes covered
        if signal.arb_type.startswith("sum_arb"):
            if legs_filled == len(signal.legs):
                total_revenue = signal.recommended_size_usd
        
        pnl_usd = total_revenue - total_cost
        pnl_pct = pnl_usd / signal.recommended_size_usd if signal.recommended_size_usd > 0 else 0
        
        return {
            "timestamp": signal.timestamp,
            "arb_type": signal.arb_type,
            "n_legs": len(signal.legs),
            "legs_filled": legs_filled,
            "all_legs_filled": legs_filled == len(signal.legs),
            "expected_profit_pct": signal.net_profit_pct,
            "actual_pnl_usd": pnl_usd,
            "actual_pnl_pct": pnl_pct,
            "confidence": signal.confidence,
            "success": pnl_usd > 0
        }


def create_arb_policy(**kwargs) -> ArbPolicy:
    """Factory function to create arb policy."""
    return ArbPolicy(**kwargs)
