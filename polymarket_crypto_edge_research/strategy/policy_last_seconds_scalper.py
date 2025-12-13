"""
Last-seconds scalping policy for Polymarket.
Trades markets with imminent resolution based on price/probability divergence.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from core.logging_utils import get_logger

logger = get_logger(__name__)


class ScalpSignal(BaseModel):
    """Trading signal for last-seconds scalping."""
    
    timestamp: datetime
    market_id: str
    question: str
    
    # Market state
    outcome: str  # "yes" or "no"
    current_price: float = Field(ge=0.0, le=1.0)
    implied_prob: float = Field(ge=0.0, le=1.0)
    
    # Time until resolution
    minutes_until_resolution: float
    is_last_hour: bool = False
    is_last_5_min: bool = False
    
    # Signal
    action: str  # "buy", "sell", "hold"
    edge: float  # Expected edge over market price
    
    # Probabilities
    predicted_outcome_prob: float = Field(ge=0.0, le=1.0)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Position sizing
    recommended_size: float = Field(default=0.0, ge=0.0, le=1.0)
    max_position_usd: float = 100.0
    
    # Order book context
    best_bid: float = 0.0
    best_ask: float = 0.0
    spread: float = 0.0
    orderbook_imbalance: float = 0.0
    
    # Risk flags
    low_liquidity: bool = False
    high_spread: bool = False
    avoid_trade: bool = False
    reason: str = ""
    
    @property
    def is_tradeable(self) -> bool:
        """Check if signal is actionable."""
        return (
            self.action != "hold" and
            not self.avoid_trade and
            self.edge > 0.02 and  # 2% minimum edge
            self.recommended_size > 0
        )
    
    @property
    def expected_profit(self) -> float:
        """Expected profit if trade is correct."""
        if self.action == "buy":
            # Profit = (1 - price) if outcome happens
            return (1 - self.current_price) * self.predicted_outcome_prob
        elif self.action == "sell":
            # Profit = price if outcome doesn't happen
            return self.current_price * (1 - self.predicted_outcome_prob)
        return 0.0


class ScalperPolicy:
    """
    Last-seconds scalping policy for Polymarket.
    
    Strategy:
    1. Monitor markets approaching resolution
    2. Look for price/probability divergence using ML + external data
    3. Enter positions in last minutes/seconds when edge is clear
    4. Let positions resolve automatically
    
    Key features:
    - Only trades markets with imminent resolution
    - Uses orderbook microstructure for timing
    - Combines ML prediction with flow analysis
    - Conservative position sizing due to binary outcomes
    """
    
    def __init__(
        self,
        min_edge: float = 0.02,  # 2% minimum edge
        max_minutes: float = 60.0,  # Only trade last hour
        min_liquidity: float = 1000.0,  # Minimum $1000 liquidity
        max_spread: float = 0.05,  # 5% max spread
        max_position_pct: float = 0.10,  # Max 10% of liquidity
        confidence_threshold: float = 0.6
    ):
        self.min_edge = min_edge
        self.max_minutes = max_minutes
        self.min_liquidity = min_liquidity
        self.max_spread = max_spread
        self.max_position_pct = max_position_pct
        self.confidence_threshold = confidence_threshold
    
    def generate_signal(
        self,
        market_id: str,
        question: str,
        yes_price: float,
        minutes_until_resolution: float,
        predicted_yes_prob: float,
        confidence: float,
        orderbook: dict[str, Any] | None = None,
        underlying_data: dict[str, Any] | None = None
    ) -> ScalpSignal:
        """
        Generate scalping signal for a market.
        
        Args:
            market_id: Polymarket market ID
            question: Market question
            yes_price: Current YES price (0-1)
            minutes_until_resolution: Minutes until resolution
            predicted_yes_prob: ML predicted probability of YES
            confidence: Prediction confidence
            orderbook: Orderbook data (bid, ask, depth)
            underlying_data: Crypto price data if applicable
            
        Returns:
            ScalpSignal
        """
        from core.time_utils import now_utc
        
        timestamp = now_utc()
        
        # Check if within trading window
        is_last_hour = minutes_until_resolution <= 60
        is_last_5_min = minutes_until_resolution <= 5
        
        # Extract orderbook info
        best_bid = 0.0
        best_ask = 1.0
        spread = 1.0
        orderbook_imbalance = 0.0
        liquidity = 0.0
        
        if orderbook:
            best_bid = orderbook.get("best_bid", 0.0)
            best_ask = orderbook.get("best_ask", 1.0)
            spread = best_ask - best_bid
            orderbook_imbalance = orderbook.get("imbalance", 0.0)
            liquidity = orderbook.get("total_liquidity", 0.0)
        
        # Risk checks
        low_liquidity = liquidity < self.min_liquidity
        high_spread = spread > self.max_spread
        
        # Calculate edge
        # Edge = predicted probability - market implied probability
        edge_yes = predicted_yes_prob - yes_price
        edge_no = (1 - predicted_yes_prob) - (1 - yes_price)
        
        # Determine action
        action = "hold"
        outcome = "yes"
        edge = 0.0
        avoid_trade = False
        reason = ""
        
        if not is_last_hour:
            avoid_trade = True
            reason = f"Too far from resolution ({minutes_until_resolution:.0f}min)"
        elif low_liquidity:
            avoid_trade = True
            reason = f"Low liquidity (${liquidity:,.0f})"
        elif high_spread:
            avoid_trade = True
            reason = f"High spread ({spread:.1%})"
        elif confidence < self.confidence_threshold:
            avoid_trade = True
            reason = f"Low confidence ({confidence:.1%})"
        else:
            # Determine direction
            if edge_yes > self.min_edge and edge_yes > edge_no:
                action = "buy"
                outcome = "yes"
                edge = edge_yes
            elif edge_no > self.min_edge and edge_no > edge_yes:
                action = "buy"
                outcome = "no"
                edge = edge_no
            else:
                avoid_trade = True
                reason = f"Insufficient edge (yes: {edge_yes:.1%}, no: {edge_no:.1%})"
        
        # Calculate position size
        # More aggressive sizing closer to resolution
        time_factor = 1.0
        if is_last_5_min:
            time_factor = 1.5
        elif is_last_hour:
            time_factor = 1.0
        
        # Edge-based sizing
        edge_factor = min(1.0, edge / 0.10)  # Max out at 10% edge
        
        # Confidence factor
        conf_factor = (confidence - 0.5) * 2  # 0.5 -> 0, 1.0 -> 1.0
        
        if not avoid_trade and edge > 0:
            base_size = edge_factor * conf_factor * time_factor
            recommended_size = min(base_size, self.max_position_pct)
            
            # Reduce for orderbook imbalance against us
            if (action == "buy" and orderbook_imbalance < -0.3) or \
               (action == "sell" and orderbook_imbalance > 0.3):
                recommended_size *= 0.5
        else:
            recommended_size = 0.0
        
        # Max position in USD
        max_position_usd = min(
            liquidity * self.max_position_pct,
            1000.0  # Hard cap at $1000 per trade
        )
        
        return ScalpSignal(
            timestamp=timestamp,
            market_id=market_id,
            question=question,
            outcome=outcome,
            current_price=yes_price if outcome == "yes" else (1 - yes_price),
            implied_prob=yes_price,
            minutes_until_resolution=minutes_until_resolution,
            is_last_hour=is_last_hour,
            is_last_5_min=is_last_5_min,
            action=action,
            edge=edge,
            predicted_outcome_prob=predicted_yes_prob if outcome == "yes" else (1 - predicted_yes_prob),
            confidence=confidence,
            recommended_size=recommended_size,
            max_position_usd=max_position_usd,
            best_bid=best_bid,
            best_ask=best_ask,
            spread=spread,
            orderbook_imbalance=orderbook_imbalance,
            low_liquidity=low_liquidity,
            high_spread=high_spread,
            avoid_trade=avoid_trade,
            reason=reason
        )
    
    def scan_markets(
        self,
        markets: list[dict[str, Any]],
        predictions: dict[str, tuple[float, float]]  # market_id -> (prob, confidence)
    ) -> list[ScalpSignal]:
        """
        Scan multiple markets for scalping opportunities.
        
        Args:
            markets: List of market dicts with price/resolution info
            predictions: ML predictions per market
            
        Returns:
            List of tradeable signals, sorted by edge
        """
        signals = []
        
        for market in markets:
            market_id = market.get("market_id", "")
            
            if market_id not in predictions:
                continue
            
            prob, confidence = predictions[market_id]
            
            signal = self.generate_signal(
                market_id=market_id,
                question=market.get("question", ""),
                yes_price=market.get("yes_price", 0.5),
                minutes_until_resolution=market.get("minutes_until_resolution", 9999),
                predicted_yes_prob=prob,
                confidence=confidence,
                orderbook=market.get("orderbook")
            )
            
            if signal.is_tradeable:
                signals.append(signal)
        
        # Sort by edge
        signals.sort(key=lambda s: s.edge, reverse=True)
        
        logger.info(f"Found {len(signals)} tradeable scalp opportunities")
        
        return signals
    
    def evaluate_trade(
        self,
        signal: ScalpSignal,
        actual_outcome: str,  # "yes" or "no"
        fill_price: float
    ) -> dict[str, Any]:
        """
        Evaluate a completed scalp trade.
        
        Args:
            signal: Original signal
            actual_outcome: What actually happened
            fill_price: Price at which we filled
            
        Returns:
            Evaluation dict
        """
        # Did we bet on the right outcome?
        correct = signal.outcome == actual_outcome
        
        # Calculate P&L
        if correct:
            pnl_pct = (1 - fill_price) / fill_price  # Profit as % of investment
        else:
            pnl_pct = -1.0  # Lost entire investment
        
        pnl_usd = pnl_pct * signal.recommended_size * signal.max_position_usd
        
        return {
            "timestamp": signal.timestamp,
            "market_id": signal.market_id,
            "action": signal.action,
            "outcome_bet": signal.outcome,
            "actual_outcome": actual_outcome,
            "correct": correct,
            "edge_predicted": signal.edge,
            "fill_price": fill_price,
            "pnl_pct": pnl_pct,
            "pnl_usd": pnl_usd,
            "confidence": signal.confidence,
            "minutes_to_resolution": signal.minutes_until_resolution
        }


def create_scalper_policy(**kwargs) -> ScalperPolicy:
    """Factory function to create scalper policy."""
    return ScalperPolicy(**kwargs)
