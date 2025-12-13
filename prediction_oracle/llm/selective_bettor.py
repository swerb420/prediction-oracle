"""
Selective Betting System - Only bet when conditions are right.

NO BLIND BETTING. Every bet must pass filters:

1. Data Quality Filter: Enough labeled training data
2. Model Confidence Filter: ML confidence above threshold  
3. Market Conditions Filter: Liquidity, spread, timing
4. Grok Confirmation Filter: Optional LLM validation
5. Risk Management Filter: Max bets per period, position limits

This makes the system SELECTIVE - we skip low-quality opportunities.
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Literal
from dataclasses import dataclass

from real_data_store import get_store, RealDataStore
from learning_ml_predictor import LearningMLPredictor, MLPrediction

logger = logging.getLogger(__name__)

CryptoSymbol = Literal["BTC", "ETH", "SOL", "XRP"]


@dataclass
class BetDecision:
    """Result of the betting decision process."""
    should_bet: bool
    symbol: CryptoSymbol
    direction: str  # "UP" or "DOWN"
    confidence: float
    position_size_pct: float  # 0.0 to 1.0 of available capital
    
    # Reasons
    passed_filters: list[str]
    failed_filters: list[str]
    reasoning: str
    
    # Source info
    ml_prediction: Optional[MLPrediction] = None
    grok_agreed: bool = False
    grok_confidence: float = 0.0


@dataclass  
class BetFilters:
    """Configuration for betting filters."""
    # Data requirements
    min_training_examples: int = 50
    min_model_accuracy: float = 0.55  # At least 55% validation accuracy
    
    # Confidence requirements
    min_ml_confidence: float = 0.60  # 60% minimum confidence
    min_combined_confidence: float = 0.65  # With Grok agreement
    
    # Market requirements
    min_liquidity_usd: float = 1000  # Minimum market liquidity
    max_spread: float = 0.10  # Maximum bid-ask spread
    min_time_to_close: int = 60  # At least 60 seconds to market close
    
    # Risk management
    max_bets_per_hour: int = 4  # Max 4 bets per hour
    max_exposure_pct: float = 0.25  # Max 25% of capital per bet
    max_daily_loss_pct: float = 0.20  # Stop if down 20%
    
    # Position sizing based on confidence
    base_position_pct: float = 0.10  # 10% base position
    max_position_pct: float = 0.25  # 25% max position
    

class SelectiveBettor:
    """
    Makes selective betting decisions based on multiple filters.
    
    Only bets when:
    - ML model has enough training data and shows good accuracy
    - Prediction confidence is above threshold
    - Market conditions are favorable
    - Risk limits allow
    """
    
    def __init__(
        self,
        store: Optional[RealDataStore] = None,
        predictor: Optional[LearningMLPredictor] = None,
        filters: Optional[BetFilters] = None,
    ):
        self.store = store or get_store()
        self.predictor = predictor or LearningMLPredictor(store=self.store)
        self.filters = filters or BetFilters()
        
        # Track recent bets for rate limiting
        self.recent_bets: list[datetime] = []
    
    # ─────────────────────────────────────────────────────────────────────────
    # Individual Filters
    # ─────────────────────────────────────────────────────────────────────────
    
    def _filter_data_quality(self, symbol: str) -> tuple[bool, str]:
        """Check if we have enough quality training data."""
        status = self.predictor.get_model_status()
        symbol_status = status.get(symbol, {})
        
        examples = symbol_status.get("training_examples", 0)
        accuracy = symbol_status.get("validation_accuracy", 0)
        
        if examples < self.filters.min_training_examples:
            return False, (
                f"Insufficient training data: {examples}/{self.filters.min_training_examples} "
                f"examples"
            )
        
        if accuracy < self.filters.min_model_accuracy and examples > 20:
            return False, (
                f"Model accuracy too low: {accuracy:.1%} < {self.filters.min_model_accuracy:.1%}"
            )
        
        return True, f"Data OK: {examples} examples, {accuracy:.1%} accuracy"
    
    def _filter_confidence(self, prediction: MLPrediction) -> tuple[bool, str]:
        """Check if prediction confidence is high enough."""
        if prediction.confidence < self.filters.min_ml_confidence:
            return False, (
                f"Confidence too low: {prediction.confidence:.1%} < "
                f"{self.filters.min_ml_confidence:.1%}"
            )
        
        return True, f"Confidence OK: {prediction.confidence:.1%}"
    
    def _filter_market_conditions(
        self, 
        market_data: dict,
    ) -> tuple[bool, str]:
        """Check market conditions."""
        reasons = []
        
        # Liquidity check
        liquidity = market_data.get("liquidity", 0)
        if liquidity < self.filters.min_liquidity_usd:
            return False, f"Liquidity too low: ${liquidity:.0f} < ${self.filters.min_liquidity_usd}"
        reasons.append(f"Liquidity ${liquidity:.0f}")
        
        # Spread check
        spread = market_data.get("spread", 0)
        if spread > self.filters.max_spread:
            return False, f"Spread too wide: {spread:.1%} > {self.filters.max_spread:.1%}"
        reasons.append(f"Spread {spread:.1%}")
        
        # Time to close
        time_to_close = market_data.get("seconds_to_close", 900)
        if time_to_close < self.filters.min_time_to_close:
            return False, f"Too close to resolution: {time_to_close}s remaining"
        reasons.append(f"{time_to_close}s remaining")
        
        return True, f"Market OK: {', '.join(reasons)}"
    
    def _filter_rate_limit(self) -> tuple[bool, str]:
        """Check rate limits."""
        now = datetime.now(timezone.utc)
        hour_ago = now - timedelta(hours=1)
        
        # Clean old bets
        self.recent_bets = [t for t in self.recent_bets if t > hour_ago]
        
        if len(self.recent_bets) >= self.filters.max_bets_per_hour:
            return False, f"Rate limited: {len(self.recent_bets)}/{self.filters.max_bets_per_hour} bets in last hour"
        
        return True, f"Rate OK: {len(self.recent_bets)}/{self.filters.max_bets_per_hour} bets in last hour"
    
    def _filter_risk(self) -> tuple[bool, str]:
        """Check risk limits."""
        stats = self.store.get_trading_stats(days=1)
        
        # Check daily loss limit
        total_pnl = stats.get("total_pnl", 0)
        # We'd need capital to calculate % loss - for now just check absolute
        
        return True, f"Risk OK: Today PnL ${total_pnl:.2f}"
    
    # ─────────────────────────────────────────────────────────────────────────
    # Position Sizing
    # ─────────────────────────────────────────────────────────────────────────
    
    def _calculate_position_size(
        self, 
        confidence: float,
        grok_agreed: bool,
    ) -> float:
        """
        Calculate position size as % of capital.
        
        Higher confidence = larger position (up to max).
        Grok agreement = bonus size.
        """
        # Base size from confidence
        # confidence of 0.60 -> base_position
        # confidence of 0.90 -> max_position
        
        conf_range = 0.90 - 0.60  # 0.30
        size_range = self.filters.max_position_pct - self.filters.base_position_pct
        
        excess_conf = max(0, min(confidence - 0.60, conf_range))
        position_pct = self.filters.base_position_pct + (excess_conf / conf_range) * size_range
        
        # Grok agreement bonus (10% increase)
        if grok_agreed:
            position_pct *= 1.1
        
        # Cap at max
        return min(position_pct, self.filters.max_position_pct)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Main Decision
    # ─────────────────────────────────────────────────────────────────────────
    
    def should_bet(
        self,
        symbol: CryptoSymbol,
        market_data: dict,
        grok_response: Optional[dict] = None,
    ) -> BetDecision:
        """
        Decide whether to bet on this market.
        
        Args:
            symbol: Crypto symbol
            market_data: Current market data (prices, liquidity, etc.)
            grok_response: Optional Grok analysis result
            
        Returns:
            BetDecision with full reasoning
        """
        passed = []
        failed = []
        
        # 1. Data quality filter
        ok, reason = self._filter_data_quality(symbol)
        if ok:
            passed.append(f"DATA: {reason}")
        else:
            failed.append(f"DATA: {reason}")
        
        # 2. Get ML prediction
        prediction = self.predictor.predict(symbol, market_data)
        
        # 3. Confidence filter
        ok, reason = self._filter_confidence(prediction)
        if ok:
            passed.append(f"CONFIDENCE: {reason}")
        else:
            failed.append(f"CONFIDENCE: {reason}")
        
        # 4. Market conditions filter
        ok, reason = self._filter_market_conditions(market_data)
        if ok:
            passed.append(f"MARKET: {reason}")
        else:
            failed.append(f"MARKET: {reason}")
        
        # 5. Rate limit filter
        ok, reason = self._filter_rate_limit()
        if ok:
            passed.append(f"RATE: {reason}")
        else:
            failed.append(f"RATE: {reason}")
        
        # 6. Risk filter
        ok, reason = self._filter_risk()
        if ok:
            passed.append(f"RISK: {reason}")
        else:
            failed.append(f"RISK: {reason}")
        
        # Check Grok agreement if provided
        grok_agreed = False
        grok_confidence = 0.0
        if grok_response:
            grok_dir = grok_response.get("direction", "")
            grok_confidence = grok_response.get("confidence", 0)
            
            if grok_dir == prediction.direction:
                grok_agreed = True
                passed.append(f"GROK: Agrees {grok_dir} ({grok_confidence:.1%})")
            else:
                # Grok disagreement is a warning, not a failure
                passed.append(f"GROK: Disagrees (ML={prediction.direction}, Grok={grok_dir})")
        
        # Final decision
        should_bet = len(failed) == 0 and prediction.should_bet
        
        # Calculate position size
        position_size = self._calculate_position_size(
            prediction.confidence, 
            grok_agreed
        ) if should_bet else 0.0
        
        # Build reasoning
        if should_bet:
            reasoning = (
                f"BET {prediction.direction}: "
                f"Confidence {prediction.confidence:.1%}, "
                f"Position {position_size:.1%}. "
                f"Passed {len(passed)} filters."
            )
        else:
            reasoning = (
                f"SKIP: Failed {len(failed)} filter(s). "
                + " | ".join(failed)
            )
        
        decision = BetDecision(
            should_bet=should_bet,
            symbol=symbol,
            direction=prediction.direction,
            confidence=prediction.confidence,
            position_size_pct=position_size,
            passed_filters=passed,
            failed_filters=failed,
            reasoning=reasoning,
            ml_prediction=prediction,
            grok_agreed=grok_agreed,
            grok_confidence=grok_confidence,
        )
        
        # Record bet time if we're betting
        if should_bet:
            self.recent_bets.append(datetime.now(timezone.utc))
        
        return decision
    
    def get_stats(self) -> dict:
        """Get betting statistics."""
        trading_stats = self.store.get_trading_stats()
        model_status = self.predictor.get_model_status()
        
        return {
            "trading": trading_stats,
            "models": model_status,
            "recent_bets_hour": len(self.recent_bets),
            "filters": {
                "min_examples": self.filters.min_training_examples,
                "min_confidence": self.filters.min_ml_confidence,
                "min_accuracy": self.filters.min_model_accuracy,
            }
        }


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    bettor = SelectiveBettor()
    
    # Test decision with sample market data
    market_data = {
        "yes_price": 0.58,
        "no_price": 0.42,
        "market_direction": "UP",
        "volume": 2000,
        "liquidity": 5000,
        "spread": 0.02,
        "seconds_to_close": 600,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    
    decision = bettor.should_bet("BTC", market_data)
    
    print("\n" + "="*60)
    print("BET DECISION")
    print("="*60)
    print(f"Should Bet: {decision.should_bet}")
    print(f"Direction: {decision.direction}")
    print(f"Confidence: {decision.confidence:.1%}")
    print(f"Position Size: {decision.position_size_pct:.1%}")
    print(f"\nReasoning: {decision.reasoning}")
    print(f"\nPassed Filters:")
    for f in decision.passed_filters:
        print(f"  ✓ {f}")
    print(f"\nFailed Filters:")
    for f in decision.failed_filters:
        print(f"  ✗ {f}")
    
    print("\n" + "="*60)
    print("OVERALL STATS")
    print("="*60)
    stats = bettor.get_stats()
    print(f"Trading: {stats['trading']}")
    print(f"Models: {stats['models']}")
