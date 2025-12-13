"""
Smart Grok Trigger - Intelligent system to call Grok 4.1 only on unusual signals.

Instead of calling Grok on every prediction, this system:
1. Detects anomalies/unusual patterns
2. Identifies high-confidence divergences (ML vs whale vs market)
3. Only triggers Grok when there's edge to be gained
4. Tracks Grok call frequency and value-add

Key triggers:
- Whale consensus diverges from ML prediction
- Price momentum contradicts prediction direction
- Unusual volume/volatility spike
- Cross-venue price divergence
- High-value opportunity (potential edge > threshold)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TriggerReason(Enum):
    """Reasons for triggering Grok."""
    WHALE_ML_DIVERGENCE = "whale_ml_divergence"
    MOMENTUM_CONTRADICTION = "momentum_contradiction"
    VOLUME_SPIKE = "volume_spike"
    VOLATILITY_SPIKE = "volatility_spike"
    VENUE_DIVERGENCE = "venue_divergence"
    HIGH_VALUE_OPPORTUNITY = "high_value_opportunity"
    SIGNAL_CLUSTERING = "signal_clustering"
    CONSENSUS_UNCERTAINTY = "consensus_uncertainty"
    MANUAL_OVERRIDE = "manual_override"


@dataclass
class SignalContext:
    """Context for evaluating whether to trigger Grok."""

    asset: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ML Prediction
    ml_direction: str = ""  # "UP" or "DOWN"
    ml_confidence: float = 0.5

    # Whale consensus
    whale_direction: str = ""
    whale_confidence: float = 0.5
    whale_volume: float = 0.0

    # Market signals
    polymarket_yes_price: float = 0.5
    polymarket_spread_bps: float = 0.0
    polymarket_volume_24h: float = 0.0

    # Technical signals
    momentum: float = 0.0  # -1 to 1
    volatility: float = 0.0  # normalized
    rsi: float = 50.0

    # Multi-venue
    venue_price_std: float = 0.0  # Price standard deviation across venues
    cross_venue_spread: float = 0.0

    # Historical
    recent_accuracy: float = 0.5  # Our recent prediction accuracy
    grok_calls_last_hour: int = 0


@dataclass
class TriggerDecision:
    """Decision on whether to trigger Grok."""

    should_trigger: bool
    reasons: list[TriggerReason] = field(default_factory=list)
    urgency_score: float = 0.0  # 0-1, higher = more urgent
    expected_value: float = 0.0  # Expected edge from Grok consultation
    context: Optional[SignalContext] = None
    explanation: str = ""


class SmartGrokTrigger:
    """
    Intelligent trigger system for Grok 4.1 API calls.

    Only calls Grok when:
    1. There's genuine uncertainty or divergence
    2. The potential edge justifies the API cost
    3. We haven't exceeded rate limits
    """

    def __init__(
        self,
        # Divergence thresholds
        whale_ml_divergence_threshold: float = 0.3,
        momentum_contradiction_threshold: float = 0.4,
        volume_spike_threshold: float = 2.0,  # 2x normal
        volatility_spike_threshold: float = 2.0,
        venue_divergence_threshold: float = 0.002,  # 0.2%

        # Rate limiting
        max_calls_per_hour: int = 10,
        min_seconds_between_calls: int = 60,

        # Value thresholds
        min_expected_value: float = 0.05,  # 5% edge
        confidence_uncertainty_zone: tuple = (0.45, 0.65),  # Trigger in this zone
    ):
        self.whale_ml_divergence_threshold = whale_ml_divergence_threshold
        self.momentum_contradiction_threshold = momentum_contradiction_threshold
        self.volume_spike_threshold = volume_spike_threshold
        self.volatility_spike_threshold = volatility_spike_threshold
        self.venue_divergence_threshold = venue_divergence_threshold

        self.max_calls_per_hour = max_calls_per_hour
        self.min_seconds_between_calls = min_seconds_between_calls
        self.min_expected_value = min_expected_value
        self.confidence_uncertainty_zone = confidence_uncertainty_zone

        # Tracking
        self._call_history: list[datetime] = []
        self._last_call: Optional[datetime] = None
        self._calls_by_asset: dict[str, int] = {}
        self._trigger_stats: dict[TriggerReason, int] = {r: 0 for r in TriggerReason}

        # Baseline stats for comparison
        self._volume_baselines: dict[str, float] = {}
        self._volatility_baselines: dict[str, float] = {}

    def evaluate(self, ctx: SignalContext) -> TriggerDecision:
        """
        Evaluate whether to trigger Grok based on context.

        Returns TriggerDecision with reasoning.
        """
        reasons: list[TriggerReason] = []
        urgency = 0.0
        expected_value = 0.0

        # Rate limit check first
        if not self._check_rate_limits():
            return TriggerDecision(
                should_trigger=False,
                reasons=[],
                urgency_score=0,
                expected_value=0,
                context=ctx,
                explanation="Rate limit reached - saving Grok calls for higher priority signals",
            )

        # ─────────────────────────────────────────────────────────────────────
        # Trigger Checks
        # ─────────────────────────────────────────────────────────────────────

        # 1. Whale-ML Divergence
        whale_ml_div = self._check_whale_ml_divergence(ctx)
        if whale_ml_div > self.whale_ml_divergence_threshold:
            reasons.append(TriggerReason.WHALE_ML_DIVERGENCE)
            urgency += 0.3 * whale_ml_div
            expected_value += 0.1

        # 2. Momentum Contradiction
        momentum_score = self._check_momentum_contradiction(ctx)
        if momentum_score > self.momentum_contradiction_threshold:
            reasons.append(TriggerReason.MOMENTUM_CONTRADICTION)
            urgency += 0.25 * momentum_score
            expected_value += 0.08

        # 3. Volume Spike
        volume_ratio = self._check_volume_spike(ctx)
        if volume_ratio > self.volume_spike_threshold:
            reasons.append(TriggerReason.VOLUME_SPIKE)
            urgency += 0.2
            expected_value += 0.05

        # 4. Volatility Spike
        vol_ratio = self._check_volatility_spike(ctx)
        if vol_ratio > self.volatility_spike_threshold:
            reasons.append(TriggerReason.VOLATILITY_SPIKE)
            urgency += 0.2
            expected_value += 0.05

        # 5. Venue Divergence
        if ctx.venue_price_std > self.venue_divergence_threshold:
            reasons.append(TriggerReason.VENUE_DIVERGENCE)
            urgency += 0.15
            expected_value += 0.03

        # 6. Confidence in Uncertainty Zone
        if self.confidence_uncertainty_zone[0] <= ctx.ml_confidence <= self.confidence_uncertainty_zone[1]:
            reasons.append(TriggerReason.CONSENSUS_UNCERTAINTY)
            urgency += 0.1
            expected_value += 0.02

        # 7. High Value Opportunity (RSI extremes, high volume, etc.)
        high_value_score = self._check_high_value(ctx)
        if high_value_score > 0.5:
            reasons.append(TriggerReason.HIGH_VALUE_OPPORTUNITY)
            urgency += 0.25 * high_value_score
            expected_value += 0.1 * high_value_score

        # 8. Signal Clustering (multiple weak signals = strong signal)
        if len(reasons) >= 3:
            reasons.append(TriggerReason.SIGNAL_CLUSTERING)
            urgency += 0.2
            expected_value += 0.1

        # ─────────────────────────────────────────────────────────────────────
        # Decision
        # ─────────────────────────────────────────────────────────────────────

        # Normalize urgency
        urgency = min(1.0, urgency)

        # Decide
        should_trigger = (
            len(reasons) >= 1 and
            expected_value >= self.min_expected_value and
            urgency >= 0.2
        )

        explanation = self._build_explanation(reasons, urgency, expected_value, ctx)

        if should_trigger:
            self._record_trigger(ctx.asset, reasons)

        return TriggerDecision(
            should_trigger=should_trigger,
            reasons=reasons,
            urgency_score=urgency,
            expected_value=expected_value,
            context=ctx,
            explanation=explanation,
        )

    def force_trigger(self, ctx: SignalContext, reason: str = "") -> TriggerDecision:
        """Force a Grok trigger (manual override)."""
        self._record_trigger(ctx.asset, [TriggerReason.MANUAL_OVERRIDE])
        return TriggerDecision(
            should_trigger=True,
            reasons=[TriggerReason.MANUAL_OVERRIDE],
            urgency_score=1.0,
            expected_value=0.1,
            context=ctx,
            explanation=f"Manual override: {reason}",
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Trigger Check Methods
    # ─────────────────────────────────────────────────────────────────────────

    def _check_whale_ml_divergence(self, ctx: SignalContext) -> float:
        """Check if whale consensus diverges from ML prediction."""
        if ctx.ml_direction == ctx.whale_direction:
            return 0.0

        # Divergence strength = confidence delta
        if ctx.ml_direction != ctx.whale_direction:
            # Opposite directions with high confidence on both = high divergence
            return (ctx.ml_confidence + ctx.whale_confidence) / 2
        return 0.0

    def _check_momentum_contradiction(self, ctx: SignalContext) -> float:
        """Check if momentum contradicts prediction."""
        ml_sign = 1 if ctx.ml_direction == "UP" else -1
        if ml_sign * ctx.momentum < -0.3:
            # Momentum is strongly opposite to prediction
            return abs(ctx.momentum)
        return 0.0

    def _check_volume_spike(self, ctx: SignalContext) -> float:
        """Check for unusual volume."""
        baseline = self._volume_baselines.get(ctx.asset, ctx.polymarket_volume_24h)
        if baseline <= 0:
            self._volume_baselines[ctx.asset] = ctx.polymarket_volume_24h
            return 1.0
        return ctx.polymarket_volume_24h / baseline

    def _check_volatility_spike(self, ctx: SignalContext) -> float:
        """Check for unusual volatility."""
        baseline = self._volatility_baselines.get(ctx.asset, ctx.volatility)
        if baseline <= 0:
            self._volatility_baselines[ctx.asset] = max(0.01, ctx.volatility)
            return 1.0
        return ctx.volatility / baseline

    def _check_high_value(self, ctx: SignalContext) -> float:
        """Check for high-value trading opportunity."""
        score = 0.0

        # RSI extremes
        if ctx.rsi < 30 or ctx.rsi > 70:
            score += 0.3

        # High volume with high confidence
        if ctx.polymarket_volume_24h > 10000 and ctx.ml_confidence > 0.7:
            score += 0.3

        # Tight spread (liquid market)
        if ctx.polymarket_spread_bps < 50:
            score += 0.2

        # Strong whale conviction
        if ctx.whale_confidence > 0.7 and ctx.whale_volume > 50000:
            score += 0.3

        return min(1.0, score)

    # ─────────────────────────────────────────────────────────────────────────
    # Rate Limiting
    # ─────────────────────────────────────────────────────────────────────────

    def _check_rate_limits(self) -> bool:
        """Check if we're within rate limits."""
        now = datetime.now(timezone.utc)

        # Clean old history
        hour_ago = now - timedelta(hours=1)
        self._call_history = [t for t in self._call_history if t > hour_ago]

        # Check hourly limit
        if len(self._call_history) >= self.max_calls_per_hour:
            return False

        # Check minimum time between calls
        if self._last_call:
            seconds_since = (now - self._last_call).total_seconds()
            if seconds_since < self.min_seconds_between_calls:
                return False

        return True

    def _record_trigger(self, asset: str, reasons: list[TriggerReason]):
        """Record a trigger for rate limiting and stats."""
        now = datetime.now(timezone.utc)
        self._call_history.append(now)
        self._last_call = now
        self._calls_by_asset[asset] = self._calls_by_asset.get(asset, 0) + 1

        for reason in reasons:
            self._trigger_stats[reason] = self._trigger_stats.get(reason, 0) + 1

    # ─────────────────────────────────────────────────────────────────────────
    # Explanation Building
    # ─────────────────────────────────────────────────────────────────────────

    def _build_explanation(
        self,
        reasons: list[TriggerReason],
        urgency: float,
        expected_value: float,
        ctx: SignalContext,
    ) -> str:
        """Build human-readable explanation."""
        if not reasons:
            return "No unusual signals detected - using cached/ML prediction"

        parts = []
        for reason in reasons:
            if reason == TriggerReason.WHALE_ML_DIVERGENCE:
                parts.append(
                    f"Whale consensus ({ctx.whale_direction}) diverges from ML ({ctx.ml_direction})"
                )
            elif reason == TriggerReason.MOMENTUM_CONTRADICTION:
                parts.append(f"Momentum ({ctx.momentum:.2f}) contradicts prediction")
            elif reason == TriggerReason.VOLUME_SPIKE:
                parts.append("Unusual volume spike detected")
            elif reason == TriggerReason.VOLATILITY_SPIKE:
                parts.append("Volatility spike detected")
            elif reason == TriggerReason.VENUE_DIVERGENCE:
                parts.append(f"Cross-venue price divergence ({ctx.venue_price_std*100:.2f}%)")
            elif reason == TriggerReason.CONSENSUS_UNCERTAINTY:
                parts.append(f"Confidence in uncertainty zone ({ctx.ml_confidence:.1%})")
            elif reason == TriggerReason.HIGH_VALUE_OPPORTUNITY:
                parts.append("High-value opportunity detected")
            elif reason == TriggerReason.SIGNAL_CLUSTERING:
                parts.append("Multiple signals clustering")
            elif reason == TriggerReason.MANUAL_OVERRIDE:
                parts.append("Manual override")

        trigger_str = "TRIGGERING" if len(reasons) > 0 else "NOT TRIGGERING"
        return f"{trigger_str} Grok: {', '.join(parts)}. Urgency: {urgency:.1%}, EV: {expected_value:.1%}"

    # ─────────────────────────────────────────────────────────────────────────
    # Stats
    # ─────────────────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        """Get trigger statistics."""
        return {
            "total_triggers": sum(self._trigger_stats.values()),
            "calls_last_hour": len(self._call_history),
            "calls_by_asset": self._calls_by_asset.copy(),
            "triggers_by_reason": {r.value: c for r, c in self._trigger_stats.items()},
            "last_call": self._last_call.isoformat() if self._last_call else None,
        }

    def update_baselines(self, asset: str, volume: float, volatility: float):
        """Update baselines for spike detection (call periodically)."""
        # Exponential moving average
        alpha = 0.1
        if asset in self._volume_baselines:
            self._volume_baselines[asset] = alpha * volume + (1 - alpha) * self._volume_baselines[asset]
        else:
            self._volume_baselines[asset] = volume

        if asset in self._volatility_baselines:
            self._volatility_baselines[asset] = alpha * volatility + (1 - alpha) * self._volatility_baselines[asset]
        else:
            self._volatility_baselines[asset] = volatility


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────


def demo():
    """Demo the smart trigger system."""
    trigger = SmartGrokTrigger(
        max_calls_per_hour=10,
        min_seconds_between_calls=30,
    )

    # Scenario 1: Normal conditions (no trigger)
    print("=" * 60)
    print("SCENARIO 1: Normal conditions")
    ctx1 = SignalContext(
        asset="BTC",
        ml_direction="UP",
        ml_confidence=0.65,
        whale_direction="UP",
        whale_confidence=0.60,
        momentum=0.3,
        volatility=0.02,
        rsi=55,
    )
    decision1 = trigger.evaluate(ctx1)
    print(f"Should trigger: {decision1.should_trigger}")
    print(f"Explanation: {decision1.explanation}")

    # Scenario 2: Whale-ML divergence (should trigger)
    print("\n" + "=" * 60)
    print("SCENARIO 2: Whale-ML divergence")
    ctx2 = SignalContext(
        asset="ETH",
        ml_direction="UP",
        ml_confidence=0.72,
        whale_direction="DOWN",
        whale_confidence=0.68,
        momentum=-0.4,
        volatility=0.05,
        rsi=35,
        polymarket_volume_24h=50000,
    )
    decision2 = trigger.evaluate(ctx2)
    print(f"Should trigger: {decision2.should_trigger}")
    print(f"Reasons: {[r.value for r in decision2.reasons]}")
    print(f"Urgency: {decision2.urgency_score:.1%}")
    print(f"Explanation: {decision2.explanation}")

    # Scenario 3: Multiple weak signals clustering
    print("\n" + "=" * 60)
    print("SCENARIO 3: Signal clustering")
    ctx3 = SignalContext(
        asset="SOL",
        ml_direction="DOWN",
        ml_confidence=0.55,  # Uncertainty zone
        whale_direction="DOWN",
        whale_confidence=0.55,
        momentum=0.35,  # Slight contradiction
        volatility=0.08,  # High
        rsi=28,  # Extreme
        venue_price_std=0.003,  # Venue divergence
        polymarket_volume_24h=75000,
    )
    decision3 = trigger.evaluate(ctx3)
    print(f"Should trigger: {decision3.should_trigger}")
    print(f"Reasons: {[r.value for r in decision3.reasons]}")
    print(f"Explanation: {decision3.explanation}")

    # Stats
    print("\n" + "=" * 60)
    print("TRIGGER STATS")
    print(trigger.get_stats())


if __name__ == "__main__":
    demo()
