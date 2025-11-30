"""
LLM cost optimizer - route queries to cheapest capable model.
Tracks spend and optimizes for budget.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Literal
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

ModelTier = Literal["mini", "standard", "premium"]


class ModelProfile(BaseModel):
    """Cost and capability profile for a model."""
    name: str
    tier: ModelTier
    cost_per_1k_input: float
    cost_per_1k_output: float
    avg_latency_ms: int
    accuracy_score: float
    supports_reasoning: bool
    provider: str


# Model cost profiles
MODEL_PROFILES: dict[str, ModelProfile] = {
    "gpt-4o-mini": ModelProfile(
        name="gpt-4o-mini",
        tier="mini",
        cost_per_1k_input=0.00015,
        cost_per_1k_output=0.0006,
        avg_latency_ms=600,
        accuracy_score=0.73,
        supports_reasoning=False,
        provider="openai"
    ),
    "claude-3-haiku-20240307": ModelProfile(
        name="claude-3-haiku-20240307",
        tier="mini",
        cost_per_1k_input=0.00025,
        cost_per_1k_output=0.00125,
        avg_latency_ms=400,
        accuracy_score=0.70,
        supports_reasoning=False,
        provider="anthropic"
    ),
    "grok-2-1212": ModelProfile(
        name="grok-2-1212",
        tier="standard",
        cost_per_1k_input=0.002,
        cost_per_1k_output=0.01,
        avg_latency_ms=1500,
        accuracy_score=0.85,
        supports_reasoning=True,
        provider="xai"
    ),
    "gpt-4o": ModelProfile(
        name="gpt-4o",
        tier="standard",
        cost_per_1k_input=0.005,
        cost_per_1k_output=0.015,
        avg_latency_ms=2000,
        accuracy_score=0.85,
        supports_reasoning=False,
        provider="openai"
    ),
    "claude-3-5-sonnet-20241022": ModelProfile(
        name="claude-3-5-sonnet-20241022",
        tier="standard",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        avg_latency_ms=1800,
        accuracy_score=0.87,
        supports_reasoning=False,
        provider="anthropic"
    ),
    "claude-sonnet-4-20250514": ModelProfile(
        name="claude-sonnet-4-20250514",
        tier="premium",
        cost_per_1k_input=0.003,
        cost_per_1k_output=0.015,
        avg_latency_ms=2500,
        accuracy_score=0.90,
        supports_reasoning=False,
        provider="anthropic"
    ),
    "gpt-4-turbo-preview": ModelProfile(
        name="gpt-4-turbo-preview",
        tier="premium",
        cost_per_1k_input=0.01,
        cost_per_1k_output=0.03,
        avg_latency_ms=3000,
        accuracy_score=0.88,
        supports_reasoning=False,
        provider="openai"
    ),
}


class CostTracker:
    """Track LLM costs over time."""
    
    def __init__(self, daily_budget: float = 10.0):
        self.daily_budget = daily_budget
        self._costs: list[tuple[datetime, str, float, int]] = []  # time, model, cost, tokens
        
    def record(self, model: str, cost: float, tokens: int = 0):
        self._costs.append((datetime.now(), model, cost, tokens))
        self._cleanup()
    
    def _cleanup(self):
        cutoff = datetime.now() - timedelta(days=2)
        self._costs = [c for c in self._costs if c[0] > cutoff]
    
    @property
    def daily_spend(self) -> float:
        today = datetime.now().date()
        return sum(c[2] for c in self._costs if c[0].date() == today)
    
    @property
    def remaining_budget(self) -> float:
        return max(0, self.daily_budget - self.daily_spend)
    
    def get_model_stats(self) -> dict[str, dict]:
        """Get spend stats by model."""
        today = datetime.now().date()
        stats: dict[str, dict] = {}
        
        for t, model, cost, tokens in self._costs:
            if t.date() != today:
                continue
            if model not in stats:
                stats[model] = {"cost": 0.0, "tokens": 0, "calls": 0}
            stats[model]["cost"] += cost
            stats[model]["tokens"] += tokens
            stats[model]["calls"] += 1
        
        return stats
    
    def log_summary(self):
        logger.info(f"Daily spend: ${self.daily_spend:.4f} / ${self.daily_budget:.2f}")
        for model, stats in self.get_model_stats().items():
            logger.info(f"  {model}: ${stats['cost']:.4f} ({stats['calls']} calls)")


class SmartRouter:
    """
    Route queries to most cost-effective model.
    Higher edge + bigger position = worth more expensive model.
    """
    
    def __init__(
        self,
        cost_tracker: CostTracker,
        available_models: list[str] | None = None
    ):
        self.cost_tracker = cost_tracker
        self.available_models = available_models or list(MODEL_PROFILES.keys())
        
    def select_model(
        self,
        edge_estimate: float,
        position_size_usd: float,
        requires_reasoning: bool = False,
        max_latency_ms: int | None = None,
        min_accuracy: float = 0.7
    ) -> str:
        """Select best model for query."""
        remaining = self.cost_tracker.remaining_budget
        
        # Calculate expected value of correct prediction
        expected_value = abs(edge_estimate) * position_size_usd
        
        # Determine target tier
        if expected_value > 50:
            target_tier = "premium"
        elif expected_value > 10:
            target_tier = "standard"
        else:
            target_tier = "mini"
        
        # Filter candidates
        candidates = []
        for name in self.available_models:
            profile = MODEL_PROFILES.get(name)
            if not profile:
                continue
            
            if requires_reasoning and not profile.supports_reasoning:
                continue
            if max_latency_ms and profile.avg_latency_ms > max_latency_ms:
                continue
            if profile.accuracy_score < min_accuracy:
                continue
            
            # Estimate cost (500 input, 300 output tokens typical)
            est_cost = profile.cost_per_1k_input * 0.5 + profile.cost_per_1k_output * 0.3
            
            if est_cost > remaining * 0.1:  # Don't blow >10% on one query
                continue
            
            candidates.append((name, profile, est_cost))
        
        if not candidates:
            # Fallback to cheapest
            return self._cheapest_available()
        
        # Score candidates
        def score(item: tuple) -> float:
            name, profile, cost = item
            
            s = profile.accuracy_score * 100
            
            if profile.tier == target_tier:
                s += 20
            elif profile.tier == "standard" and target_tier == "mini":
                s += 10
            
            if expected_value > 0:
                s -= (cost / expected_value) * 50
            
            if profile.supports_reasoning and abs(edge_estimate) > 0.1:
                s += 15
            
            return s
        
        candidates.sort(key=score, reverse=True)
        return candidates[0][0]
    
    def get_consensus_models(self, num: int = 2) -> list[str]:
        """Select diverse models for consensus voting."""
        providers_seen = set()
        selected = []
        
        # Sort by accuracy
        sorted_models = sorted(
            self.available_models,
            key=lambda m: MODEL_PROFILES.get(m, ModelProfile(
                name=m, tier="mini", cost_per_1k_input=0,
                cost_per_1k_output=0, avg_latency_ms=0,
                accuracy_score=0, supports_reasoning=False,
                provider="unknown"
            )).accuracy_score,
            reverse=True
        )
        
        for model in sorted_models:
            profile = MODEL_PROFILES.get(model)
            if not profile:
                continue
            
            if profile.provider not in providers_seen:
                selected.append(model)
                providers_seen.add(profile.provider)
            
            if len(selected) >= num:
                break
        
        # Fill if needed
        for model in sorted_models:
            if model not in selected and len(selected) < num:
                selected.append(model)
        
        return selected[:num]
    
    def _cheapest_available(self) -> str:
        """Get cheapest available model."""
        cheapest = None
        cheapest_cost = float('inf')
        
        for name in self.available_models:
            profile = MODEL_PROFILES.get(name)
            if profile:
                cost = profile.cost_per_1k_input + profile.cost_per_1k_output
                if cost < cheapest_cost:
                    cheapest_cost = cost
                    cheapest = name
        
        return cheapest or "gpt-4o-mini"


# Global instances
cost_tracker = CostTracker(daily_budget=10.0)
smart_router = SmartRouter(cost_tracker)
