"""
Cross-market feature builder for arbitrage and correlation detection.
Finds sum≠1 opportunities and semantically related markets.
"""

from datetime import datetime
from typing import Any

import numpy as np
from pydantic import BaseModel

from core.logging_utils import get_logger
from data.schemas import PolymarketMarket

logger = get_logger(__name__)


class CrossMarketFeatures(BaseModel):
    """Features for cross-market analysis."""
    
    timestamp: datetime
    
    # Sum arbitrage features
    markets_with_sum_arb: int
    avg_sum_deviation: float
    max_sum_deviation: float
    best_arb_market_id: str | None = None
    best_arb_profit_pct: float = 0.0
    
    # Market correlation features
    crypto_market_count: int
    politics_market_count: int
    sports_market_count: int
    
    # Semantic clustering (from Grok)
    semantic_cluster_count: int = 0
    duplicate_market_pairs: int = 0
    
    # Volume concentration
    top_3_volume_share: float
    hhi_concentration: float  # Herfindahl-Hirschman Index
    
    # Resolution timing
    markets_resolving_1h: int
    markets_resolving_24h: int
    resolution_clustering: float  # How clustered resolution times are
    
    def to_dict(self) -> dict[str, float]:
        """Convert to dict for ML model."""
        return {
            "markets_with_sum_arb": float(self.markets_with_sum_arb),
            "avg_sum_deviation": self.avg_sum_deviation,
            "max_sum_deviation": self.max_sum_deviation,
            "best_arb_profit_pct": self.best_arb_profit_pct,
            "crypto_market_count": float(self.crypto_market_count),
            "politics_market_count": float(self.politics_market_count),
            "sports_market_count": float(self.sports_market_count),
            "semantic_cluster_count": float(self.semantic_cluster_count),
            "duplicate_market_pairs": float(self.duplicate_market_pairs),
            "top_3_volume_share": self.top_3_volume_share,
            "hhi_concentration": self.hhi_concentration,
            "markets_resolving_1h": float(self.markets_resolving_1h),
            "markets_resolving_24h": float(self.markets_resolving_24h),
            "resolution_clustering": self.resolution_clustering,
        }


class ArbOpportunity(BaseModel):
    """Detected arbitrage opportunity."""
    
    market_id: str
    question: str
    outcome_prices: list[float]
    outcome_sum: float
    deviation_pct: float
    expected_profit_pct: float
    confidence: float
    arb_type: str  # "sum_under_1", "sum_over_1", "semantic_mismatch"
    
    @property
    def is_profitable(self) -> bool:
        return self.expected_profit_pct > 0.01  # 1% threshold


class SemanticMatch(BaseModel):
    """Semantically related markets that might be duplicates."""
    
    market_id_1: str
    market_id_2: str
    question_1: str
    question_2: str
    similarity_score: float
    price_divergence: float
    
    @property
    def has_arb_potential(self) -> bool:
        return self.similarity_score > 0.9 and self.price_divergence > 0.05


class CrossMarketFeatureBuilder:
    """
    Builds cross-market features for arbitrage and correlation analysis.
    """
    
    @staticmethod
    def feature_names() -> list[str]:
        """Get list of feature names."""
        return [
            "markets_with_sum_arb", "avg_sum_deviation", "max_sum_deviation",
            "best_arb_profit_pct",
            "crypto_market_count", "politics_market_count", "sports_market_count",
            "semantic_cluster_count", "duplicate_market_pairs",
            "top_3_volume_share", "hhi_concentration",
            "markets_resolving_1h", "markets_resolving_24h",
            "resolution_clustering",
        ]
    
    def build(
        self,
        markets: list[PolymarketMarket],
        semantic_clusters: dict[str, list[str]] | None = None
    ) -> CrossMarketFeatures:
        """
        Build cross-market features from a list of markets.
        
        Args:
            markets: List of markets to analyze
            semantic_clusters: Optional clustering from Grok
            
        Returns:
            CrossMarketFeatures
        """
        from core.time_utils import now_utc
        
        timestamp = now_utc()
        
        if not markets:
            return CrossMarketFeatures(
                timestamp=timestamp,
                markets_with_sum_arb=0,
                avg_sum_deviation=0.0,
                max_sum_deviation=0.0,
                crypto_market_count=0,
                politics_market_count=0,
                sports_market_count=0,
                top_3_volume_share=0.0,
                hhi_concentration=0.0,
                markets_resolving_1h=0,
                markets_resolving_24h=0,
                resolution_clustering=0.0,
            )
        
        # Sum arbitrage analysis
        sum_deviations = []
        arb_markets = []
        
        for market in markets:
            outcome_sum = market.outcome_sum
            deviation = abs(outcome_sum - 1.0)
            sum_deviations.append(deviation)
            
            if deviation > 0.02:  # 2% threshold
                arb_markets.append((market.market_id, deviation))
        
        markets_with_sum_arb = len(arb_markets)
        avg_sum_deviation = np.mean(sum_deviations) if sum_deviations else 0.0
        max_sum_deviation = np.max(sum_deviations) if sum_deviations else 0.0
        
        best_arb_market_id = None
        best_arb_profit_pct = 0.0
        if arb_markets:
            arb_markets.sort(key=lambda x: x[1], reverse=True)
            best_arb_market_id = arb_markets[0][0]
            best_arb_profit_pct = arb_markets[0][1]
        
        # Category counts
        crypto_keywords = ["btc", "bitcoin", "eth", "ethereum", "sol", "solana", "crypto"]
        politics_keywords = ["election", "president", "vote", "congress", "senate", "trump", "biden"]
        sports_keywords = ["nfl", "nba", "mlb", "nhl", "soccer", "football", "basketball"]
        
        crypto_count = sum(
            1 for m in markets
            if any(kw in m.question.lower() for kw in crypto_keywords)
        )
        politics_count = sum(
            1 for m in markets
            if any(kw in m.question.lower() for kw in politics_keywords)
        )
        sports_count = sum(
            1 for m in markets
            if any(kw in m.question.lower() for kw in sports_keywords)
        )
        
        # Semantic clustering
        semantic_cluster_count = len(semantic_clusters) if semantic_clusters else 0
        duplicate_market_pairs = 0
        if semantic_clusters:
            for cluster_markets in semantic_clusters.values():
                if len(cluster_markets) > 1:
                    duplicate_market_pairs += len(cluster_markets) * (len(cluster_markets) - 1) // 2
        
        # Volume concentration
        volumes = [m.volume_24h for m in markets if m.volume_24h > 0]
        if volumes:
            volumes.sort(reverse=True)
            total_volume = sum(volumes)
            top_3_volume = sum(volumes[:3])
            top_3_volume_share = top_3_volume / total_volume if total_volume > 0 else 0
            
            # HHI (sum of squared market shares)
            shares = [v / total_volume for v in volumes]
            hhi_concentration = sum(s ** 2 for s in shares)
        else:
            top_3_volume_share = 0.0
            hhi_concentration = 0.0
        
        # Resolution timing
        markets_resolving_1h = sum(1 for m in markets if m.is_last_hour)
        markets_resolving_24h = sum(
            1 for m in markets
            if m.minutes_until_resolution and 0 < m.minutes_until_resolution <= 1440
        )
        
        # Resolution clustering (std of resolution times)
        resolution_times = [
            m.minutes_until_resolution for m in markets
            if m.minutes_until_resolution and m.minutes_until_resolution > 0
        ]
        if len(resolution_times) >= 2:
            resolution_clustering = 1.0 / (1.0 + np.std(resolution_times) / 60)  # Normalize by hours
        else:
            resolution_clustering = 0.0
        
        return CrossMarketFeatures(
            timestamp=timestamp,
            markets_with_sum_arb=markets_with_sum_arb,
            avg_sum_deviation=float(avg_sum_deviation),
            max_sum_deviation=float(max_sum_deviation),
            best_arb_market_id=best_arb_market_id,
            best_arb_profit_pct=best_arb_profit_pct,
            crypto_market_count=crypto_count,
            politics_market_count=politics_count,
            sports_market_count=sports_count,
            semantic_cluster_count=semantic_cluster_count,
            duplicate_market_pairs=duplicate_market_pairs,
            top_3_volume_share=float(top_3_volume_share),
            hhi_concentration=float(hhi_concentration),
            markets_resolving_1h=markets_resolving_1h,
            markets_resolving_24h=markets_resolving_24h,
            resolution_clustering=float(resolution_clustering),
        )
    
    def find_sum_arb_opportunities(
        self,
        markets: list[PolymarketMarket],
        min_deviation: float = 0.02
    ) -> list[ArbOpportunity]:
        """
        Find markets where outcome prices don't sum to 1.0.
        
        Args:
            markets: Markets to analyze
            min_deviation: Minimum deviation to flag (default 2%)
            
        Returns:
            List of arbitrage opportunities
        """
        opportunities = []
        
        for market in markets:
            outcome_sum = market.outcome_sum
            deviation = abs(outcome_sum - 1.0)
            
            if deviation < min_deviation:
                continue
            
            # Calculate expected profit
            if outcome_sum < 1.0:
                # Can buy all outcomes for less than 1.0
                profit_pct = (1.0 - outcome_sum) / outcome_sum
                arb_type = "sum_under_1"
            else:
                # Can sell all outcomes for more than 1.0
                profit_pct = (outcome_sum - 1.0) / outcome_sum
                arb_type = "sum_over_1"
            
            # Confidence based on liquidity and volume
            liquidity_factor = min(1.0, market.liquidity / 10000)
            volume_factor = min(1.0, market.volume_24h / 50000)
            confidence = (liquidity_factor + volume_factor) / 2
            
            opportunities.append(ArbOpportunity(
                market_id=market.market_id,
                question=market.question,
                outcome_prices=[o.price for o in market.outcomes],
                outcome_sum=outcome_sum,
                deviation_pct=deviation,
                expected_profit_pct=profit_pct,
                confidence=confidence,
                arb_type=arb_type
            ))
        
        # Sort by expected profit
        opportunities.sort(key=lambda x: x.expected_profit_pct, reverse=True)
        
        return opportunities
    
    def find_semantic_matches(
        self,
        markets: list[PolymarketMarket],
        similarity_threshold: float = 0.8
    ) -> list[SemanticMatch]:
        """
        Find semantically similar markets using simple text matching.
        For better results, use Grok-based clustering.
        
        Args:
            markets: Markets to analyze
            similarity_threshold: Minimum similarity to flag
            
        Returns:
            List of semantic matches
        """
        matches = []
        
        # Simple word overlap similarity
        def word_similarity(q1: str, q2: str) -> float:
            words1 = set(q1.lower().split())
            words2 = set(q2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
        
        for i, m1 in enumerate(markets):
            for m2 in markets[i+1:]:
                similarity = word_similarity(m1.question, m2.question)
                
                if similarity >= similarity_threshold:
                    # Calculate price divergence
                    if m1.outcomes and m2.outcomes:
                        p1 = m1.outcomes[0].price
                        p2 = m2.outcomes[0].price
                        price_div = abs(p1 - p2)
                    else:
                        price_div = 0.0
                    
                    matches.append(SemanticMatch(
                        market_id_1=m1.market_id,
                        market_id_2=m2.market_id,
                        question_1=m1.question,
                        question_2=m2.question,
                        similarity_score=similarity,
                        price_divergence=price_div
                    ))
        
        # Sort by similarity
        matches.sort(key=lambda x: x.similarity_score, reverse=True)
        
        return matches


def build_cross_market_features(
    markets: list[PolymarketMarket],
    semantic_clusters: dict[str, list[str]] | None = None
) -> CrossMarketFeatures:
    """Convenience function to build cross-market features."""
    builder = CrossMarketFeatureBuilder()
    return builder.build(markets, semantic_clusters)
