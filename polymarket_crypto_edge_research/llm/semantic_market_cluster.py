"""
Grok-powered semantic market clustering.
Groups related/duplicate Polymarket markets for arbitrage detection.
"""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from core.logging_utils import get_logger
from llm.grok_client import GrokClient, GrokRequest, SYSTEM_PROMPTS

logger = get_logger(__name__)


class MarketCluster(BaseModel):
    """A cluster of semantically related markets."""
    
    cluster_id: str
    cluster_name: str
    central_theme: str
    
    # Markets in cluster
    market_ids: list[str] = Field(default_factory=list)
    questions: list[str] = Field(default_factory=list)
    
    # Confidence and metrics
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    similarity_score: float = Field(default=0.5, ge=0.0, le=1.0)
    
    # Price analysis
    price_spread: float = 0.0  # Max difference in yes prices
    arb_potential: float = 0.0  # Potential arbitrage profit
    
    # Metadata
    timestamp: datetime | None = None
    
    @property
    def is_duplicate(self) -> bool:
        """Check if cluster likely contains duplicate markets."""
        return self.similarity_score > 0.9 and len(self.market_ids) > 1
    
    @property
    def has_arb(self) -> bool:
        """Check if cluster has arbitrage potential."""
        return self.arb_potential > 0.02  # 2% threshold


class ClusteringResult(BaseModel):
    """Result of market clustering."""
    
    timestamp: datetime
    
    # Clusters
    clusters: list[MarketCluster] = Field(default_factory=list)
    
    # Summary stats
    n_markets_analyzed: int = 0
    n_clusters_found: int = 0
    n_duplicate_pairs: int = 0
    n_arb_opportunities: int = 0
    
    # Grok metadata
    tokens_used: int = 0
    latency_ms: float = 0.0
    
    @property
    def arb_clusters(self) -> list[MarketCluster]:
        """Get clusters with arbitrage potential."""
        return [c for c in self.clusters if c.has_arb]


class SemanticMarketClusterer:
    """
    Uses Grok to identify semantically related markets.
    
    Key features:
    - Groups duplicate/related markets
    - Detects arbitrage opportunities from price discrepancies
    - Batches requests to minimize API calls
    """
    
    PROMPT_TEMPLATE = """Analyze these Polymarket prediction markets and group them by semantic similarity.
Find markets that ask essentially the same question (duplicates) or are closely related.

Markets:
{markets_text}

Respond with ONLY this JSON structure:
{{
    "clusters": [
        {{
            "cluster_id": "cluster_1",
            "cluster_name": "descriptive name",
            "central_theme": "what unifies these markets",
            "market_ids": ["id1", "id2"],
            "similarity_score": 0.0-1.0,
            "confidence": 0.0-1.0
        }}
    ],
    "unclustered_ids": ["ids that don't fit any cluster"]
}}

Rules:
- Only group markets that are clearly related (similarity > 0.7)
- Duplicate markets should have similarity >= 0.95
- Single markets can be in their own cluster
- Be conservative - when in doubt, don't cluster"""

    def __init__(self, client: GrokClient | None = None):
        self.client = client
        self._cache: dict[str, tuple[datetime, ClusteringResult]] = {}
        self._cache_ttl_seconds = 1800  # 30 minutes
    
    async def cluster(
        self,
        markets: list[tuple[str, str, float]],  # (market_id, question, yes_price)
        force_refresh: bool = False
    ) -> ClusteringResult:
        """
        Cluster markets by semantic similarity.
        
        Args:
            markets: List of (market_id, question, yes_price) tuples
            force_refresh: Ignore cache
            
        Returns:
            ClusteringResult
        """
        from core.time_utils import now_utc
        
        if not markets:
            return ClusteringResult(
                timestamp=now_utc(),
                n_markets_analyzed=0
            )
        
        # Generate cache key from market IDs
        cache_key = "|".join(sorted(m[0] for m in markets))
        
        # Check cache
        if not force_refresh:
            cached = self._get_cached(cache_key)
            if cached is not None:
                return cached
        
        # Format markets for prompt
        markets_text = "\n".join(
            f"- [{market_id}] {question} (Yes: {price:.1%})"
            for market_id, question, price in markets[:50]  # Limit to 50
        )
        
        prompt = self.PROMPT_TEMPLATE.format(markets_text=markets_text)
        
        # Ensure client exists
        if self.client is None:
            from llm.grok_client import create_grok_client
            self.client = create_grok_client()
        
        # Make request
        request = GrokRequest(
            prompt=prompt,
            system_prompt=SYSTEM_PROMPTS["clustering"],
            max_tokens=800,
            temperature=0.1,
            request_type="clustering"
        )
        
        response = await self.client.complete(request)
        
        if not response.success or response.parsed is None:
            logger.warning(f"Market clustering failed: {response.error or response.parse_error}")
            return self._fallback_clustering(markets)
        
        # Parse response
        data = response.parsed
        timestamp = now_utc()
        
        # Build clusters
        clusters = []
        market_prices = {m[0]: m[2] for m in markets}
        market_questions = {m[0]: m[1] for m in markets}
        
        for cluster_data in data.get("clusters", []):
            market_ids = cluster_data.get("market_ids", [])
            
            # Get questions for this cluster
            questions = [market_questions.get(mid, "") for mid in market_ids]
            
            # Calculate price spread and arb potential
            prices = [market_prices.get(mid, 0.5) for mid in market_ids]
            if len(prices) >= 2:
                price_spread = max(prices) - min(prices)
                # Arb potential: if similarity is high but prices differ
                arb_potential = price_spread if cluster_data.get("similarity_score", 0) > 0.9 else 0.0
            else:
                price_spread = 0.0
                arb_potential = 0.0
            
            cluster = MarketCluster(
                cluster_id=cluster_data.get("cluster_id", f"cluster_{len(clusters)}"),
                cluster_name=cluster_data.get("cluster_name", "Unknown"),
                central_theme=cluster_data.get("central_theme", ""),
                market_ids=market_ids,
                questions=questions,
                confidence=float(cluster_data.get("confidence", 0.5)),
                similarity_score=float(cluster_data.get("similarity_score", 0.5)),
                price_spread=price_spread,
                arb_potential=arb_potential,
                timestamp=timestamp
            )
            clusters.append(cluster)
        
        # Count duplicates and arb opportunities
        n_duplicate_pairs = sum(
            len(c.market_ids) * (len(c.market_ids) - 1) // 2
            for c in clusters if c.is_duplicate
        )
        n_arb = sum(1 for c in clusters if c.has_arb)
        
        result = ClusteringResult(
            timestamp=timestamp,
            clusters=clusters,
            n_markets_analyzed=len(markets),
            n_clusters_found=len(clusters),
            n_duplicate_pairs=n_duplicate_pairs,
            n_arb_opportunities=n_arb,
            tokens_used=response.total_tokens,
            latency_ms=response.latency_ms
        )
        
        # Cache result
        self._cache[cache_key] = (timestamp, result)
        
        logger.info(
            f"Clustered {len(markets)} markets into {len(clusters)} clusters "
            f"({n_duplicate_pairs} potential duplicates, {n_arb} arb opportunities)"
        )
        
        return result
    
    def _get_cached(self, cache_key: str) -> ClusteringResult | None:
        """Get cached result if still valid."""
        from core.time_utils import now_utc
        
        if cache_key not in self._cache:
            return None
        
        cached_time, result = self._cache[cache_key]
        elapsed = (now_utc() - cached_time).total_seconds()
        
        if elapsed > self._cache_ttl_seconds:
            del self._cache[cache_key]
            return None
        
        return result
    
    def _fallback_clustering(
        self,
        markets: list[tuple[str, str, float]]
    ) -> ClusteringResult:
        """
        Fallback clustering using simple text matching.
        Less accurate but doesn't require API.
        """
        from core.time_utils import now_utc
        
        timestamp = now_utc()
        clusters = []
        
        # Simple word overlap similarity
        def word_similarity(q1: str, q2: str) -> float:
            words1 = set(q1.lower().split())
            words2 = set(q2.lower().split())
            
            if not words1 or not words2:
                return 0.0
            
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            
            return intersection / union if union > 0 else 0.0
        
        # Find similar pairs
        used_ids = set()
        cluster_id = 0
        
        for i, (id1, q1, p1) in enumerate(markets):
            if id1 in used_ids:
                continue
            
            cluster_markets = [(id1, q1, p1)]
            
            for id2, q2, p2 in markets[i+1:]:
                if id2 in used_ids:
                    continue
                
                similarity = word_similarity(q1, q2)
                if similarity >= 0.7:
                    cluster_markets.append((id2, q2, p2))
                    used_ids.add(id2)
            
            if len(cluster_markets) >= 2:
                prices = [m[2] for m in cluster_markets]
                price_spread = max(prices) - min(prices)
                
                clusters.append(MarketCluster(
                    cluster_id=f"cluster_{cluster_id}",
                    cluster_name=f"Similar to: {q1[:50]}...",
                    central_theme="Text similarity grouping",
                    market_ids=[m[0] for m in cluster_markets],
                    questions=[m[1] for m in cluster_markets],
                    confidence=0.5,
                    similarity_score=0.7,
                    price_spread=price_spread,
                    arb_potential=price_spread if price_spread > 0.02 else 0.0,
                    timestamp=timestamp
                ))
                cluster_id += 1
            
            used_ids.add(id1)
        
        return ClusteringResult(
            timestamp=timestamp,
            clusters=clusters,
            n_markets_analyzed=len(markets),
            n_clusters_found=len(clusters),
            n_duplicate_pairs=sum(len(c.market_ids) * (len(c.market_ids) - 1) // 2 for c in clusters),
            n_arb_opportunities=sum(1 for c in clusters if c.has_arb)
        )
    
    def get_arb_opportunities(
        self,
        result: ClusteringResult
    ) -> list[dict[str, Any]]:
        """
        Extract actionable arbitrage opportunities from clustering result.
        
        Returns:
            List of arb opportunity dicts
        """
        opportunities = []
        
        for cluster in result.arb_clusters:
            if len(cluster.market_ids) < 2:
                continue
            
            # Find best arb pair (highest price difference)
            opportunities.append({
                "cluster_id": cluster.cluster_id,
                "theme": cluster.central_theme,
                "market_ids": cluster.market_ids,
                "questions": cluster.questions,
                "price_spread": cluster.price_spread,
                "arb_potential": cluster.arb_potential,
                "similarity": cluster.similarity_score,
                "confidence": cluster.confidence,
                "recommendation": self._generate_recommendation(cluster)
            })
        
        # Sort by potential profit
        opportunities.sort(key=lambda x: x["arb_potential"], reverse=True)
        
        return opportunities
    
    def _generate_recommendation(self, cluster: MarketCluster) -> str:
        """Generate trading recommendation for arb cluster."""
        if cluster.arb_potential > 0.10:
            return f"STRONG ARB: Buy low ({min(cluster.price_spread):.1%}), sell high. Potential {cluster.arb_potential:.1%}"
        elif cluster.arb_potential > 0.05:
            return f"MODERATE ARB: Consider positions on spread. Potential {cluster.arb_potential:.1%}"
        else:
            return f"WEAK ARB: Monitor for spread widening. Current {cluster.arb_potential:.1%}"


async def cluster_markets(
    markets: list[tuple[str, str, float]]
) -> ClusteringResult:
    """Convenience function for one-off clustering."""
    clusterer = SemanticMarketClusterer()
    return await clusterer.cluster(markets)
