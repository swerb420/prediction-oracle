"""
Smart market screener - multi-stage pipeline to minimize LLM costs.
Uses free signals first, then cheap LLM, then expensive only for best.
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any
from pydantic import BaseModel
import logging

from ..markets import Market, Venue
from .free_apis import FreeAPIProvider, WikipediaSignal, RedditSignal, GDELTSignal

logger = logging.getLogger(__name__)


class ScreenedMarket(BaseModel):
    """A market that passed screening with enriched data."""
    market_id: str
    question: str
    venue: str
    current_price: float
    close_time: datetime | None
    
    # Enrichment data
    attention_score: float = 0.0
    news_volume: int = 0
    sentiment_score: float = 0.5
    discussion_volume: int = 0
    
    # Quick LLM estimate (optional)
    quick_edge: float = 0.0
    quick_confidence: float = 0.0
    quick_rationale: str = ""
    
    # Final priority score
    priority_score: float = 0.0
    screen_reason: str = ""


class SmartScreener:
    """
    Multi-stage market screener that minimizes costs.
    
    Stage 1: Basic filters (liquidity, time, price range)
    Stage 2: Free signal enrichment (Wikipedia, Reddit, GDELT)
    Stage 3: Quick LLM screen (GPT-4o-mini)
    Stage 4: Deep analysis (Grok/Claude) - only top candidates
    """
    
    def __init__(
        self,
        free_api: FreeAPIProvider | None = None,
        min_volume: float = 100.0,
        min_hours_to_close: int = 6,
        max_days_to_close: int = 30,
        top_n_for_deep: int = 20
    ):
        self.free_api = free_api or FreeAPIProvider()
        self.min_volume = min_volume
        self.min_hours = min_hours_to_close
        self.max_days = max_days_to_close
        self.top_n = top_n_for_deep
        
    async def screen_markets(
        self,
        markets: list[Market],
        quick_llm = None  # Optional FastScreeningProvider
    ) -> list[ScreenedMarket]:
        """
        Run full screening pipeline.
        Returns prioritized list for deep analysis.
        """
        logger.info(f"Screening {len(markets)} markets")
        
        # Stage 1: Basic filters
        stage1 = self._basic_filter(markets)
        logger.info(f"Stage 1 (basic): {len(stage1)} passed")
        
        if not stage1:
            return []
        
        # Stage 2: Enrich with free signals
        stage2 = await self._enrich_signals(stage1)
        logger.info(f"Stage 2 (signals): {len(stage2)} enriched")
        
        # Stage 3: Optional quick LLM screen
        if quick_llm and len(stage2) > self.top_n:
            stage3 = await self._quick_llm_screen(stage2, quick_llm)
            logger.info(f"Stage 3 (quick LLM): {len(stage3)} scored")
        else:
            stage3 = stage2
        
        # Stage 4: Score and rank
        scored = self._score_and_rank(stage3)
        
        return scored[:self.top_n]
    
    def _basic_filter(self, markets: list[Market]) -> list[Market]:
        """Stage 1: Fast filters."""
        from datetime import timezone
        now = datetime.now(timezone.utc)
        filtered = []
        
        for market in markets:
            # Time to close
            if market.close_time:
                # Handle timezone-naive datetimes
                close_time = market.close_time
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=timezone.utc)
                
                hours = (close_time - now).total_seconds() / 3600
                if hours < self.min_hours or hours > self.max_days * 24:
                    continue
            
            # Volume check
            if market.volume_24h and market.volume_24h < self.min_volume:
                continue
            
            # Valid prices
            if not any(0.02 <= o.price <= 0.98 for o in market.outcomes):
                continue
                
            filtered.append(market)
        
        return filtered
    
    async def _enrich_signals(
        self, 
        markets: list[Market]
    ) -> list[ScreenedMarket]:
        """Stage 2: Add free signal data."""
        enriched = []
        
        # Process in small batches
        batch_size = 3
        for i in range(0, len(markets), batch_size):
            batch = markets[i:i + batch_size]
            
            tasks = [self._get_signals(m) for m in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for market, result in zip(batch, results):
                if isinstance(result, Exception):
                    result = {}
                
                # Get first outcome price as current price
                current_price = market.outcomes[0].price if market.outcomes else 0.5
                
                enriched.append(ScreenedMarket(
                    market_id=market.market_id,
                    question=market.question,
                    venue=market.venue.value,
                    current_price=current_price,
                    close_time=market.close_time,
                    attention_score=result.get("attention_score", 0),
                    news_volume=result.get("news_volume", 0),
                    sentiment_score=result.get("sentiment_score", 0.5),
                    discussion_volume=result.get("discussion_volume", 0),
                ))
            
            # Rate limit delay
            if i + batch_size < len(markets):
                await asyncio.sleep(0.5)
        
        return enriched
    
    async def _get_signals(self, market: Market) -> dict:
        """Get all free signals for a market."""
        topic = self._extract_topic(market.question)
        signals = {}
        
        # Skip Wikipedia for mock data (fake topics return 403)
        # Only query for real market questions
        if not market.question.startswith("Will event") and not market.question.startswith("Will crypto"):
            try:
                wiki = await self.free_api.get_wikipedia_attention(topic)
                if wiki:
                    signals["attention_score"] = min(wiki.trend_ratio / 2, 1.0)
            except Exception:
                pass
        
        # Skip Reddit (often blocked from VPS IPs)
        # GDELT is more reliable
        
        # Get GDELT news
        try:
            gdelt = await self.free_api.get_gdelt_signal(topic)
            if gdelt:
                signals["news_volume"] = gdelt.article_count
        except Exception:
            pass
        
        return signals
    
    async def _quick_llm_screen(
        self,
        markets: list[ScreenedMarket],
        quick_llm
    ) -> list[ScreenedMarket]:
        """Stage 3: Quick LLM screening."""
        # Prepare batch
        batch = [
            {
                "market_id": m.market_id,
                "question": m.question,
                "current_price": m.current_price
            }
            for m in markets
        ]
        
        # Run quick screen
        results = await quick_llm.batch_screen(batch)
        
        # Map results back
        result_map = {r[0]: r for r in results}
        
        for market in markets:
            if market.market_id in result_map:
                _, edge, conf, rationale = result_map[market.market_id]
                market.quick_edge = edge
                market.quick_confidence = conf
                market.quick_rationale = rationale
        
        return markets
    
    def _score_and_rank(
        self, 
        markets: list[ScreenedMarket]
    ) -> list[ScreenedMarket]:
        """Score and rank markets."""
        for market in markets:
            # Component scores
            attention = market.attention_score
            sentiment = abs(market.sentiment_score - 0.5) * 2  # Deviation from neutral
            volume = min((market.news_volume + market.discussion_volume) / 50, 1.0)
            
            # Edge opportunity from price
            price = market.current_price
            if 0.45 <= price <= 0.55:
                edge_opp = 0.3  # Contested
            elif price <= 0.15 or price >= 0.85:
                edge_opp = 0.5  # Longshot
            else:
                edge_opp = 0.2
            
            # Quick LLM edge if available
            llm_edge = abs(market.quick_edge) * market.quick_confidence
            
            # Time score
            if market.close_time:
                from datetime import timezone
                now = datetime.now(timezone.utc)
                close_time = market.close_time
                if close_time.tzinfo is None:
                    close_time = close_time.replace(tzinfo=timezone.utc)
                hours = (close_time - now).total_seconds() / 3600
                if 24 <= hours <= 168:
                    time_score = 1.0
                elif hours < 24:
                    time_score = 0.5
                else:
                    time_score = max(0.3, 1.0 - hours / 720)
            else:
                time_score = 0.5
            
            # Combined score
            market.priority_score = (
                attention * 0.15 +
                sentiment * 0.10 +
                volume * 0.15 +
                edge_opp * 0.20 +
                llm_edge * 0.25 +
                time_score * 0.15
            )
            
            # Reason
            reasons = []
            if attention > 0.6:
                reasons.append("high_attention")
            if llm_edge > 0.05:
                reasons.append("llm_edge")
            if edge_opp > 0.3:
                reasons.append("price_opportunity")
            if volume > 0.5:
                reasons.append("high_volume")
            
            market.screen_reason = ",".join(reasons) if reasons else "base"
        
        # Sort by priority
        markets.sort(key=lambda x: x.priority_score, reverse=True)
        
        return markets
    
    def _extract_topic(self, question: str) -> str:
        """Extract main topic from question."""
        noise = ["Will", "Does", "Is", "Are", "Has", "Have", "by", "before", "after", "?", "!"]
        
        topic = question
        for n in noise:
            topic = topic.replace(n, " ")
        
        words = topic.split()
        key_words = [w for w in words if len(w) > 3 and (w[0].isupper() or len(w) > 6)]
        
        return " ".join(key_words[:4]) if key_words else topic[:50]
    
    def _get_subreddits(self, category: str) -> list[str]:
        """Map category to subreddits."""
        mapping = {
            "politics": ["politics", "news", "worldnews"],
            "crypto": ["cryptocurrency", "bitcoin"],
            "sports": ["sports", "nba", "nfl"],
            "science": ["science", "technology"],
            "economics": ["economics", "finance"],
        }
        
        cat_lower = (category or "").lower()
        for key, subs in mapping.items():
            if key in cat_lower:
                return subs
        
        return ["news", "worldnews"]

    async def close(self):
        await self.free_api.close()
