"""
Social media signals - FREE APIs
- Reddit API (free with account)
"""
import asyncio
import logging
from datetime import datetime, timedelta

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class SocialBuzz(BaseModel):
    """Social media buzz metrics."""
    query: str
    reddit_posts_24h: int
    reddit_comments_24h: int
    reddit_sentiment: float  # -1 to 1
    reddit_trending_score: float  # 0 to 1
    total_mentions: int
    buzz_velocity: float  # mentions per hour
    cached_at: datetime


class SocialSignalProvider:
    """
    Free social media monitoring.
    Reddit via public JSON endpoints (no auth needed).
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=15.0)
        self._cache: dict[str, tuple[SocialBuzz, datetime]] = {}
        self._cache_ttl = timedelta(minutes=30)
    
    async def get_social_buzz(self, query: str) -> SocialBuzz:
        """Get social media buzz for a topic."""
        cache_key = query.lower()
        
        if cache_key in self._cache:
            buzz, cached_at = self._cache[cache_key]
            if datetime.utcnow() - cached_at < self._cache_ttl:
                return buzz
        
        # Fetch Reddit data
        reddit_data = await self._fetch_reddit(query)
        
        buzz = SocialBuzz(
            query=query,
            reddit_posts_24h=reddit_data["posts"],
            reddit_comments_24h=reddit_data["comments"],
            reddit_sentiment=reddit_data["sentiment"],
            reddit_trending_score=reddit_data["trending"],
            total_mentions=reddit_data["posts"] + reddit_data["comments"],
            buzz_velocity=(reddit_data["posts"] + reddit_data["comments"]) / 24,
            cached_at=datetime.utcnow(),
        )
        
        self._cache[cache_key] = (buzz, datetime.utcnow())
        return buzz
    
    async def _fetch_reddit(self, query: str) -> dict:
        """Search Reddit for mentions."""
        try:
            # Use Reddit's public JSON endpoints (no auth needed)
            search_url = "https://www.reddit.com/search.json"
            
            resp = await self.client.get(
                search_url,
                params={
                    "q": query,
                    "sort": "new",
                    "limit": 100,
                    "t": "day",  # Last 24 hours
                },
                headers={"User-Agent": "prediction-oracle/1.0"}
            )
            
            if resp.status_code != 200:
                return {"posts": 0, "comments": 0, "sentiment": 0, "trending": 0}
            
            data = resp.json()
            posts = data.get("data", {}).get("children", [])
            
            post_count = len(posts)
            comment_count = sum(p.get("data", {}).get("num_comments", 0) for p in posts)
            
            # Simple sentiment from titles
            bullish_keywords = ["bullish", "moon", "win", "yes", "likely", "confirmed"]
            bearish_keywords = ["bearish", "crash", "lose", "no", "unlikely", "denied"]
            
            bullish = 0
            bearish = 0
            for post in posts:
                title = post.get("data", {}).get("title", "").lower()
                bullish += sum(1 for k in bullish_keywords if k in title)
                bearish += sum(1 for k in bearish_keywords if k in title)
            
            total = bullish + bearish
            sentiment = (bullish - bearish) / total if total > 0 else 0
            
            # Trending score based on upvotes
            total_score = sum(p.get("data", {}).get("score", 0) for p in posts)
            trending = min(1.0, total_score / 1000)  # Normalize
            
            return {
                "posts": post_count,
                "comments": comment_count,
                "sentiment": sentiment,
                "trending": trending,
            }
        except Exception as e:
            logger.debug(f"Reddit fetch failed: {e}")
            return {"posts": 0, "comments": 0, "sentiment": 0, "trending": 0}
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# Singleton
social_signals = SocialSignalProvider()
