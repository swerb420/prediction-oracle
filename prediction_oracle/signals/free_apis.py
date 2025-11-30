"""
Free news and data APIs for enhanced market intelligence.
No API keys required for basic usage!
"""

import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import Any
import httpx
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)


class WikipediaSignal(BaseModel):
    """Wikipedia page view trends - great for public attention."""
    topic: str
    views_today: int
    views_week_avg: int
    trend_ratio: float  # >1 = increasing attention


class RedditSignal(BaseModel):
    """Reddit discussion sentiment from free API."""
    subreddit: str
    topic: str
    post_count_24h: int
    avg_score: float
    sentiment_ratio: float  # positive / (positive + negative)
    top_keywords: list[str]


class GDELTSignal(BaseModel):
    """GDELT global news signal."""
    topic: str
    article_count: int
    unique_sources: int
    countries_covered: list[str]
    avg_tone: float
    tone_std: float
    top_sources: list[str]


class FreeAPIProvider:
    """
    Aggregates FREE APIs - no keys needed!
    
    Sources:
    - Wikipedia Pageviews (unlimited, public)
    - Reddit JSON (rate limited but free)
    - GDELT Project (unlimited, public)
    - Polymarket public APIs (unlimited)
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=15.0)
        self._cache: dict[str, tuple[datetime, Any]] = {}
        self._cache_ttl = timedelta(minutes=30)
        
    def _cache_key(self, prefix: str, query: str) -> str:
        return f"{prefix}:{hashlib.md5(query.encode()).hexdigest()[:12]}"
    
    def _get_cached(self, key: str) -> Any | None:
        if key in self._cache:
            ts, data = self._cache[key]
            if datetime.now() - ts < self._cache_ttl:
                return data
            del self._cache[key]
        return None
    
    def _set_cache(self, key: str, data: Any):
        self._cache[key] = (datetime.now(), data)
        if len(self._cache) > 500:
            oldest = min(self._cache.keys(), key=lambda k: self._cache[k][0])
            del self._cache[oldest]

    async def get_wikipedia_attention(self, topic: str) -> WikipediaSignal | None:
        """
        Get Wikipedia pageview trends - FREE, no API key!
        Great for measuring public attention on topics.
        """
        cache_key = self._cache_key("wiki", topic)
        if cached := self._get_cached(cache_key):
            return cached
            
        wiki_topic = topic.replace(" ", "_").title()
        today = datetime.now()
        week_ago = today - timedelta(days=7)
        
        url = (
            f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
            f"en.wikipedia/all-access/all-agents/{wiki_topic}/daily/"
            f"{week_ago.strftime('%Y%m%d')}/{today.strftime('%Y%m%d')}"
        )
        
        try:
            resp = await self.client.get(url)
            if resp.status_code != 200:
                return None
                
            data = resp.json()
            items = data.get("items", [])
            if not items:
                return None
                
            views = [item["views"] for item in items]
            views_today = views[-1] if views else 0
            views_week_avg = sum(views) / len(views) if views else 0
            
            signal = WikipediaSignal(
                topic=topic,
                views_today=views_today,
                views_week_avg=int(views_week_avg),
                trend_ratio=views_today / views_week_avg if views_week_avg > 0 else 1.0
            )
            self._set_cache(cache_key, signal)
            return signal
            
        except Exception as e:
            logger.debug(f"Wikipedia API error: {e}")
            return None

    async def get_reddit_signal(
        self, 
        topic: str, 
        subreddits: list[str] | None = None
    ) -> RedditSignal | None:
        """
        Get Reddit discussion data - FREE via JSON API!
        """
        cache_key = self._cache_key("reddit", topic)
        if cached := self._get_cached(cache_key):
            return cached
            
        if subreddits is None:
            subreddits = ["news", "worldnews", "politics"]
        
        all_posts = []
        
        for subreddit in subreddits[:2]:  # Limit to avoid rate limiting
            url = f"https://www.reddit.com/r/{subreddit}/search.json"
            params = {
                "q": topic,
                "sort": "new",
                "limit": 20,
                "t": "day",
                "restrict_sr": "true"
            }
            headers = {"User-Agent": "PredictionOracle/1.0"}
            
            try:
                resp = await self.client.get(url, params=params, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    posts = data.get("data", {}).get("children", [])
                    all_posts.extend([p["data"] for p in posts])
            except Exception:
                continue
            
            await asyncio.sleep(1)  # Rate limit respect
        
        if not all_posts:
            return None
            
        scores = [p.get("score", 0) for p in all_posts]
        upvote_ratios = [p.get("upvote_ratio", 0.5) for p in all_posts]
        
        all_words = []
        for p in all_posts:
            title = p.get("title", "").lower()
            words = [w for w in title.split() if len(w) > 4 and w.isalpha()]
            all_words.extend(words)
        
        word_counts: dict[str, int] = {}
        for w in all_words:
            word_counts[w] = word_counts.get(w, 0) + 1
        top_keywords = sorted(word_counts.keys(), key=lambda x: word_counts[x], reverse=True)[:10]
        
        signal = RedditSignal(
            subreddit=",".join(subreddits),
            topic=topic,
            post_count_24h=len(all_posts),
            avg_score=sum(scores) / len(scores) if scores else 0,
            sentiment_ratio=sum(upvote_ratios) / len(upvote_ratios) if upvote_ratios else 0.5,
            top_keywords=top_keywords
        )
        self._set_cache(cache_key, signal)
        return signal

    async def get_gdelt_signal(self, topic: str) -> GDELTSignal | None:
        """
        GDELT Project - FREE global event database!
        """
        cache_key = self._cache_key("gdelt", topic)
        if cached := self._get_cached(cache_key):
            return cached
            
        url = "https://api.gdeltproject.org/api/v2/doc/doc"
        params = {
            "query": topic,
            "mode": "artlist",
            "maxrecords": 50,
            "format": "json",
            "timespan": "24h"
        }
        
        try:
            resp = await self.client.get(url, params=params)
            if resp.status_code != 200:
                return None
                
            data = resp.json()
            articles = data.get("articles", [])
            
            if not articles:
                return None
            
            tones = []
            sources = set()
            countries = set()
            
            for article in articles:
                if "tone" in article:
                    tones.append(article["tone"])
                if "sourcecountry" in article:
                    countries.add(article["sourcecountry"])
                if "domain" in article:
                    sources.add(article["domain"])
            
            signal = GDELTSignal(
                topic=topic,
                article_count=len(articles),
                unique_sources=len(sources),
                countries_covered=list(countries)[:10],
                avg_tone=sum(tones) / len(tones) if tones else 0,
                tone_std=(sum((t - sum(tones)/len(tones))**2 for t in tones) / len(tones))**0.5 if len(tones) > 1 else 0,
                top_sources=list(sources)[:5]
            )
            self._set_cache(cache_key, signal)
            return signal
            
        except Exception as e:
            logger.debug(f"GDELT API error: {e}")
            return None

    async def get_polymarket_orderbook(self, token_id: str) -> dict | None:
        """Get Polymarket order book for smart money analysis."""
        cache_key = self._cache_key("poly_book", token_id)
        if cached := self._get_cached(cache_key):
            return cached
            
        url = f"https://clob.polymarket.com/book"
        params = {"token_id": token_id}
        
        try:
            resp = await self.client.get(url, params=params)
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            self._set_cache(cache_key, data)
            return data
        except Exception:
            return None

    async def get_polymarket_trades(self, token_id: str, limit: int = 50) -> list[dict] | None:
        """Get recent Polymarket trades for momentum analysis."""
        cache_key = self._cache_key("poly_trades", token_id)
        if cached := self._get_cached(cache_key):
            return cached
            
        url = f"https://clob.polymarket.com/trades"
        params = {"token_id": token_id, "limit": limit}
        
        try:
            resp = await self.client.get(url, params=params)
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            self._set_cache(cache_key, data)
            return data
        except Exception:
            return None

    async def close(self):
        await self.client.aclose()


# Global instance
free_api = FreeAPIProvider()
