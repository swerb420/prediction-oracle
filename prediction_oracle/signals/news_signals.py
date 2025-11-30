"""
Free news and sentiment APIs for better market timing.
- NewsAPI.org (free tier: 100 requests/day)
- GNews.io (free tier: 100 requests/day)
- GDELT Project (completely free, no key needed!)
"""
import asyncio
import hashlib
import logging
from datetime import datetime, timedelta
from typing import Any

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class NewsArticle(BaseModel):
    title: str
    description: str | None
    source: str
    published_at: datetime
    url: str
    sentiment_score: float = 0.0  # -1 to 1
    relevance_score: float = 0.0  # 0 to 1


class NewsSignal(BaseModel):
    query: str
    articles: list[NewsArticle]
    avg_sentiment: float
    news_velocity: float  # articles per hour (momentum indicator)
    bullish_ratio: float  # % of positive articles
    cached_at: datetime


class NewsSignalProvider:
    """
    Aggregates news from multiple free sources.
    Use this to detect:
    - Breaking news that moves markets
    - Sentiment shifts before price moves
    - News velocity (more coverage = higher confidence)
    """
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=15.0)
        self._cache: dict[str, tuple[NewsSignal, datetime]] = {}
        self._cache_ttl = timedelta(minutes=15)
        
        # Free API keys - user sets in .env
        self.newsapi_key = ""
        self.gnews_key = ""
    
    def set_api_keys(self, newsapi_key: str = "", gnews_key: str = ""):
        """Set API keys if available."""
        self.newsapi_key = newsapi_key
        self.gnews_key = gnews_key
    
    def _cache_key(self, query: str) -> str:
        return hashlib.md5(query.lower().encode()).hexdigest()
    
    async def get_news_signal(self, query: str, hours_back: int = 24) -> NewsSignal:
        """
        Get aggregated news signal for a topic/market question.
        
        Args:
            query: Search query (e.g., "Trump election", "Bitcoin ETF")
            hours_back: How far back to search
            
        Returns:
            NewsSignal with sentiment, velocity, and articles
        """
        cache_key = self._cache_key(query)
        
        # Check cache
        if cache_key in self._cache:
            signal, cached_at = self._cache[cache_key]
            if datetime.utcnow() - cached_at < self._cache_ttl:
                return signal
        
        articles: list[NewsArticle] = []
        
        # Fetch from multiple sources in parallel
        tasks = []
        if self.newsapi_key:
            tasks.append(self._fetch_newsapi(query, hours_back))
        if self.gnews_key:
            tasks.append(self._fetch_gnews(query, hours_back))
        
        # Always try GDELT - completely free, no key needed!
        tasks.append(self._fetch_gdelt(query, hours_back))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in results:
            if isinstance(result, list):
                articles.extend(result)
        
        # Dedupe by URL
        seen_urls = set()
        unique_articles = []
        for article in articles:
            if article.url not in seen_urls:
                seen_urls.add(article.url)
                unique_articles.append(article)
        
        # Calculate sentiment using simple keyword analysis (free, no API needed)
        for article in unique_articles:
            article.sentiment_score = self._analyze_sentiment(
                f"{article.title} {article.description or ''}"
            )
        
        # Build signal
        if unique_articles:
            avg_sentiment = sum(a.sentiment_score for a in unique_articles) / len(unique_articles)
            bullish = sum(1 for a in unique_articles if a.sentiment_score > 0.1)
            bullish_ratio = bullish / len(unique_articles)
            
            # News velocity: articles per hour
            time_span = max(1, hours_back)
            news_velocity = len(unique_articles) / time_span
        else:
            avg_sentiment = 0.0
            bullish_ratio = 0.5
            news_velocity = 0.0
        
        signal = NewsSignal(
            query=query,
            articles=unique_articles[:20],  # Keep top 20
            avg_sentiment=avg_sentiment,
            news_velocity=news_velocity,
            bullish_ratio=bullish_ratio,
            cached_at=datetime.utcnow()
        )
        
        self._cache[cache_key] = (signal, datetime.utcnow())
        return signal
    
    async def _fetch_newsapi(self, query: str, hours_back: int) -> list[NewsArticle]:
        """NewsAPI.org - 100 free requests/day"""
        try:
            from_date = (datetime.utcnow() - timedelta(hours=hours_back)).isoformat()
            resp = await self.client.get(
                "https://newsapi.org/v2/everything",
                params={
                    "q": query,
                    "from": from_date,
                    "sortBy": "publishedAt",
                    "pageSize": 50,
                    "apiKey": self.newsapi_key,
                }
            )
            resp.raise_for_status()
            data = resp.json()
            
            articles = []
            for item in data.get("articles", []):
                try:
                    articles.append(NewsArticle(
                        title=item["title"] or "",
                        description=item.get("description"),
                        source=item.get("source", {}).get("name", "unknown"),
                        published_at=datetime.fromisoformat(
                            item["publishedAt"].replace("Z", "+00:00")
                        ),
                        url=item["url"],
                    ))
                except Exception:
                    continue
            return articles
        except Exception as e:
            logger.debug(f"NewsAPI fetch failed: {e}")
            return []
    
    async def _fetch_gnews(self, query: str, hours_back: int) -> list[NewsArticle]:
        """GNews.io - 100 free requests/day"""
        try:
            resp = await self.client.get(
                "https://gnews.io/api/v4/search",
                params={
                    "q": query,
                    "max": 50,
                    "token": self.gnews_key,
                }
            )
            resp.raise_for_status()
            data = resp.json()
            
            articles = []
            for item in data.get("articles", []):
                try:
                    articles.append(NewsArticle(
                        title=item["title"],
                        description=item.get("description"),
                        source=item.get("source", {}).get("name", "unknown"),
                        published_at=datetime.fromisoformat(
                            item["publishedAt"].replace("Z", "+00:00")
                        ),
                        url=item["url"],
                    ))
                except Exception:
                    continue
            return articles
        except Exception as e:
            logger.debug(f"GNews fetch failed: {e}")
            return []
    
    async def _fetch_gdelt(self, query: str, hours_back: int) -> list[NewsArticle]:
        """
        GDELT Project - Completely FREE, no API key needed!
        Best free news API for global events.
        """
        try:
            # GDELT DOC API - free and powerful
            resp = await self.client.get(
                "https://api.gdeltproject.org/api/v2/doc/doc",
                params={
                    "query": query,
                    "mode": "artlist",
                    "maxrecords": 50,
                    "format": "json",
                    "timespan": f"{hours_back}h",
                },
                timeout=20.0,
            )
            resp.raise_for_status()
            data = resp.json()
            
            articles = []
            for item in data.get("articles", []):
                try:
                    articles.append(NewsArticle(
                        title=item.get("title", ""),
                        description=None,
                        source=item.get("domain", "unknown"),
                        published_at=datetime.strptime(
                            item["seendate"], "%Y%m%dT%H%M%SZ"
                        ),
                        url=item["url"],
                    ))
                except Exception:
                    continue
            return articles
        except Exception as e:
            logger.debug(f"GDELT fetch failed: {e}")
            return []
    
    def _analyze_sentiment(self, text: str) -> float:
        """
        Simple keyword-based sentiment (free, no API).
        Returns -1 to 1.
        """
        text = text.lower()
        
        # Bullish/positive keywords
        bullish = [
            "surge", "soar", "rally", "gain", "win", "victory", "success",
            "approve", "pass", "support", "positive", "up", "rise", "jump",
            "breakthrough", "record", "best", "strong", "bullish", "boom",
            "accept", "agree", "confirm", "yes", "likely", "probable",
        ]
        
        # Bearish/negative keywords
        bearish = [
            "crash", "plunge", "fall", "drop", "lose", "defeat", "fail",
            "reject", "block", "oppose", "negative", "down", "sink", "dump",
            "crisis", "worst", "weak", "bearish", "bust", "collapse",
            "deny", "refuse", "unlikely", "improbable", "no", "cancel",
        ]
        
        bullish_count = sum(1 for word in bullish if word in text)
        bearish_count = sum(1 for word in bearish if word in text)
        
        total = bullish_count + bearish_count
        if total == 0:
            return 0.0
        
        return (bullish_count - bearish_count) / total
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# Singleton instance
news_provider = NewsSignalProvider()
