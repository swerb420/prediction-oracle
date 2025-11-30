"""External signal sources for enhanced market analysis."""

from .news_signals import NewsSignal, NewsSignalProvider, news_provider
from .polymarket_signals import SmartMoneySignal, PolymarketSignalProvider, polymarket_signals
from .social_signals import SocialBuzz, SocialSignalProvider, social_signals
from .free_apis import (
    FreeAPIProvider,
    WikipediaSignal,
    RedditSignal,
    GDELTSignal,
    free_api,
)
from .smart_screener import SmartScreener, ScreenedMarket

__all__ = [
    "NewsSignal",
    "NewsSignalProvider",
    "news_provider",
    "SmartMoneySignal",
    "PolymarketSignalProvider",
    "polymarket_signals",
    "SocialBuzz",
    "SocialSignalProvider",
    "social_signals",
    "FreeAPIProvider",
    "WikipediaSignal",
    "RedditSignal",
    "GDELTSignal",
    "free_api",
    "SmartScreener",
    "ScreenedMarket",
]
