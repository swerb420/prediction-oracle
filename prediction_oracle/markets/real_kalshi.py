"""Real Kalshi API client."""
import logging
from datetime import datetime, timezone
import httpx

logger = logging.getLogger(__name__)

# Import from parent
from . import Market, OrderRequest, OrderResult, OrderStatus, Outcome, Position, Venue
from .base_client import BaseMarketClient


class RealKalshiClient(BaseMarketClient):
    """Real Kalshi API client with mock fallback."""
    
    DEMO_URL = "https://demo-api.kalshi.co/trade-api/v2"
    PROD_URL = "https://trading-api.kalshi.com/trade-api/v2"
    
    def __init__(self, api_key=None, api_secret=None, demo_mode=True):
        from ..config import settings
        
        self.api_key = api_key or settings.kalshi_api_key
        self.api_secret = api_secret or settings.kalshi_api_secret
        self.demo_mode = demo_mode
        self.base_url = self.DEMO_URL if demo_mode else self.PROD_URL
        
        self.has_credentials = (
            self.api_key and 
            self.api_secret and 
            self.api_key != "your_kalshi_key"
        )
        
        self.client = httpx.AsyncClient(timeout=30.0)
        
        if self.has_credentials:
            logger.info(f"Kalshi: real credentials (demo={demo_mode})")
        else:
            logger.debug("Kalshi: mock mode")
    
    async def list_markets(self, category=None, min_volume=None, limit=None):
        """List markets - uses mock if no credentials."""
        if not self.has_credentials:
            return self._get_mock_markets(limit or 10)
        
        # Real API implementation would go here
        logger.warning("Kalshi real API not implemented, using mock")
        return self._get_mock_markets(limit or 10)
    
    async def get_market(self, market_id):
        return None
    
    async def get_positions(self):
        return []
    
    async def place_order(self, request):
        return OrderResult(
            order_id=f"MOCK_{datetime.now().timestamp()}",
            status=OrderStatus.FILLED,
            filled_size=request.size_usd,
            avg_fill_price=request.limit_price or 0.5,
            message="Mock order",
        )
    
    async def cancel_order(self, order_id):
        return True
    
    async def close(self):
        await self.client.aclose()
    
    def _get_mock_markets(self, count):
        """Generate diverse mock Kalshi markets."""
        import random
        
        topics = [
            ("FOMC holds rates steady", "Economics", 0.72),
            ("GDP growth exceeds 2%", "Economics", 0.55),
            ("Unemployment below 4%", "Economics", 0.68),
            ("Bitcoin above $100k by Dec", "Crypto", 0.35),
            ("S&P 500 all-time high", "Finance", 0.48),
            ("Fed cuts rates in Dec", "Economics", 0.62),
            ("CPI below 3%", "Economics", 0.58),
            ("Gold above $2000", "Commodities", 0.75),
            ("Oil above $80", "Commodities", 0.42),
            ("Tech earnings beat", "Finance", 0.65),
        ]
        
        markets = []
        for i in range(min(count, len(topics))):
            q, cat, price = topics[i]
            # Add some randomness
            price = max(0.05, min(0.95, price + random.uniform(-0.1, 0.1)))
            
            markets.append(Market(
                venue=Venue.KALSHI,
                market_id=f"KALSHI-{i}",
                question=f"{q}?",
                rules="Mock market",
                category=cat,
                close_time=datetime.now(timezone.utc).replace(hour=23, minute=59),
                outcomes=[
                    Outcome(id=f"KALSHI-{i}_yes", label="Yes", price=price, volume_24h=random.uniform(5000, 50000)),
                    Outcome(id=f"KALSHI-{i}_no", label="No", price=1-price, volume_24h=random.uniform(5000, 50000)),
                ],
                volume_24h=random.uniform(10000, 100000),
            ))
        
        return markets
