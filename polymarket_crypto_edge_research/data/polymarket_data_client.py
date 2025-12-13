"""
Polymarket Data API client.
Fetches historical trades, price history, and market statistics.
"""

from datetime import datetime, timedelta
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.logging_utils import get_logger
from core.time_utils import now_utc, parse_iso_datetime, to_timestamp_ms
from .rate_limiter import AdaptiveRateLimiter
from .schemas import PolymarketTrade

logger = get_logger(__name__)


class PolymarketDataClient:
    """
    Client for Polymarket Data API.
    Fetches historical data, price timeseries, and trade history.
    """
    
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.polymarket_data_url
        self._client = httpx.AsyncClient(timeout=30.0)
        self._rate_limiter = AdaptiveRateLimiter(
            requests_per_minute=60,
            requests_per_day=5000
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def get_market_trades(
        self,
        market_id: str,
        limit: int = 100,
        before: datetime | None = None,
        after: datetime | None = None
    ) -> list[PolymarketTrade]:
        """
        Fetch trade history for a market.
        
        Args:
            market_id: Market/condition ID
            limit: Max trades to return
            before: Filter trades before this time
            after: Filter trades after this time
            
        Returns:
            List of PolymarketTrade objects
        """
        await self._rate_limiter.acquire()
        
        params: dict[str, Any] = {
            "market": market_id,
            "limit": limit
        }
        
        if before:
            params["before"] = to_timestamp_ms(before)
        if after:
            params["after"] = to_timestamp_ms(after)
        
        resp = await self._client.get(
            f"{self.base_url}/trades",
            params=params
        )
        resp.raise_for_status()
        self._rate_limiter.on_success()
        
        trades = []
        for t in resp.json():
            trades.append(self._parse_trade(t, market_id))
        
        logger.debug(f"Fetched {len(trades)} trades for market {market_id}")
        return trades
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def get_price_history(
        self,
        market_id: str,
        interval: str = "1h",
        limit: int = 100
    ) -> list[dict]:
        """
        Fetch price history timeseries.
        
        Args:
            market_id: Market/condition ID
            interval: Time interval (1m, 5m, 15m, 1h, 1d)
            limit: Number of data points
            
        Returns:
            List of price data points
        """
        await self._rate_limiter.acquire()
        
        resp = await self._client.get(
            f"{self.base_url}/prices",
            params={
                "market": market_id,
                "interval": interval,
                "limit": limit
            }
        )
        resp.raise_for_status()
        self._rate_limiter.on_success()
        
        return resp.json()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def get_market_stats(self, market_id: str) -> dict:
        """
        Fetch market statistics (volume, liquidity, trades).
        
        Args:
            market_id: Market/condition ID
            
        Returns:
            Dict with market statistics
        """
        await self._rate_limiter.acquire()
        
        resp = await self._client.get(
            f"{self.base_url}/markets/{market_id}/stats"
        )
        
        if resp.status_code == 404:
            return {}
        
        resp.raise_for_status()
        self._rate_limiter.on_success()
        
        return resp.json()
    
    async def get_recent_trades(
        self,
        market_id: str,
        minutes: int = 60
    ) -> list[PolymarketTrade]:
        """Get trades from the last N minutes."""
        after = now_utc() - timedelta(minutes=minutes)
        return await self.get_market_trades(
            market_id=market_id,
            limit=500,
            after=after
        )
    
    async def get_trade_flow(
        self,
        market_id: str,
        minutes: int = 10
    ) -> dict:
        """
        Analyze recent trade flow for a market.
        
        Returns:
            Dict with buy/sell volume, trade count, avg size
        """
        trades = await self.get_recent_trades(market_id, minutes)
        
        if not trades:
            return {
                "buy_volume": 0.0,
                "sell_volume": 0.0,
                "buy_count": 0,
                "sell_count": 0,
                "net_flow": 0.0,
                "avg_trade_size": 0.0,
                "vwap": 0.0
            }
        
        buy_volume = sum(t.size for t in trades if t.side == "buy")
        sell_volume = sum(t.size for t in trades if t.side == "sell")
        buy_count = sum(1 for t in trades if t.side == "buy")
        sell_count = sum(1 for t in trades if t.side == "sell")
        
        total_volume = buy_volume + sell_volume
        total_count = len(trades)
        
        # VWAP calculation
        vwap = 0.0
        if total_volume > 0:
            vwap = sum(t.price * t.size for t in trades) / total_volume
        
        return {
            "buy_volume": buy_volume,
            "sell_volume": sell_volume,
            "buy_count": buy_count,
            "sell_count": sell_count,
            "net_flow": buy_volume - sell_volume,
            "avg_trade_size": total_volume / total_count if total_count > 0 else 0,
            "vwap": vwap
        }
    
    async def get_price_momentum(
        self,
        market_id: str,
        lookback_minutes: int = 60
    ) -> dict:
        """
        Calculate price momentum metrics.
        
        Returns:
            Dict with price change, volatility, trend
        """
        try:
            history = await self.get_price_history(
                market_id,
                interval="1m",
                limit=lookback_minutes
            )
        except Exception:
            return {
                "price_change": 0.0,
                "volatility": 0.0,
                "trend": "flat"
            }
        
        if len(history) < 2:
            return {
                "price_change": 0.0,
                "volatility": 0.0,
                "trend": "flat"
            }
        
        prices = [h.get("price", 0) for h in history if h.get("price")]
        
        if len(prices) < 2:
            return {
                "price_change": 0.0,
                "volatility": 0.0,
                "trend": "flat"
            }
        
        # Price change
        price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] else 0
        
        # Volatility (std of returns)
        import numpy as np
        returns = np.diff(prices) / np.array(prices[:-1])
        volatility = float(np.std(returns)) if len(returns) > 0 else 0
        
        # Trend
        if price_change > 0.02:
            trend = "up"
        elif price_change < -0.02:
            trend = "down"
        else:
            trend = "flat"
        
        return {
            "price_change": price_change,
            "volatility": volatility,
            "trend": trend
        }
    
    def _parse_trade(self, data: dict, market_id: str) -> PolymarketTrade:
        """Parse API response into PolymarketTrade."""
        timestamp = now_utc()
        if "timestamp" in data:
            try:
                timestamp = parse_iso_datetime(str(data["timestamp"]))
            except Exception:
                pass
        
        return PolymarketTrade(
            market_id=market_id,
            outcome_id=data.get("outcome", ""),
            timestamp=timestamp,
            price=float(data.get("price", 0)),
            size=float(data.get("size", 0)),
            side=data.get("side", "buy"),
            maker=data.get("maker"),
            taker=data.get("taker")
        )
    
    async def close(self) -> None:
        """Close HTTP client."""
        await self._client.aclose()


# Singleton instance
_data_client: PolymarketDataClient | None = None


def get_data_client() -> PolymarketDataClient:
    """Get or create global Data client."""
    global _data_client
    if _data_client is None:
        _data_client = PolymarketDataClient()
    return _data_client
