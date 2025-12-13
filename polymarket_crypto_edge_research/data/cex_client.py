"""
CEX client for Binance WebSocket + REST API.
Fetches BTC/ETH/SOL price data, candles, and orderbook.
"""

import asyncio
import json
from datetime import datetime
from typing import Callable, Literal

import httpx
import websockets
from tenacity import retry, stop_after_attempt, wait_exponential

from core.config import get_settings
from core.logging_utils import get_logger
from core.time_utils import from_timestamp_ms, now_utc
from .rate_limiter import AdaptiveRateLimiter
from .schemas import Candle, OrderBook, OrderBookLevel, PriceData, Trade

logger = get_logger(__name__)

Symbol = Literal["BTC", "ETH", "SOL"]

BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
}


class CEXClient:
    """Base class for CEX clients."""
    
    async def get_candles(
        self,
        symbol: Symbol,
        interval: str = "15m",
        limit: int = 100
    ) -> list[Candle]:
        raise NotImplementedError
    
    async def get_current_price(self, symbol: Symbol) -> PriceData:
        raise NotImplementedError
    
    async def get_orderbook(self, symbol: Symbol, depth: int = 20) -> OrderBook:
        raise NotImplementedError
    
    async def subscribe_trades(
        self,
        symbol: Symbol,
        callback: Callable[[Trade], None]
    ) -> None:
        raise NotImplementedError


class BinanceClient(CEXClient):
    """
    Binance API client with REST and WebSocket support.
    Rate-limited and retry-enabled.
    """
    
    def __init__(self):
        settings = get_settings()
        self.base_url = settings.binance_base_url
        self.ws_url = settings.binance_ws_url
        self.api_key = settings.binance_api_key
        self.secret = settings.binance_secret
        
        self._client = httpx.AsyncClient(timeout=30.0)
        self._rate_limiter = AdaptiveRateLimiter(
            requests_per_minute=1200,  # Binance limit
            requests_per_day=100000
        )
        self._ws_connections: dict[str, websockets.WebSocketClientProtocol] = {}
        self._running = False
    
    async def __aenter__(self) -> "BinanceClient":
        """Async context manager entry."""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def get_candles(
        self,
        symbol: Symbol,
        interval: str = "15m",
        limit: int = 100,
        start_time: int | None = None,
        end_time: int | None = None
    ) -> list[Candle]:
        """
        Fetch OHLCV candles from Binance.
        
        Args:
            symbol: BTC, ETH, or SOL
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles (max 1000)
            start_time: Start timestamp in ms
            end_time: End timestamp in ms
            
        Returns:
            List of Candle objects
        """
        await self._rate_limiter.acquire()
        
        binance_symbol = BINANCE_SYMBOLS.get(symbol)
        if not binance_symbol:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        params = {
            "symbol": binance_symbol,
            "interval": interval,
            "limit": min(limit, 1000)
        }
        
        if start_time:
            params["startTime"] = start_time
        if end_time:
            params["endTime"] = end_time
        
        resp = await self._client.get(
            f"{self.base_url}/api/v3/klines",
            params=params
        )
        resp.raise_for_status()
        self._rate_limiter.on_success()
        
        candles = []
        for k in resp.json():
            candles.append(Candle(
                symbol=symbol,
                timestamp=from_timestamp_ms(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                trades=int(k[8])
            ))
        
        logger.debug(f"Fetched {len(candles)} candles for {symbol}/{interval}")
        return candles
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def get_current_price(self, symbol: Symbol) -> PriceData:
        """Get current price and 24h stats."""
        await self._rate_limiter.acquire()
        
        binance_symbol = BINANCE_SYMBOLS.get(symbol)
        if not binance_symbol:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        resp = await self._client.get(
            f"{self.base_url}/api/v3/ticker/24hr",
            params={"symbol": binance_symbol}
        )
        resp.raise_for_status()
        self._rate_limiter.on_success()
        
        data = resp.json()
        
        return PriceData(
            symbol=symbol,
            timestamp=now_utc(),
            price=float(data["lastPrice"]),
            bid=float(data["bidPrice"]),
            ask=float(data["askPrice"]),
            volume_24h=float(data["volume"]),
            change_24h_pct=float(data["priceChangePercent"])
        )
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10)
    )
    async def get_orderbook(self, symbol: Symbol, depth: int = 20) -> OrderBook:
        """Get current orderbook snapshot."""
        await self._rate_limiter.acquire()
        
        binance_symbol = BINANCE_SYMBOLS.get(symbol)
        if not binance_symbol:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        resp = await self._client.get(
            f"{self.base_url}/api/v3/depth",
            params={"symbol": binance_symbol, "limit": min(depth, 1000)}
        )
        resp.raise_for_status()
        self._rate_limiter.on_success()
        
        data = resp.json()
        
        bids = [
            OrderBookLevel(price=float(b[0]), size=float(b[1]))
            for b in data["bids"]
        ]
        asks = [
            OrderBookLevel(price=float(a[0]), size=float(a[1]))
            for a in data["asks"]
        ]
        
        return OrderBook(
            symbol=symbol,
            timestamp=now_utc(),
            bids=bids,
            asks=asks
        )
    
    async def get_all_prices(self) -> dict[Symbol, PriceData]:
        """Get current prices for all symbols."""
        symbols: list[Symbol] = ["BTC", "ETH", "SOL"]
        tasks = [self.get_current_price(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        prices = {}
        for sym, result in zip(symbols, results):
            if not isinstance(result, Exception):
                prices[sym] = result
            else:
                logger.warning(f"Failed to fetch {sym} price: {result}")
        
        return prices
    
    async def get_all_candles(
        self,
        interval: str = "15m",
        limit: int = 100
    ) -> dict[Symbol, list[Candle]]:
        """Get candles for all symbols."""
        symbols: list[Symbol] = ["BTC", "ETH", "SOL"]
        tasks = [self.get_candles(s, interval, limit) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        candles = {}
        for sym, result in zip(symbols, results):
            if not isinstance(result, Exception):
                candles[sym] = result
            else:
                logger.warning(f"Failed to fetch {sym} candles: {result}")
        
        return candles
    
    async def subscribe_trades(
        self,
        symbol: Symbol,
        callback: Callable[[Trade], None]
    ) -> None:
        """
        Subscribe to real-time trade stream via WebSocket.
        
        Args:
            symbol: Symbol to subscribe to
            callback: Function to call with each trade
        """
        binance_symbol = BINANCE_SYMBOLS.get(symbol, "").lower()
        stream_name = f"{binance_symbol}@trade"
        
        ws_url = f"{self.ws_url}/{stream_name}"
        
        self._running = True
        
        while self._running:
            try:
                async with websockets.connect(ws_url) as ws:
                    self._ws_connections[symbol] = ws
                    logger.info(f"Connected to {symbol} trade stream")
                    
                    async for message in ws:
                        if not self._running:
                            break
                        
                        data = json.loads(message)
                        trade = Trade(
                            symbol=symbol,
                            timestamp=from_timestamp_ms(data["T"]),
                            price=float(data["p"]),
                            size=float(data["q"]),
                            side="buy" if data["m"] else "sell",
                            trade_id=str(data["t"])
                        )
                        callback(trade)
                        
            except websockets.ConnectionClosed:
                logger.warning(f"{symbol} WebSocket closed, reconnecting...")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error(f"{symbol} WebSocket error: {e}")
                await asyncio.sleep(5)
    
    async def close(self) -> None:
        """Close all connections."""
        self._running = False
        
        for symbol, ws in self._ws_connections.items():
            await ws.close()
        
        await self._client.aclose()


# Singleton instance
_cex_client: BinanceClient | None = None


def get_cex_client() -> BinanceClient:
    """Get or create global CEX client."""
    global _cex_client
    if _cex_client is None:
        _cex_client = BinanceClient()
    return _cex_client
