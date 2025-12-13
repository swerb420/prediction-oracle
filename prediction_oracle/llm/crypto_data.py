"""
Crypto price data fetcher for ML predictions.
Fetches 15-minute OHLCV candles for BTC, ETH, SOL, XRP.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Literal

import httpx
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

CryptoSymbol = Literal["BTC", "ETH", "SOL", "XRP"]

# Binance symbol mappings
BINANCE_SYMBOLS = {
    "BTC": "BTCUSDT",
    "ETH": "ETHUSDT",
    "SOL": "SOLUSDT",
    "XRP": "XRPUSDT",
}


class Candle(BaseModel):
    """Single OHLCV candle."""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    

class CandleData(BaseModel):
    """Collection of candles for a symbol."""
    symbol: CryptoSymbol
    interval: str  # "15m", "1h", etc.
    candles: list[Candle]
    
    @property
    def closes(self) -> np.ndarray:
        return np.array([c.close for c in self.candles])
    
    @property
    def highs(self) -> np.ndarray:
        return np.array([c.high for c in self.candles])
    
    @property
    def lows(self) -> np.ndarray:
        return np.array([c.low for c in self.candles])
    
    @property
    def volumes(self) -> np.ndarray:
        return np.array([c.volume for c in self.candles])
    
    @property
    def opens(self) -> np.ndarray:
        return np.array([c.open for c in self.candles])


class CryptoDataFetcher:
    """
    Fetches crypto price data from Binance public API.
    No API key required for public market data.
    """
    
    def __init__(self, timeout: float = 30.0):
        self.base_url = "https://api.binance.com/api/v3"
        self.client = httpx.AsyncClient(timeout=timeout)
        
    async def get_candles(
        self,
        symbol: CryptoSymbol,
        interval: str = "15m",
        limit: int = 100
    ) -> CandleData:
        """
        Fetch OHLCV candles from Binance.
        
        Args:
            symbol: BTC, ETH, or SOL
            interval: Candle interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of candles (max 1000)
            
        Returns:
            CandleData with parsed candles
        """
        binance_symbol = BINANCE_SYMBOLS.get(symbol)
        if not binance_symbol:
            raise ValueError(f"Unknown symbol: {symbol}")
        
        try:
            resp = await self.client.get(
                f"{self.base_url}/klines",
                params={
                    "symbol": binance_symbol,
                    "interval": interval,
                    "limit": limit
                }
            )
            resp.raise_for_status()
            
            data = resp.json()
            candles = []
            
            for k in data:
                # Binance kline format:
                # [open_time, open, high, low, close, volume, close_time, ...]
                candles.append(Candle(
                    timestamp=datetime.fromtimestamp(k[0] / 1000),
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5])
                ))
            
            logger.info(f"Fetched {len(candles)} candles for {symbol} ({interval})")
            return CandleData(symbol=symbol, interval=interval, candles=candles)
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol} candles: {e}")
            raise
    
    async def get_current_price(self, symbol: CryptoSymbol) -> float:
        """Get current price for a symbol."""
        binance_symbol = BINANCE_SYMBOLS.get(symbol)
        if not binance_symbol:
            raise ValueError(f"Unknown symbol: {symbol}")
            
        resp = await self.client.get(
            f"{self.base_url}/ticker/price",
            params={"symbol": binance_symbol}
        )
        resp.raise_for_status()
        return float(resp.json()["price"])
    
    async def get_all_candles(
        self,
        interval: str = "15m",
        limit: int = 100
    ) -> dict[CryptoSymbol, CandleData]:
        """Fetch candles for all supported symbols in parallel."""
        symbols: list[CryptoSymbol] = ["BTC", "ETH", "SOL", "XRP"]
        
        tasks = [
            self.get_candles(sym, interval, limit)
            for sym in symbols
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        data = {}
        for sym, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.error(f"Failed to fetch {sym}: {result}")
            else:
                data[sym] = result
                
        return data
    
    async def close(self):
        await self.client.aclose()


# Singleton instance
_fetcher: CryptoDataFetcher | None = None


def get_fetcher() -> CryptoDataFetcher:
    """Get or create global fetcher instance."""
    global _fetcher
    if _fetcher is None:
        _fetcher = CryptoDataFetcher()
    return _fetcher


async def fetch_crypto_data(
    symbols: list[CryptoSymbol] | None = None,
    interval: str = "15m",
    limit: int = 100
) -> dict[CryptoSymbol, CandleData]:
    """
    Convenience function to fetch crypto data.
    
    Args:
        symbols: List of symbols to fetch, or None for all
        interval: Candle interval
        limit: Number of candles
        
    Returns:
        Dict mapping symbols to their candle data
    """
    fetcher = get_fetcher()
    
    if symbols is None:
        return await fetcher.get_all_candles(interval, limit)
    
    tasks = [fetcher.get_candles(sym, interval, limit) for sym in symbols]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    data = {}
    for sym, result in zip(symbols, results):
        if not isinstance(result, Exception):
            data[sym] = result
    
    return data
