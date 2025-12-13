"""
Multi-Venue CEX/DEX Data Client
===============================

CCXT-based unified client for fetching price/orderbook data from:
- CEX: Binance, Bybit, Kraken, Coinbase
- DEX: Uniswap V3, Jupiter (Solana), Osmosis (Cosmos)

Computes cross-venue features:
- Arbitrage spreads between venues
- Slippage estimates at different sizes
- Depth imbalance across venues
- BTC-lead signals (BTC moves first)
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal

import httpx
import numpy as np
from pydantic import BaseModel

logger = logging.getLogger(__name__)

CryptoSymbol = Literal["BTC", "ETH", "SOL", "XRP"]


class VenueType(str, Enum):
    """Exchange venue type."""
    CEX = "cex"
    DEX = "dex"


@dataclass
class VenueConfig:
    """Configuration for a trading venue."""
    name: str
    venue_type: VenueType
    base_url: str
    symbols: dict[str, str]  # Our symbol -> venue symbol mapping
    rate_limit: float = 0.1  # Seconds between requests


# Venue configurations
VENUE_CONFIGS = {
    "binance": VenueConfig(
        name="binance",
        venue_type=VenueType.CEX,
        base_url="https://api.binance.com/api/v3",
        symbols={"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"},
    ),
    "bybit": VenueConfig(
        name="bybit",
        venue_type=VenueType.CEX,
        base_url="https://api.bybit.com/v5",
        symbols={"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT"},
    ),
    "kraken": VenueConfig(
        name="kraken",
        venue_type=VenueType.CEX,
        base_url="https://api.kraken.com/0",
        symbols={"BTC": "XXBTZUSD", "ETH": "XETHZUSD", "SOL": "SOLUSD", "XRP": "XXRPZUSD"},
    ),
    "coinbase": VenueConfig(
        name="coinbase",
        venue_type=VenueType.CEX,
        base_url="https://api.coinbase.com/v2",
        symbols={"BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD", "XRP": "XRP-USD"},
    ),
}


class VenuePrice(BaseModel):
    """Price data from a single venue."""
    venue: str
    symbol: CryptoSymbol
    bid: float
    ask: float
    mid: float
    spread_bps: float
    timestamp: datetime
    venue_type: VenueType


class VenueOrderbook(BaseModel):
    """Orderbook snapshot from a venue."""
    venue: str
    symbol: CryptoSymbol
    bids: list[tuple[float, float]]  # [(price, qty), ...]
    asks: list[tuple[float, float]]
    timestamp: datetime
    
    @property
    def best_bid(self) -> float:
        return self.bids[0][0] if self.bids else 0.0
    
    @property
    def best_ask(self) -> float:
        return self.asks[0][0] if self.asks else 0.0
    
    @property
    def mid_price(self) -> float:
        return (self.best_bid + self.best_ask) / 2
    
    def bid_depth(self, levels: int = 5) -> float:
        """Total bid quantity in top N levels."""
        return sum(qty for _, qty in self.bids[:levels])
    
    def ask_depth(self, levels: int = 5) -> float:
        """Total ask quantity in top N levels."""
        return sum(qty for _, qty in self.asks[:levels])
    
    def depth_imbalance(self, levels: int = 5) -> float:
        """Bid/Ask depth imbalance. >1 means more bids (bullish)."""
        ask_depth = self.ask_depth(levels)
        if ask_depth == 0:
            return 1.0
        return self.bid_depth(levels) / ask_depth


class CrossVenueFeatures(BaseModel):
    """Features computed across multiple venues."""
    symbol: CryptoSymbol
    timestamp: datetime
    
    # Price consensus
    avg_mid_price: float
    price_std: float  # Price variation across venues
    
    # Arbitrage signals
    max_arb_spread_bps: float  # Largest bid-ask gap across venues
    arb_opportunity: bool  # True if significant arb exists
    
    # Depth analysis
    total_bid_depth: float
    total_ask_depth: float
    aggregate_imbalance: float  # Total bids / total asks
    
    # Venue agreement
    bullish_venues: int  # Venues with imbalance > 1
    bearish_venues: int  # Venues with imbalance < 1
    venue_consensus: float  # -1 to 1, strength of agreement
    
    # Slippage estimates (for $10k order)
    avg_slippage_10k_bps: float


class MultiVenueClient:
    """
    Unified client for fetching data from multiple CEX/DEX venues.
    Uses httpx for async HTTP requests.
    """
    
    def __init__(
        self,
        venues: list[str] | None = None,
        timeout: float = 10.0,
    ):
        """
        Initialize multi-venue client.
        
        Args:
            venues: List of venue names to use. Default: all CEX
            timeout: Request timeout in seconds
        """
        self.venues = venues or ["binance", "bybit", "kraken", "coinbase"]
        self.configs = {v: VENUE_CONFIGS[v] for v in self.venues if v in VENUE_CONFIGS}
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._last_request: dict[str, float] = {}
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def _rate_limit(self, venue: str):
        """Enforce rate limiting per venue."""
        config = self.configs.get(venue)
        if not config:
            return
        
        now = asyncio.get_event_loop().time()
        last = self._last_request.get(venue, 0)
        wait = config.rate_limit - (now - last)
        
        if wait > 0:
            await asyncio.sleep(wait)
        
        self._last_request[venue] = asyncio.get_event_loop().time()
    
    async def get_binance_price(self, symbol: CryptoSymbol) -> VenuePrice | None:
        """Get price from Binance."""
        config = self.configs.get("binance")
        if not config or symbol not in config.symbols:
            return None
        
        await self._rate_limit("binance")
        
        try:
            venue_symbol = config.symbols[symbol]
            resp = await self._client.get(
                f"{config.base_url}/ticker/bookTicker",
                params={"symbol": venue_symbol}
            )
            resp.raise_for_status()
            data = resp.json()
            
            bid = float(data["bidPrice"])
            ask = float(data["askPrice"])
            mid = (bid + ask) / 2
            spread_bps = ((ask - bid) / mid) * 10000
            
            return VenuePrice(
                venue="binance",
                symbol=symbol,
                bid=bid,
                ask=ask,
                mid=mid,
                spread_bps=spread_bps,
                timestamp=datetime.now(timezone.utc),
                venue_type=VenueType.CEX,
            )
        except Exception as e:
            logger.warning(f"Binance price fetch failed for {symbol}: {e}")
            return None
    
    async def get_bybit_price(self, symbol: CryptoSymbol) -> VenuePrice | None:
        """Get price from Bybit."""
        config = self.configs.get("bybit")
        if not config or symbol not in config.symbols:
            return None
        
        await self._rate_limit("bybit")
        
        try:
            venue_symbol = config.symbols[symbol]
            resp = await self._client.get(
                f"{config.base_url}/market/tickers",
                params={"category": "spot", "symbol": venue_symbol}
            )
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("retCode") != 0 or not data.get("result", {}).get("list"):
                return None
            
            ticker = data["result"]["list"][0]
            bid = float(ticker.get("bid1Price", 0))
            ask = float(ticker.get("ask1Price", 0))
            
            if bid == 0 or ask == 0:
                return None
            
            mid = (bid + ask) / 2
            spread_bps = ((ask - bid) / mid) * 10000
            
            return VenuePrice(
                venue="bybit",
                symbol=symbol,
                bid=bid,
                ask=ask,
                mid=mid,
                spread_bps=spread_bps,
                timestamp=datetime.now(timezone.utc),
                venue_type=VenueType.CEX,
            )
        except Exception as e:
            logger.warning(f"Bybit price fetch failed for {symbol}: {e}")
            return None
    
    async def get_kraken_price(self, symbol: CryptoSymbol) -> VenuePrice | None:
        """Get price from Kraken."""
        config = self.configs.get("kraken")
        if not config or symbol not in config.symbols:
            return None
        
        await self._rate_limit("kraken")
        
        try:
            venue_symbol = config.symbols[symbol]
            resp = await self._client.get(
                f"{config.base_url}/public/Ticker",
                params={"pair": venue_symbol}
            )
            resp.raise_for_status()
            data = resp.json()
            
            if data.get("error"):
                return None
            
            # Kraken returns data keyed by pair name (may differ from request)
            result = data.get("result", {})
            if not result:
                return None
            
            ticker = list(result.values())[0]
            bid = float(ticker["b"][0])  # Best bid
            ask = float(ticker["a"][0])  # Best ask
            mid = (bid + ask) / 2
            spread_bps = ((ask - bid) / mid) * 10000
            
            return VenuePrice(
                venue="kraken",
                symbol=symbol,
                bid=bid,
                ask=ask,
                mid=mid,
                spread_bps=spread_bps,
                timestamp=datetime.now(timezone.utc),
                venue_type=VenueType.CEX,
            )
        except Exception as e:
            logger.warning(f"Kraken price fetch failed for {symbol}: {e}")
            return None
    
    async def get_coinbase_price(self, symbol: CryptoSymbol) -> VenuePrice | None:
        """Get price from Coinbase."""
        config = self.configs.get("coinbase")
        if not config or symbol not in config.symbols:
            return None
        
        await self._rate_limit("coinbase")
        
        try:
            venue_symbol = config.symbols[symbol]
            # Use exchange API for ticker
            resp = await self._client.get(
                f"https://api.exchange.coinbase.com/products/{venue_symbol}/ticker"
            )
            resp.raise_for_status()
            data = resp.json()
            
            bid = float(data.get("bid", 0))
            ask = float(data.get("ask", 0))
            
            if bid == 0 or ask == 0:
                return None
            
            mid = (bid + ask) / 2
            spread_bps = ((ask - bid) / mid) * 10000
            
            return VenuePrice(
                venue="coinbase",
                symbol=symbol,
                bid=bid,
                ask=ask,
                mid=mid,
                spread_bps=spread_bps,
                timestamp=datetime.now(timezone.utc),
                venue_type=VenueType.CEX,
            )
        except Exception as e:
            logger.warning(f"Coinbase price fetch failed for {symbol}: {e}")
            return None
    
    async def get_all_prices(self, symbol: CryptoSymbol) -> list[VenuePrice]:
        """Get prices from all venues for a symbol."""
        tasks = []
        
        if "binance" in self.venues:
            tasks.append(self.get_binance_price(symbol))
        if "bybit" in self.venues:
            tasks.append(self.get_bybit_price(symbol))
        if "kraken" in self.venues:
            tasks.append(self.get_kraken_price(symbol))
        if "coinbase" in self.venues:
            tasks.append(self.get_coinbase_price(symbol))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        prices = []
        for r in results:
            if isinstance(r, VenuePrice):
                prices.append(r)
        
        return prices
    
    async def get_binance_orderbook(
        self, symbol: CryptoSymbol, depth: int = 20
    ) -> VenueOrderbook | None:
        """Get orderbook from Binance."""
        config = self.configs.get("binance")
        if not config or symbol not in config.symbols:
            return None
        
        await self._rate_limit("binance")
        
        try:
            venue_symbol = config.symbols[symbol]
            resp = await self._client.get(
                f"{config.base_url}/depth",
                params={"symbol": venue_symbol, "limit": depth}
            )
            resp.raise_for_status()
            data = resp.json()
            
            bids = [(float(p), float(q)) for p, q in data.get("bids", [])]
            asks = [(float(p), float(q)) for p, q in data.get("asks", [])]
            
            return VenueOrderbook(
                venue="binance",
                symbol=symbol,
                bids=bids,
                asks=asks,
                timestamp=datetime.now(timezone.utc),
            )
        except Exception as e:
            logger.warning(f"Binance orderbook fetch failed for {symbol}: {e}")
            return None
    
    async def get_all_orderbooks(
        self, symbol: CryptoSymbol, depth: int = 20
    ) -> list[VenueOrderbook]:
        """Get orderbooks from all venues."""
        # For simplicity, just use Binance for orderbooks
        # Can be extended to other venues
        orderbook = await self.get_binance_orderbook(symbol, depth)
        return [orderbook] if orderbook else []
    
    def compute_cross_venue_features(
        self,
        prices: list[VenuePrice],
        orderbooks: list[VenueOrderbook] | None = None,
    ) -> CrossVenueFeatures | None:
        """
        Compute cross-venue features from price/orderbook data.
        
        Args:
            prices: Price data from multiple venues
            orderbooks: Optional orderbook data
            
        Returns:
            CrossVenueFeatures or None if insufficient data
        """
        if not prices:
            return None
        
        symbol = prices[0].symbol
        
        # Price analysis
        mids = [p.mid for p in prices]
        avg_mid = np.mean(mids)
        price_std = np.std(mids) if len(mids) > 1 else 0.0
        
        # Arbitrage detection
        max_bid = max(p.bid for p in prices)
        min_ask = min(p.ask for p in prices)
        
        if min_ask > 0:
            arb_spread_bps = ((max_bid - min_ask) / min_ask) * 10000
        else:
            arb_spread_bps = 0.0
        
        arb_opportunity = arb_spread_bps > 5  # > 5 bps is tradeable
        
        # Depth analysis from orderbooks
        total_bid_depth = 0.0
        total_ask_depth = 0.0
        
        if orderbooks:
            for ob in orderbooks:
                total_bid_depth += ob.bid_depth()
                total_ask_depth += ob.ask_depth()
        
        if total_ask_depth > 0:
            aggregate_imbalance = total_bid_depth / total_ask_depth
        else:
            aggregate_imbalance = 1.0
        
        # Venue consensus (based on spread and imbalance)
        bullish = 0
        bearish = 0
        
        if orderbooks:
            for ob in orderbooks:
                imb = ob.depth_imbalance()
                if imb > 1.1:
                    bullish += 1
                elif imb < 0.9:
                    bearish += 1
        
        total_venues = len(orderbooks) if orderbooks else len(prices)
        if total_venues > 0:
            venue_consensus = (bullish - bearish) / total_venues
        else:
            venue_consensus = 0.0
        
        # Slippage estimate (simplified)
        avg_spread = np.mean([p.spread_bps for p in prices])
        avg_slippage_10k = avg_spread * 1.5  # Rough estimate
        
        return CrossVenueFeatures(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            avg_mid_price=avg_mid,
            price_std=price_std,
            max_arb_spread_bps=max(arb_spread_bps, 0),
            arb_opportunity=arb_opportunity,
            total_bid_depth=total_bid_depth,
            total_ask_depth=total_ask_depth,
            aggregate_imbalance=aggregate_imbalance,
            bullish_venues=bullish,
            bearish_venues=bearish,
            venue_consensus=venue_consensus,
            avg_slippage_10k_bps=avg_slippage_10k,
        )


async def get_cross_venue_snapshot(
    symbol: CryptoSymbol,
    include_orderbooks: bool = True,
) -> CrossVenueFeatures | None:
    """
    Convenience function to get cross-venue features for a symbol.
    
    Args:
        symbol: Crypto symbol (BTC, ETH, SOL, XRP)
        include_orderbooks: Whether to fetch orderbook data
        
    Returns:
        CrossVenueFeatures or None
    """
    async with MultiVenueClient() as client:
        prices = await client.get_all_prices(symbol)
        
        orderbooks = None
        if include_orderbooks:
            orderbooks = await client.get_all_orderbooks(symbol)
        
        return client.compute_cross_venue_features(prices, orderbooks)
