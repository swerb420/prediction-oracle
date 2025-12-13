"""
Multi-venue CEX/DEX client using CCXT.
Supports Binance, Bybit, Kraken, Coinbase (CEX) and 
Uniswap, Jupiter, Osmosis (DEX) via aggregators.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field

from core.config import get_settings
from core.logging_utils import get_logger
from core.time_utils import now_utc

logger = get_logger(__name__)

# Venue types
CEXVenue = Literal["binance", "bybit", "kraken", "coinbase"]
DEXVenue = Literal["uniswap", "jupiter", "osmosis"]
Venue = CEXVenue | DEXVenue
Symbol = Literal["BTC", "ETH", "SOL", "XRP"]

CEX_VENUES: list[CEXVenue] = ["binance", "bybit", "kraken", "coinbase"]
DEX_VENUES: list[DEXVenue] = ["uniswap", "jupiter", "osmosis"]
ALL_VENUES: list[Venue] = CEX_VENUES + DEX_VENUES


class VenuePrice(BaseModel):
    """Price data from a specific venue."""
    venue: str
    symbol: str
    price: float
    bid: float | None = None
    ask: float | None = None
    spread_pct: float = 0.0
    volume_24h: float = 0.0
    timestamp: datetime = Field(default_factory=now_utc)


class OrderBookLevel(BaseModel):
    """Single order book level."""
    price: float
    size: float


class VenueOrderBook(BaseModel):
    """Order book from a venue."""
    venue: str
    symbol: str
    bids: list[OrderBookLevel] = Field(default_factory=list)
    asks: list[OrderBookLevel] = Field(default_factory=list)
    timestamp: datetime = Field(default_factory=now_utc)
    
    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.bids[0].price + self.asks[0].price) / 2
        return 0.0
    
    @property
    def spread_pct(self) -> float:
        if self.bids and self.asks and self.mid_price > 0:
            return (self.asks[0].price - self.bids[0].price) / self.mid_price * 100
        return 0.0
    
    @property
    def bid_depth_usd(self) -> float:
        """Total USD value of top 10 bids."""
        return sum(lvl.price * lvl.size for lvl in self.bids[:10])
    
    @property
    def ask_depth_usd(self) -> float:
        """Total USD value of top 10 asks."""
        return sum(lvl.price * lvl.size for lvl in self.asks[:10])
    
    @property
    def imbalance(self) -> float:
        """Order book imbalance: (bid_depth - ask_depth) / total."""
        total = self.bid_depth_usd + self.ask_depth_usd
        if total == 0:
            return 0.0
        return (self.bid_depth_usd - self.ask_depth_usd) / total


class FundingData(BaseModel):
    """Funding rate data for perpetuals."""
    venue: str
    symbol: str
    funding_rate: float  # As decimal (0.0001 = 0.01%)
    next_funding_time: datetime | None = None
    open_interest: float = 0.0
    timestamp: datetime = Field(default_factory=now_utc)


class ArbOpportunity(BaseModel):
    """Arbitrage opportunity between venues."""
    symbol: str
    buy_venue: str
    sell_venue: str
    buy_price: float
    sell_price: float
    spread_pct: float
    net_profit_pct: float  # After fees
    timestamp: datetime = Field(default_factory=now_utc)
    is_actionable: bool = False  # True if > min threshold


class VenueMetrics(BaseModel):
    """Aggregated metrics across venues for a symbol."""
    symbol: str
    timestamp: datetime = Field(default_factory=now_utc)
    
    # Price metrics
    prices: dict[str, float] = Field(default_factory=dict)
    avg_price: float = 0.0
    price_std: float = 0.0
    max_spread_pct: float = 0.0
    
    # Liquidity
    total_bid_depth: float = 0.0
    total_ask_depth: float = 0.0
    avg_imbalance: float = 0.0
    
    # Funding (CEX only)
    avg_funding_rate: float = 0.0
    funding_divergence: float = 0.0  # Std of funding rates
    total_open_interest: float = 0.0
    
    # Arb
    best_arb_pct: float = 0.0
    arb_opportunities: list[ArbOpportunity] = Field(default_factory=list)
    
    # Clean score components
    is_low_arb: bool = False  # < 0.1% spread
    is_stable_funding: bool = False  # ±0.01%
    is_low_vol_spike: bool = False  # < 2x avg
    is_high_liquidity: bool = False  # > $10M depth
    
    @property
    def clean_score(self) -> float:
        """0-1 score for trading cleanliness."""
        score = 0.0
        if self.is_low_arb:
            score += 0.25
        if self.is_stable_funding:
            score += 0.25
        if self.is_low_vol_spike:
            score += 0.25
        if self.is_high_liquidity:
            score += 0.25
        return score


class MultiVenueClient:
    """
    Unified client for multiple CEX and DEX venues.
    Uses CCXT for CEX, custom APIs for DEX.
    """
    
    # Fee estimates per venue (maker/taker avg)
    VENUE_FEES: dict[str, float] = {
        "binance": 0.001,  # 0.1%
        "bybit": 0.001,
        "kraken": 0.0016,
        "coinbase": 0.004,
        "uniswap": 0.003,  # 0.3% pool fee
        "jupiter": 0.0025,
        "osmosis": 0.002,
    }
    
    # Symbol mappings per venue
    SYMBOL_MAP: dict[str, dict[str, str]] = {
        "binance": {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT", "XRP": "XRP/USDT"},
        "bybit": {"BTC": "BTC/USDT", "ETH": "ETH/USDT", "SOL": "SOL/USDT", "XRP": "XRP/USDT"},
        "kraken": {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD", "XRP": "XRP/USD"},
        "coinbase": {"BTC": "BTC/USD", "ETH": "ETH/USD", "SOL": "SOL/USD", "XRP": "XRP/USD"},
        "uniswap": {"ETH": "WETH/USDC", "SOL": None, "BTC": "WBTC/USDC", "XRP": None},
        "jupiter": {"SOL": "SOL/USDC", "ETH": None, "BTC": None, "XRP": None},
        "osmosis": {"XRP": None, "ETH": None, "SOL": None, "BTC": None},  # Limited
    }
    
    def __init__(self):
        self._http = httpx.AsyncClient(timeout=30.0)
        self._ccxt_exchanges: dict[str, Any] = {}
        self._rate_limits: dict[str, float] = {}  # Last call timestamps
        self._cache: dict[str, tuple[Any, datetime]] = {}
        self._cache_ttl = 5  # seconds
    
    async def __aenter__(self) -> "MultiVenueClient":
        await self._init_exchanges()
        return self
    
    async def __aexit__(self, *args) -> None:
        await self.close()
    
    async def _init_exchanges(self) -> None:
        """Initialize CCXT exchange instances."""
        try:
            import ccxt.async_support as ccxt
            
            settings = get_settings()
            
            # Initialize CEX exchanges
            self._ccxt_exchanges["binance"] = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"}
            })
            
            self._ccxt_exchanges["bybit"] = ccxt.bybit({
                "enableRateLimit": True,
            })
            
            self._ccxt_exchanges["kraken"] = ccxt.kraken({
                "enableRateLimit": True,
            })
            
            self._ccxt_exchanges["coinbase"] = ccxt.coinbase({
                "enableRateLimit": True,
            })
            
            logger.info(f"Initialized {len(self._ccxt_exchanges)} CCXT exchanges")
            
        except ImportError:
            logger.warning("CCXT not installed, using HTTP fallbacks")
    
    async def close(self) -> None:
        """Close all connections."""
        for exchange in self._ccxt_exchanges.values():
            await exchange.close()
        await self._http.aclose()
    
    def _get_cache(self, key: str) -> Any | None:
        """Get cached value if not expired."""
        if key in self._cache:
            value, ts = self._cache[key]
            if (now_utc() - ts).total_seconds() < self._cache_ttl:
                return value
        return None
    
    def _set_cache(self, key: str, value: Any) -> None:
        """Set cache value."""
        self._cache[key] = (value, now_utc())
    
    async def get_price(self, venue: Venue, symbol: Symbol) -> VenuePrice | None:
        """Get current price from a venue."""
        cache_key = f"price:{venue}:{symbol}"
        cached = self._get_cache(cache_key)
        if cached:
            return cached
        
        try:
            if venue in self._ccxt_exchanges:
                result = await self._get_ccxt_price(venue, symbol)
            else:
                result = await self._get_dex_price(venue, symbol)
            
            if result:
                self._set_cache(cache_key, result)
            return result
            
        except Exception as e:
            logger.error(f"Price fetch error {venue}/{symbol}: {e}")
            return None
    
    async def _get_ccxt_price(self, venue: CEXVenue, symbol: Symbol) -> VenuePrice | None:
        """Get price via CCXT."""
        exchange = self._ccxt_exchanges.get(venue)
        if not exchange:
            return None
        
        pair = self.SYMBOL_MAP.get(venue, {}).get(symbol)
        if not pair:
            return None
        
        try:
            ticker = await exchange.fetch_ticker(pair)
            
            bid = ticker.get("bid", 0) or 0
            ask = ticker.get("ask", 0) or 0
            mid = (bid + ask) / 2 if bid and ask else ticker.get("last", 0)
            spread = (ask - bid) / mid * 100 if mid > 0 else 0
            
            return VenuePrice(
                venue=venue,
                symbol=symbol,
                price=mid,
                bid=bid,
                ask=ask,
                spread_pct=spread,
                volume_24h=ticker.get("quoteVolume", 0) or 0
            )
        except Exception as e:
            logger.debug(f"CCXT price error {venue}/{symbol}: {e}")
            return None
    
    async def _get_dex_price(self, venue: DEXVenue, symbol: Symbol) -> VenuePrice | None:
        """Get DEX price via CoinGecko or direct API."""
        # Use CoinGecko for DEX prices (free tier)
        coingecko_ids = {
            "BTC": "bitcoin",
            "ETH": "ethereum", 
            "SOL": "solana",
            "XRP": "ripple"
        }
        
        cg_id = coingecko_ids.get(symbol)
        if not cg_id:
            return None
        
        try:
            resp = await self._http.get(
                f"https://api.coingecko.com/api/v3/simple/price",
                params={
                    "ids": cg_id,
                    "vs_currencies": "usd",
                    "include_24hr_vol": "true"
                }
            )
            resp.raise_for_status()
            data = resp.json()
            
            if cg_id in data:
                return VenuePrice(
                    venue=venue,
                    symbol=symbol,
                    price=data[cg_id]["usd"],
                    volume_24h=data[cg_id].get("usd_24h_vol", 0)
                )
        except Exception as e:
            logger.debug(f"DEX price error {venue}/{symbol}: {e}")
        
        return None
    
    async def get_orderbook(
        self,
        venue: Venue,
        symbol: Symbol,
        depth: int = 10
    ) -> VenueOrderBook | None:
        """Get order book from venue."""
        if venue not in self._ccxt_exchanges:
            # DEX order books need specialized APIs
            return None
        
        exchange = self._ccxt_exchanges[venue]
        pair = self.SYMBOL_MAP.get(venue, {}).get(symbol)
        if not pair:
            return None
        
        try:
            book = await exchange.fetch_order_book(pair, limit=depth)
            
            return VenueOrderBook(
                venue=venue,
                symbol=symbol,
                bids=[OrderBookLevel(price=b[0], size=b[1]) for b in book["bids"][:depth]],
                asks=[OrderBookLevel(price=a[0], size=a[1]) for a in book["asks"][:depth]]
            )
        except Exception as e:
            logger.debug(f"Orderbook error {venue}/{symbol}: {e}")
            return None
    
    async def get_funding(self, venue: CEXVenue, symbol: Symbol) -> FundingData | None:
        """Get funding rate data (perps only)."""
        if venue not in self._ccxt_exchanges:
            return None
        
        exchange = self._ccxt_exchanges[venue]
        
        # Funding is typically on perp contracts
        perp_symbols = {
            "binance": {"BTC": "BTC/USDT:USDT", "ETH": "ETH/USDT:USDT", "SOL": "SOL/USDT:USDT"},
            "bybit": {"BTC": "BTC/USDT:USDT", "ETH": "ETH/USDT:USDT", "SOL": "SOL/USDT:USDT"},
        }
        
        pair = perp_symbols.get(venue, {}).get(symbol)
        if not pair:
            return None
        
        try:
            # Try to get funding rate
            funding = await exchange.fetch_funding_rate(pair)
            
            return FundingData(
                venue=venue,
                symbol=symbol,
                funding_rate=funding.get("fundingRate", 0) or 0,
                next_funding_time=funding.get("fundingDatetime"),
                open_interest=funding.get("openInterest", 0) or 0
            )
        except Exception as e:
            logger.debug(f"Funding error {venue}/{symbol}: {e}")
            return None
    
    async def get_all_prices(self, symbol: Symbol) -> dict[Venue, VenuePrice]:
        """Get prices from all venues for a symbol."""
        tasks = []
        venues = []
        
        for venue in ALL_VENUES:
            if self.SYMBOL_MAP.get(venue, {}).get(symbol):
                tasks.append(self.get_price(venue, symbol))
                venues.append(venue)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        prices = {}
        for venue, result in zip(venues, results):
            if isinstance(result, VenuePrice):
                prices[venue] = result
        
        return prices
    
    async def get_all_orderbooks(
        self,
        symbol: Symbol,
        depth: int = 10
    ) -> dict[Venue, VenueOrderBook]:
        """Get order books from all CEX venues."""
        tasks = []
        venues = []
        
        for venue in CEX_VENUES:
            if self.SYMBOL_MAP.get(venue, {}).get(symbol):
                tasks.append(self.get_orderbook(venue, symbol, depth))
                venues.append(venue)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        books = {}
        for venue, result in zip(venues, results):
            if isinstance(result, VenueOrderBook):
                books[venue] = result
        
        return books
    
    async def get_all_funding(self, symbol: Symbol) -> dict[CEXVenue, FundingData]:
        """Get funding rates from all CEX venues."""
        tasks = []
        venues = []
        
        for venue in CEX_VENUES:
            tasks.append(self.get_funding(venue, symbol))
            venues.append(venue)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        funding = {}
        for venue, result in zip(venues, results):
            if isinstance(result, FundingData):
                funding[venue] = result
        
        return funding
    
    def find_arb_opportunities(
        self,
        prices: dict[Venue, VenuePrice],
        min_profit_pct: float = 0.1
    ) -> list[ArbOpportunity]:
        """Find arbitrage opportunities between venues."""
        opportunities = []
        venues = list(prices.keys())
        
        for i, buy_venue in enumerate(venues):
            for sell_venue in venues[i+1:]:
                buy_price = prices[buy_venue].ask or prices[buy_venue].price
                sell_price = prices[sell_venue].bid or prices[sell_venue].price
                
                # Check both directions
                for bv, sv, bp, sp in [
                    (buy_venue, sell_venue, buy_price, sell_price),
                    (sell_venue, buy_venue, prices[sell_venue].ask or prices[sell_venue].price,
                     prices[buy_venue].bid or prices[buy_venue].price)
                ]:
                    if bp <= 0 or sp <= 0:
                        continue
                    
                    spread_pct = (sp - bp) / bp * 100
                    fees = self.VENUE_FEES.get(bv, 0.001) + self.VENUE_FEES.get(sv, 0.001)
                    net_profit = spread_pct - (fees * 100)
                    
                    if net_profit > min_profit_pct:
                        opportunities.append(ArbOpportunity(
                            symbol=prices[bv].symbol,
                            buy_venue=bv,
                            sell_venue=sv,
                            buy_price=bp,
                            sell_price=sp,
                            spread_pct=spread_pct,
                            net_profit_pct=net_profit,
                            is_actionable=net_profit > 0.2
                        ))
        
        return sorted(opportunities, key=lambda x: x.net_profit_pct, reverse=True)
    
    async def get_venue_metrics(self, symbol: Symbol) -> VenueMetrics:
        """Get aggregated metrics across all venues."""
        import numpy as np
        
        # Fetch all data in parallel
        prices_task = self.get_all_prices(symbol)
        books_task = self.get_all_orderbooks(symbol)
        funding_task = self.get_all_funding(symbol)
        
        prices, books, funding = await asyncio.gather(
            prices_task, books_task, funding_task
        )
        
        metrics = VenueMetrics(symbol=symbol)
        
        # Price metrics
        if prices:
            price_values = [p.price for p in prices.values() if p.price > 0]
            metrics.prices = {v: p.price for v, p in prices.items()}
            metrics.avg_price = float(np.mean(price_values)) if price_values else 0
            metrics.price_std = float(np.std(price_values)) if len(price_values) > 1 else 0
            metrics.max_spread_pct = max(p.spread_pct for p in prices.values()) if prices else 0
        
        # Order book metrics
        if books:
            metrics.total_bid_depth = sum(b.bid_depth_usd for b in books.values())
            metrics.total_ask_depth = sum(b.ask_depth_usd for b in books.values())
            imbalances = [b.imbalance for b in books.values()]
            metrics.avg_imbalance = float(np.mean(imbalances)) if imbalances else 0
        
        # Funding metrics
        if funding:
            rates = [f.funding_rate for f in funding.values()]
            metrics.avg_funding_rate = float(np.mean(rates)) if rates else 0
            metrics.funding_divergence = float(np.std(rates)) if len(rates) > 1 else 0
            metrics.total_open_interest = sum(f.open_interest for f in funding.values())
        
        # Arb opportunities
        if prices:
            arbs = self.find_arb_opportunities(prices)
            metrics.arb_opportunities = arbs
            metrics.best_arb_pct = arbs[0].net_profit_pct if arbs else 0
        
        # Clean score components
        metrics.is_low_arb = metrics.best_arb_pct < 0.1
        metrics.is_stable_funding = abs(metrics.avg_funding_rate) < 0.0001  # ±0.01%
        metrics.is_low_vol_spike = True  # Would need historical comparison
        metrics.is_high_liquidity = (metrics.total_bid_depth + metrics.total_ask_depth) > 10_000_000
        
        return metrics
    
    def estimate_slippage(
        self,
        book: VenueOrderBook,
        size_usd: float,
        side: Literal["buy", "sell"]
    ) -> float:
        """Estimate slippage for a given order size."""
        levels = book.asks if side == "buy" else book.bids
        if not levels:
            return 0.01  # Default 1% if no book
        
        mid = book.mid_price
        if mid <= 0:
            return 0.01
        
        remaining = size_usd
        weighted_price = 0.0
        total_filled = 0.0
        
        for level in levels:
            level_usd = level.price * level.size
            fill = min(remaining, level_usd)
            weighted_price += level.price * fill
            total_filled += fill
            remaining -= fill
            
            if remaining <= 0:
                break
        
        if total_filled <= 0:
            return 0.01
        
        avg_price = weighted_price / total_filled
        slippage = abs(avg_price - mid) / mid
        
        return slippage


# Singleton
_multi_venue_client: MultiVenueClient | None = None


async def get_multi_venue_client() -> MultiVenueClient:
    """Get or create multi-venue client."""
    global _multi_venue_client
    if _multi_venue_client is None:
        _multi_venue_client = MultiVenueClient()
        await _multi_venue_client._init_exchanges()
    return _multi_venue_client
