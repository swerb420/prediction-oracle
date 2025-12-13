"""
Unified Market Intelligence - Aggregates data from multiple sources.

Combines:
1. 15M Polymarket prices (from outcome_collector)
2. Order book depth and imbalance (from polymarket_signals)
3. Whale trade detection (from whale_scanner patterns)
4. Multi-venue crypto prices (Binance, Bybit, Coinbase, Kraken)
5. Momentum indicators

This gives the ML and Grok rich context for better predictions.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional, Literal

import httpx

logger = logging.getLogger(__name__)

CryptoSymbol = Literal["BTC", "ETH", "SOL", "XRP"]
SYMBOLS = ["BTC", "ETH", "SOL", "XRP"]


@dataclass
class OrderBookData:
    """Order book analysis."""
    best_bid: float
    best_ask: float
    spread_pct: float
    bid_depth_usd: float  # $ within 5% of best bid
    ask_depth_usd: float  # $ within 5% of best ask
    imbalance: float  # -1 to 1, positive = more bids (bullish)


@dataclass
class MomentumData:
    """Price momentum indicators."""
    price_now: float
    price_1min_ago: float
    price_5min_ago: float
    change_1min_pct: float
    change_5min_pct: float
    velocity: float  # Rate of change
    acceleration: float  # Change in velocity


@dataclass
class WhaleSignal:
    """Whale/large trade activity."""
    large_buys_5min: int
    large_sells_5min: int
    net_whale_flow_usd: float  # Positive = net buying
    whale_bias: float  # -1 to 1


@dataclass
class VenuePrice:
    """Price from a specific venue."""
    venue: str
    price: float
    timestamp: datetime


@dataclass
class CryptoMarketData:
    """Comprehensive market data for a crypto symbol."""
    symbol: CryptoSymbol
    timestamp: datetime
    
    # Polymarket market data
    poly_yes_price: float  # Probability of UP
    poly_no_price: float
    poly_direction: str  # Market implied direction
    poly_volume: float
    poly_liquidity: float
    
    # Timeframe (15M, 1H, 4H)
    timeframe: str = "15M"
    
    # Order book
    orderbook: Optional[OrderBookData] = None
    
    # Multi-venue prices
    venue_prices: list[VenuePrice] = field(default_factory=list)
    avg_spot_price: float = 0.0
    venue_divergence: float = 0.0  # Max difference between venues (%)
    
    # Momentum
    momentum: Optional[MomentumData] = None
    
    # Whale activity
    whales: Optional[WhaleSignal] = None
    
    # Composite signals
    signal_strength: float = 0.0  # -1 to 1
    confidence: float = 0.0
    
    def to_dict(self) -> dict:
        """Convert to dict for ML/Grok input."""
        return {
            "symbol": self.symbol,
            "timestamp": self.timestamp.isoformat(),
            # Polymarket
            "yes_price": self.poly_yes_price,
            "no_price": self.poly_no_price,
            "market_direction": self.poly_direction,
            "volume": self.poly_volume,
            "liquidity": self.poly_liquidity,
            # Order book
            "spread_pct": self.orderbook.spread_pct if self.orderbook else 0,
            "orderbook_imbalance": self.orderbook.imbalance if self.orderbook else 0,
            "bid_depth": self.orderbook.bid_depth_usd if self.orderbook else 0,
            "ask_depth": self.orderbook.ask_depth_usd if self.orderbook else 0,
            # Venue
            "avg_spot_price": self.avg_spot_price,
            "venue_divergence": self.venue_divergence,
            "venue_count": len(self.venue_prices),
            # Momentum
            "change_1min_pct": self.momentum.change_1min_pct if self.momentum else 0,
            "change_5min_pct": self.momentum.change_5min_pct if self.momentum else 0,
            "velocity": self.momentum.velocity if self.momentum else 0,
            # Whales
            "whale_bias": self.whales.whale_bias if self.whales else 0,
            "net_whale_flow": self.whales.net_whale_flow_usd if self.whales else 0,
            # Composite
            "signal_strength": self.signal_strength,
            "confidence": self.confidence,
        }


class MarketIntelligence:
    """
    Aggregates market data from multiple sources for rich ML context.
    
    Sources:
    - Polymarket event pages (15M market prices)
    - Polymarket CLOB API (order book depth)
    - Multi-venue spot prices (Binance, Bybit, Kraken, Coinbase)
    - Internal price history (momentum calculation)
    """
    
    POLYMARKET_BASE = "https://polymarket.com"
    CLOB_API = "https://clob.polymarket.com"
    
    # CEX APIs
    VENUE_APIS = {
        "binance": "https://api.binance.com/api/v3/ticker/price?symbol={symbol}USDT",
        "bybit": "https://api.bybit.com/v5/market/tickers?category=spot&symbol={symbol}USDT",
        "kraken": "https://api.kraken.com/0/public/Ticker?pair={symbol}USD",
        "coinbase": "https://api.coinbase.com/v2/prices/{symbol}-USD/spot",
    }
    
    # Timeframe configurations
    TIMEFRAME_CONFIG = {
        "15M": {
            "url": "crypto/15M",
            "pattern": "{symbol}-updown-15m-\\d+",
            "window_secs": 900,
        },
        "1H": {
            "url": "crypto/hourly",
            "pattern": "{full_name}-up-or-down-[a-z]+-\\d+-\\d+[ap]m-et",
            "window_secs": 3600,
        },
        "4H": {
            "url": "crypto/4hour",
            "pattern": "{symbol}-updown-4h-\\d+",
            "window_secs": 14400,
        },
    }
    
    # Map short symbols to full names for hourly slugs
    SYMBOL_FULL_NAMES = {
        "BTC": "bitcoin",
        "ETH": "ethereum", 
        "SOL": "solana",
        "XRP": "xrp",
    }
    
    def __init__(self):
        self.client: Optional[httpx.AsyncClient] = None
        
        # Price history for momentum
        self.price_history: dict[str, list[tuple[datetime, float]]] = {
            s: [] for s in SYMBOLS
        }
        
        # Discovered market slugs by timeframe: {"15M": {"BTC": slug, ...}, "1H": {...}}
        self.market_slugs: dict[str, dict[str, str]] = {"15M": {}, "1H": {}, "4H": {}}
        self.token_ids: dict[str, dict[str, str]] = {"15M": {}, "1H": {}, "4H": {}}
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=15.0)
        return self
    
    async def __aexit__(self, *args):
        if self.client:
            await self.client.aclose()
    
    async def _ensure_client(self):
        if not self.client:
            self.client = httpx.AsyncClient(timeout=15.0)
    
    # ─────────────────────────────────────────────────────────────────────────
    # Market Discovery
    # ─────────────────────────────────────────────────────────────────────────
    
    async def discover_markets(self, timeframe: str = "15M") -> dict[str, str]:
        """Discover current market slugs for a timeframe (15M, 1H, 4H)."""
        await self._ensure_client()
        
        import re
        from datetime import datetime
        
        config = self.TIMEFRAME_CONFIG.get(timeframe)
        if not config:
            logger.error(f"Unknown timeframe: {timeframe}")
            return {}
        
        try:
            resp = await self.client.get(
                f"{self.POLYMARKET_BASE}/{config['url']}",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            resp.raise_for_status()
            
            # Build patterns for each symbol
            for symbol in SYMBOLS:
                full_name = self.SYMBOL_FULL_NAMES.get(symbol, symbol.lower())
                pattern = config["pattern"].format(
                    symbol=symbol.lower(),
                    symbol_lower=symbol.lower(),
                    full_name=full_name
                )
                
                # For 1H, find all matches and pick the most recent date
                if timeframe == "1H":
                    all_matches = re.findall(pattern, resp.text.lower())
                    if all_matches:
                        # Parse dates from slugs like "december-11-8pm-et" and pick latest
                        # For now, just pick the last unique match (usually latest)
                        unique = list(set(all_matches))
                        # Sort by trying to extract the date - pick the one closest to today
                        now = datetime.now()
                        
                        def extract_date_key(slug):
                            # Extract month-day from slug like bitcoin-up-or-down-december-11-7pm-et
                            month_map = {'january': 1, 'february': 2, 'march': 3, 'april': 4,
                                       'may': 5, 'june': 6, 'july': 7, 'august': 8,
                                       'september': 9, 'october': 10, 'november': 11, 'december': 12}
                            parts = slug.split('-')
                            for i, p in enumerate(parts):
                                if p in month_map and i + 1 < len(parts):
                                    try:
                                        month = month_map[p]
                                        day = int(parts[i + 1])
                                        # Create a date (use current year)
                                        test_date = datetime(now.year, month, day)
                                        days_diff = (test_date.date() - now.date()).days
                                        # Priority: today/future closest first, then most recent past
                                        if days_diff >= 0:
                                            return (0, days_diff)  # Future: smaller diff = better
                                        else:
                                            return (1, -days_diff)  # Past: smaller (abs) diff = more recent = better
                                    except:
                                        pass
                            return (2, 9999)  # Can't parse
                        
                        unique.sort(key=extract_date_key)
                        self.market_slugs[timeframe][symbol] = unique[0]
                else:
                    match = re.search(pattern, resp.text.lower())
                    if match:
                        self.market_slugs[timeframe][symbol] = match.group(0)
            
            logger.info(f"Discovered {len(self.market_slugs[timeframe])} {timeframe} markets")
            return self.market_slugs[timeframe]
            
        except Exception as e:
            logger.error(f"{timeframe} market discovery failed: {e}")
            return {}
    
    async def discover_all_markets(self) -> dict[str, dict[str, str]]:
        """Discover markets for all timeframes (15M, 1H, 4H)."""
        tasks = [
            self.discover_markets("15M"),
            self.discover_markets("1H"),
            self.discover_markets("4H"),
        ]
        await asyncio.gather(*tasks)
        return self.market_slugs
    
    # ─────────────────────────────────────────────────────────────────────────
    # Polymarket Data
    # ─────────────────────────────────────────────────────────────────────────
    
    async def fetch_polymarket_data(self, symbol: str, timeframe: str = "15M") -> Optional[dict]:
        """Fetch market data from Polymarket for a given timeframe."""
        await self._ensure_client()
        
        import json
        import re
        
        slug = self.market_slugs.get(timeframe, {}).get(symbol)
        if not slug:
            await self.discover_markets(timeframe)
            slug = self.market_slugs.get(timeframe, {}).get(symbol)
            if not slug:
                return None
        
        try:
            resp = await self.client.get(
                f"{self.POLYMARKET_BASE}/event/{slug}",
                headers={"User-Agent": "Mozilla/5.0"}
            )
            
            if resp.status_code != 200:
                return None
            
            # Extract __NEXT_DATA__
            match = re.search(
                r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>',
                resp.text,
                re.DOTALL,
            )
            if not match:
                return None
            
            data = json.loads(match.group(1))
            queries = (
                data.get("props", {})
                .get("pageProps", {})
                .get("dehydratedState", {})
                .get("queries", [])
            )
            
            # Build pattern for this timeframe
            tf_patterns = {
                "15M": "updown-15m",
                "1H": ["updown-1h", "up-or-down"],  
                "4H": "updown-4h",
            }
            pattern = tf_patterns.get(timeframe, "updown")
            
            for q in queries:
                query_key = q.get("queryKey", [])
                key_str = str(query_key[1]).lower() if len(query_key) >= 2 else ""
                
                # Check if this query matches our timeframe
                matches = False
                if isinstance(pattern, list):
                    matches = any(p in key_str for p in pattern)
                else:
                    matches = pattern in key_str
                
                if matches:
                    event_data = q.get("state", {}).get("data", {})
                    markets = event_data.get("markets", [])
                    
                    if markets:
                        m = markets[0]
                        prices = m.get("outcomePrices", ["0.5", "0.5"])
                        
                        # Store token ID for orderbook queries (by timeframe)
                        clob_ids = m.get("clobTokenIds", [])
                        if clob_ids:
                            if timeframe not in self.token_ids:
                                self.token_ids[timeframe] = {}
                            self.token_ids[timeframe][symbol] = clob_ids[0]
                        
                        return {
                            "yes_price": float(prices[0]) if prices else 0.5,
                            "no_price": float(prices[1]) if len(prices) > 1 else 0.5,
                            "volume": float(m.get("volume", 0)),
                            "liquidity": float(m.get("liquidity", 0)),
                            "question": m.get("question", ""),
                            "timeframe": timeframe,
                        }
            
            return None
            
        except Exception as e:
            logger.error(f"Polymarket fetch error for {symbol}: {e}")
            return None
    
    async def fetch_orderbook(self, symbol: str, timeframe: str = "15M") -> Optional[OrderBookData]:
        """Fetch order book data from CLOB API."""
        await self._ensure_client()
        
        token_id = self.token_ids.get(timeframe, {}).get(symbol)
        if not token_id:
            return None
        
        try:
            resp = await self.client.get(
                f"{self.CLOB_API}/book",
                params={"token_id": token_id}
            )
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            bids = data.get("bids", [])
            asks = data.get("asks", [])
            
            if not bids or not asks:
                return None
            
            best_bid = float(bids[0]["price"])
            best_ask = float(asks[0]["price"])
            spread = best_ask - best_bid
            
            # Calculate depth within 5%
            bid_depth = sum(
                float(b["size"]) * float(b["price"])
                for b in bids[:20]
                if float(b["price"]) >= best_bid * 0.95
            )
            
            ask_depth = sum(
                float(a["size"]) * float(a["price"])
                for a in asks[:20]
                if float(a["price"]) <= best_ask * 1.05
            )
            
            total_depth = bid_depth + ask_depth
            imbalance = (bid_depth - ask_depth) / total_depth if total_depth > 0 else 0
            
            return OrderBookData(
                best_bid=best_bid,
                best_ask=best_ask,
                spread_pct=spread / best_bid * 100 if best_bid > 0 else 0,
                bid_depth_usd=bid_depth,
                ask_depth_usd=ask_depth,
                imbalance=imbalance,
            )
            
        except Exception as e:
            logger.debug(f"Orderbook fetch error for {symbol}: {e}")
            return None
    
    # ─────────────────────────────────────────────────────────────────────────
    # Multi-Venue Spot Prices
    # ─────────────────────────────────────────────────────────────────────────
    
    async def fetch_venue_price(self, symbol: str, venue: str) -> Optional[VenuePrice]:
        """Fetch price from a specific venue."""
        await self._ensure_client()
        
        # Symbol mapping
        sym_map = {
            "XRP": {"kraken": "XXRPZUSD"}
        }
        
        try:
            url_template = self.VENUE_APIS.get(venue)
            if not url_template:
                return None
            
            url = url_template.format(symbol=symbol)
            resp = await self.client.get(url, timeout=5.0)
            
            if resp.status_code != 200:
                return None
            
            data = resp.json()
            price = None
            
            if venue == "binance":
                price = float(data.get("price", 0))
            elif venue == "bybit":
                result = data.get("result", {}).get("list", [])
                if result:
                    price = float(result[0].get("lastPrice", 0))
            elif venue == "kraken":
                # Kraken uses different symbol format
                pair_key = sym_map.get(symbol, {}).get("kraken", f"X{symbol}ZUSD")
                result = data.get("result", {})
                if pair_key in result:
                    price = float(result[pair_key]["c"][0])
                else:
                    # Try alternative formats
                    for key in result:
                        if symbol.upper() in key.upper():
                            price = float(result[key]["c"][0])
                            break
            elif venue == "coinbase":
                price = float(data.get("data", {}).get("amount", 0))
            
            if price and price > 0:
                return VenuePrice(
                    venue=venue,
                    price=price,
                    timestamp=datetime.now(timezone.utc),
                )
            
            return None
            
        except Exception as e:
            logger.debug(f"Venue {venue} fetch error for {symbol}: {e}")
            return None
    
    async def fetch_all_venue_prices(self, symbol: str) -> list[VenuePrice]:
        """Fetch prices from all venues in parallel."""
        venues = ["binance", "bybit", "coinbase"]  # Kraken is slow
        
        tasks = [self.fetch_venue_price(symbol, v) for v in venues]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        prices = []
        for r in results:
            if isinstance(r, VenuePrice):
                prices.append(r)
        
        return prices
    
    # ─────────────────────────────────────────────────────────────────────────
    # Momentum
    # ─────────────────────────────────────────────────────────────────────────
    
    def update_price_history(self, symbol: str, price: float):
        """Update price history for momentum calculation."""
        now = datetime.now(timezone.utc)
        history = self.price_history[symbol]
        
        # Add new price
        history.append((now, price))
        
        # Keep only last 10 minutes
        cutoff = now - timedelta(minutes=10)
        self.price_history[symbol] = [(t, p) for t, p in history if t > cutoff]
    
    def calculate_momentum(self, symbol: str, current_price: float) -> Optional[MomentumData]:
        """Calculate momentum indicators from price history."""
        history = self.price_history[symbol]
        
        if len(history) < 2:
            return None
        
        now = datetime.now(timezone.utc)
        
        # Find prices at different time points
        price_1min = current_price
        price_5min = current_price
        
        for t, p in reversed(history):
            age = (now - t).total_seconds()
            if age >= 55 and price_1min == current_price:  # ~1 min ago
                price_1min = p
            if age >= 280:  # ~5 min ago
                price_5min = p
                break
        
        change_1min = (current_price - price_1min) / price_1min * 100 if price_1min > 0 else 0
        change_5min = (current_price - price_5min) / price_5min * 100 if price_5min > 0 else 0
        
        # Velocity and acceleration
        velocity = change_1min  # % per minute
        
        # Calculate acceleration from recent changes
        if len(history) >= 3:
            recent_changes = []
            for i in range(min(5, len(history) - 1)):
                t1, p1 = history[-(i+1)]
                t2, p2 = history[-(i+2)]
                dt = (t1 - t2).total_seconds()
                if dt > 0:
                    recent_changes.append((p1 - p2) / p2 / dt * 60)  # % per minute
            
            if len(recent_changes) >= 2:
                acceleration = recent_changes[0] - recent_changes[-1]
            else:
                acceleration = 0
        else:
            acceleration = 0
        
        return MomentumData(
            price_now=current_price,
            price_1min_ago=price_1min,
            price_5min_ago=price_5min,
            change_1min_pct=change_1min,
            change_5min_pct=change_5min,
            velocity=velocity,
            acceleration=acceleration,
        )
    
    # ─────────────────────────────────────────────────────────────────────────
    # Aggregated Intelligence
    # ─────────────────────────────────────────────────────────────────────────
    
    async def get_market_intelligence(self, symbol: str, timeframe: str = "15M") -> Optional[CryptoMarketData]:
        """
        Get comprehensive market intelligence for a symbol and timeframe.
        
        Fetches all data sources in parallel for speed.
        """
        await self._ensure_client()
        
        # Fetch polymarket data first (populates token_ids needed for orderbook)
        poly_data = await self.fetch_polymarket_data(symbol, timeframe)
        
        if poly_data is None:
            logger.warning(f"No Polymarket data for {symbol}/{timeframe}")
            return None
        
        # Now fetch orderbook and venue prices in parallel
        orderbook_task = self.fetch_orderbook(symbol, timeframe)
        venues_task = self.fetch_all_venue_prices(symbol)
        
        orderbook, venue_prices = await asyncio.gather(
            orderbook_task, venues_task,
            return_exceptions=True
        )
        
        # Handle errors
        if isinstance(orderbook, Exception):
            orderbook = None
        
        if isinstance(venue_prices, Exception):
            venue_prices = []
        
        # Calculate average spot price
        if venue_prices:
            avg_price = sum(vp.price for vp in venue_prices) / len(venue_prices)
            max_price = max(vp.price for vp in venue_prices)
            min_price = min(vp.price for vp in venue_prices)
            divergence = (max_price - min_price) / avg_price * 100 if avg_price > 0 else 0
            
            # Update price history
            self.update_price_history(symbol, avg_price)
        else:
            avg_price = 0
            divergence = 0
        
        # Calculate momentum
        momentum = self.calculate_momentum(symbol, avg_price) if avg_price > 0 else None
        
        # Calculate composite signal strength
        signal = 0.0
        confidence = 0.0
        
        # Orderbook imbalance contribution (40%)
        if orderbook:
            signal += orderbook.imbalance * 0.4
            confidence += 0.3
        
        # Momentum contribution (30%)
        if momentum:
            # Normalize momentum to -1 to 1
            mom_signal = max(-1, min(1, momentum.change_5min_pct / 2))
            signal += mom_signal * 0.3
            confidence += 0.3
        
        # Polymarket price contribution (30%)
        # Market already expressing directional view
        poly_signal = (poly_data["yes_price"] - 0.5) * 2  # -1 to 1
        signal += poly_signal * 0.3
        confidence += 0.4
        
        # Clamp signal
        signal = max(-1, min(1, signal))
        confidence = min(1, confidence)
        
        return CryptoMarketData(
            symbol=symbol,
            timestamp=datetime.now(timezone.utc),
            poly_yes_price=poly_data["yes_price"],
            poly_no_price=poly_data["no_price"],
            poly_direction="UP" if poly_data["yes_price"] > 0.5 else "DOWN",
            poly_volume=poly_data["volume"],
            poly_liquidity=poly_data["liquidity"],
            timeframe=timeframe,
            orderbook=orderbook,
            venue_prices=venue_prices,
            avg_spot_price=avg_price,
            venue_divergence=divergence,
            momentum=momentum,
            whales=None,  # Would need whale scanner running
            signal_strength=signal,
            confidence=confidence,
        )
    
    async def get_all_markets(self, timeframe: str = "15M") -> dict[str, CryptoMarketData]:
        """Get intelligence for all symbols for a given timeframe."""
        # First discover markets for this timeframe
        await self.discover_markets(timeframe)
        
        # Fetch all in parallel
        tasks = {s: self.get_market_intelligence(s, timeframe) for s in SYMBOLS}
        results = await asyncio.gather(*tasks.values(), return_exceptions=True)
        
        markets = {}
        for symbol, result in zip(SYMBOLS, results):
            if isinstance(result, CryptoMarketData):
                markets[symbol] = result
        
        return markets
    
    async def get_all_timeframe_markets(self) -> dict[str, dict[str, CryptoMarketData]]:
        """Get markets for all timeframes (15M, 1H, 4H).
        
        Returns: {"15M": {"BTC": data, ...}, "1H": {...}, "4H": {...}}
        """
        # Discover all markets first
        await self.discover_all_markets()
        
        # Fetch all timeframes in parallel
        results = {}
        for tf in ["15M", "1H", "4H"]:
            results[tf] = await self.get_all_markets(tf)
        
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Test
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    logging.basicConfig(level=logging.INFO)
    
    async with MarketIntelligence() as intel:
        print("Fetching market intelligence...")
        markets = await intel.get_all_markets()
        
        for symbol, data in markets.items():
            print(f"\n{'='*60}")
            print(f"{symbol}")
            print(f"{'='*60}")
            print(f"  Polymarket: YES={data.poly_yes_price:.1%} → {data.poly_direction}")
            print(f"  Volume: ${data.poly_volume:,.0f}  Liquidity: ${data.poly_liquidity:,.0f}")
            
            if data.orderbook:
                print(f"  Orderbook: Spread={data.orderbook.spread_pct:.2f}% "
                      f"Imbalance={data.orderbook.imbalance:+.2f}")
            
            if data.venue_prices:
                print(f"  Spot Price: ${data.avg_spot_price:,.2f} "
                      f"({len(data.venue_prices)} venues, {data.venue_divergence:.3f}% div)")
            
            if data.momentum:
                print(f"  Momentum: 1m={data.momentum.change_1min_pct:+.2f}% "
                      f"5m={data.momentum.change_5min_pct:+.2f}%")
            
            print(f"  Signal: {data.signal_strength:+.2f} (conf={data.confidence:.0%})")


if __name__ == "__main__":
    asyncio.run(main())
