"""
Polymarket Whale Tracker
========================

Tracks top 25-50 Polymarket traders for crypto markets.
Fetches their recent trades via Data-API to compute:
- Whale consensus (bullish/bearish agreement)
- Trade velocity (recent activity intensity)
- Whale alpha signals (top traders moving together)

Methods:
1. Scrape leaderboard for top addresses
2. Fetch wallet trades via Polymarket Data-API
3. Filter for BTC/ETH/SOL/XRP crypto markets
4. Compute consensus features
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Literal

import httpx
from pydantic import BaseModel

logger = logging.getLogger(__name__)

CryptoSymbol = Literal["BTC", "ETH", "SOL", "XRP"]


@dataclass
class WhaleProfile:
    """Profile of a top Polymarket trader."""
    address: str
    rank: int
    total_pnl: float
    win_rate: float
    volume_traded: float
    last_seen: datetime | None = None


@dataclass
class WhaleTrade:
    """A trade by a whale on Polymarket."""
    address: str
    market_id: str
    market_title: str
    outcome: str  # "Yes" or "No"
    side: str  # "BUY" or "SELL"
    price: float
    size: float
    timestamp: datetime
    crypto_symbol: CryptoSymbol | None = None  # If crypto-related


class WhaleConsensus(BaseModel):
    """Aggregated whale consensus for a crypto symbol."""
    symbol: CryptoSymbol
    timestamp: datetime
    
    # Consensus metrics
    bullish_count: int  # Whales betting UP
    bearish_count: int  # Whales betting DOWN
    neutral_count: int  # No recent position
    
    # Consensus score: -1 (all bearish) to +1 (all bullish)
    consensus_score: float
    
    # Confidence based on participation
    participation_rate: float  # % of tracked whales with positions
    
    # Volume-weighted consensus
    bullish_volume: float
    bearish_volume: float
    volume_weighted_score: float
    
    # Trade velocity (trades in last hour)
    recent_trade_count: int
    trade_velocity: float  # Trades per hour
    
    # Top whale alignment (top 10)
    top_10_consensus: float


# Known crypto market patterns for Polymarket
CRYPTO_PATTERNS = {
    "BTC": [
        r"bitcoin",
        r"\bbtc\b",
        r"btc.*price",
        r"bitcoin.*\d+k",
    ],
    "ETH": [
        r"ethereum",
        r"\beth\b",
        r"eth.*price",
        r"ethereum.*\d+",
    ],
    "SOL": [
        r"solana",
        r"\bsol\b",
        r"sol.*price",
    ],
    "XRP": [
        r"\bxrp\b",
        r"ripple",
        r"xrp.*price",
    ],
}


def detect_crypto_symbol(market_title: str) -> CryptoSymbol | None:
    """Detect which crypto a market is about from its title."""
    title_lower = market_title.lower()
    
    for symbol, patterns in CRYPTO_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, title_lower):
                return symbol  # type: ignore
    
    return None


class PolyWhalesClient:
    """
    Client for tracking Polymarket whales.
    
    Uses:
    - Polymarket Data-API for trade history
    - Scraping for leaderboard (when needed)
    """
    
    # Polymarket Data-API base
    DATA_API_BASE = "https://data-api.polymarket.com"
    
    def __init__(
        self,
        top_n: int = 25,
        timeout: float = 30.0,
    ):
        """
        Initialize whale tracker.
        
        Args:
            top_n: Number of top traders to track
            timeout: HTTP timeout in seconds
        """
        self.top_n = top_n
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        
        # Cache
        self._whale_profiles: list[WhaleProfile] = []
        self._last_profile_update: datetime | None = None
        self._profile_cache_ttl = timedelta(hours=1)
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._client = httpx.AsyncClient(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._client:
            await self._client.aclose()
            self._client = None
    
    async def get_top_traders(self, force_refresh: bool = False) -> list[WhaleProfile]:
        """
        Get top trader addresses.
        
        Uses cached data if available and fresh.
        Currently returns mock data - in production, scrape polymarket.com/leaderboard
        """
        now = datetime.now(timezone.utc)
        
        # Check cache
        if (
            not force_refresh
            and self._whale_profiles
            and self._last_profile_update
            and (now - self._last_profile_update) < self._profile_cache_ttl
        ):
            return self._whale_profiles[:self.top_n]
        
        # In production, scrape the leaderboard
        # For now, return known active addresses (placeholder)
        logger.info("Fetching top trader profiles...")
        
        # Mock top traders - in production, scrape from polymarket.com/leaderboard
        # These would be real addresses from the leaderboard
        mock_profiles = [
            WhaleProfile(
                address=f"0x{i:040x}",
                rank=i + 1,
                total_pnl=1000000 - i * 10000,
                win_rate=0.65 - i * 0.005,
                volume_traded=5000000 - i * 100000,
            )
            for i in range(self.top_n)
        ]
        
        self._whale_profiles = mock_profiles
        self._last_profile_update = now
        
        return mock_profiles[:self.top_n]
    
    async def get_wallet_trades(
        self,
        address: str,
        limit: int = 50,
        since: datetime | None = None,
    ) -> list[WhaleTrade]:
        """
        Get recent trades for a wallet address.
        
        Args:
            address: Ethereum address
            limit: Max trades to fetch
            since: Only fetch trades after this time
            
        Returns:
            List of WhaleTrade objects
        """
        if not self._client:
            raise RuntimeError("Client not initialized. Use async context manager.")
        
        try:
            # Polymarket Data-API endpoint for trades
            resp = await self._client.get(
                f"{self.DATA_API_BASE}/trades",
                params={
                    "maker": address,
                    "limit": limit,
                }
            )
            
            if resp.status_code == 404:
                return []
            
            resp.raise_for_status()
            data = resp.json()
            
            trades = []
            for t in data if isinstance(data, list) else data.get("trades", []):
                try:
                    ts = datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00"))
                    
                    if since and ts < since:
                        continue
                    
                    market_title = t.get("market_title", t.get("market", ""))
                    crypto_symbol = detect_crypto_symbol(market_title)
                    
                    trade = WhaleTrade(
                        address=address,
                        market_id=t.get("market_id", t.get("condition_id", "")),
                        market_title=market_title,
                        outcome=t.get("outcome", ""),
                        side=t.get("side", "BUY"),
                        price=float(t.get("price", 0.5)),
                        size=float(t.get("size", 0)),
                        timestamp=ts,
                        crypto_symbol=crypto_symbol,
                    )
                    trades.append(trade)
                except Exception as e:
                    logger.debug(f"Failed to parse trade: {e}")
                    continue
            
            return trades
            
        except httpx.HTTPError as e:
            logger.warning(f"Failed to fetch trades for {address[:10]}...: {e}")
            return []
    
    async def get_crypto_trades(
        self,
        symbol: CryptoSymbol,
        hours_back: int = 24,
    ) -> list[WhaleTrade]:
        """
        Get all whale trades for a specific crypto symbol.
        
        Args:
            symbol: Crypto symbol to filter for
            hours_back: How many hours back to look
            
        Returns:
            List of WhaleTrade objects for this crypto
        """
        whales = await self.get_top_traders()
        since = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        all_trades = []
        
        # Fetch trades in batches to respect rate limits
        batch_size = 5
        for i in range(0, len(whales), batch_size):
            batch = whales[i:i + batch_size]
            
            tasks = [
                self.get_wallet_trades(w.address, limit=50, since=since)
                for w in batch
            ]
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, list):
                    for trade in result:
                        if trade.crypto_symbol == symbol:
                            all_trades.append(trade)
            
            # Small delay between batches
            if i + batch_size < len(whales):
                await asyncio.sleep(0.5)
        
        return all_trades
    
    async def get_whale_consensus(
        self,
        symbol: CryptoSymbol,
        hours_back: int = 6,
    ) -> WhaleConsensus:
        """
        Compute whale consensus for a crypto symbol.
        
        Args:
            symbol: Crypto symbol
            hours_back: Time window for recent trades
            
        Returns:
            WhaleConsensus with aggregated metrics
        """
        whales = await self.get_top_traders()
        trades = await self.get_crypto_trades(symbol, hours_back)
        
        now = datetime.now(timezone.utc)
        
        # Aggregate by whale
        whale_positions: dict[str, dict] = {}
        
        for trade in trades:
            addr = trade.address
            if addr not in whale_positions:
                whale_positions[addr] = {
                    "bullish_volume": 0.0,
                    "bearish_volume": 0.0,
                    "trade_count": 0,
                    "last_trade": trade.timestamp,
                }
            
            pos = whale_positions[addr]
            pos["trade_count"] += 1
            
            if trade.timestamp > pos["last_trade"]:
                pos["last_trade"] = trade.timestamp
            
            # Determine if bullish or bearish
            # "Yes" on price up = bullish, "No" on price up = bearish
            # This is simplified - real logic depends on market structure
            is_bullish = (
                (trade.outcome.lower() == "yes" and trade.side == "BUY") or
                (trade.outcome.lower() == "no" and trade.side == "SELL")
            )
            
            if is_bullish:
                pos["bullish_volume"] += trade.size * trade.price
            else:
                pos["bearish_volume"] += trade.size * trade.price
        
        # Compute consensus metrics
        bullish_count = 0
        bearish_count = 0
        neutral_count = 0
        
        total_bullish_vol = 0.0
        total_bearish_vol = 0.0
        
        for addr, pos in whale_positions.items():
            if pos["bullish_volume"] > pos["bearish_volume"] * 1.2:
                bullish_count += 1
            elif pos["bearish_volume"] > pos["bullish_volume"] * 1.2:
                bearish_count += 1
            else:
                neutral_count += 1
            
            total_bullish_vol += pos["bullish_volume"]
            total_bearish_vol += pos["bearish_volume"]
        
        # Whales with no positions
        inactive_count = len(whales) - len(whale_positions)
        neutral_count += inactive_count
        
        # Consensus scores
        active = bullish_count + bearish_count
        if active > 0:
            consensus_score = (bullish_count - bearish_count) / active
        else:
            consensus_score = 0.0
        
        participation_rate = len(whale_positions) / len(whales) if whales else 0.0
        
        total_vol = total_bullish_vol + total_bearish_vol
        if total_vol > 0:
            volume_weighted_score = (total_bullish_vol - total_bearish_vol) / total_vol
        else:
            volume_weighted_score = 0.0
        
        # Trade velocity
        recent_trade_count = len(trades)
        trade_velocity = recent_trade_count / hours_back
        
        # Top 10 consensus
        top_10_whales = set(w.address for w in whales[:10])
        top_10_bull = sum(1 for addr, pos in whale_positions.items() 
                         if addr in top_10_whales and pos["bullish_volume"] > pos["bearish_volume"])
        top_10_bear = sum(1 for addr, pos in whale_positions.items() 
                         if addr in top_10_whales and pos["bearish_volume"] > pos["bullish_volume"])
        top_10_active = top_10_bull + top_10_bear
        if top_10_active > 0:
            top_10_consensus = (top_10_bull - top_10_bear) / top_10_active
        else:
            top_10_consensus = 0.0
        
        return WhaleConsensus(
            symbol=symbol,
            timestamp=now,
            bullish_count=bullish_count,
            bearish_count=bearish_count,
            neutral_count=neutral_count,
            consensus_score=consensus_score,
            participation_rate=participation_rate,
            bullish_volume=total_bullish_vol,
            bearish_volume=total_bearish_vol,
            volume_weighted_score=volume_weighted_score,
            recent_trade_count=recent_trade_count,
            trade_velocity=trade_velocity,
            top_10_consensus=top_10_consensus,
        )


async def get_whale_signals(
    symbol: CryptoSymbol,
    hours_back: int = 6,
) -> WhaleConsensus:
    """
    Convenience function to get whale consensus for a symbol.
    
    Args:
        symbol: Crypto symbol (BTC, ETH, SOL, XRP)
        hours_back: Time window for analysis
        
    Returns:
        WhaleConsensus with aggregated metrics
    """
    async with PolyWhalesClient(top_n=25) as client:
        return await client.get_whale_consensus(symbol, hours_back)
