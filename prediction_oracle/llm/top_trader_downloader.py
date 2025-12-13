"""
Top Trader Downloader - Scrape Polymarket leaderboard for ML training data.

Downloads trade history from top crypto traders on Polymarket:
- https://polymarket.com/leaderboard/crypto/monthly/profit
- https://polymarket.com/leaderboard/crypto/all/profit

Uses this data to train ML models on whale behavior patterns.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Data Models
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class TopTrader:
    """A top trader from Polymarket leaderboard."""

    address: str
    rank: int
    username: Optional[str] = None
    profit: float = 0.0
    volume: float = 0.0
    markets_traded: int = 0
    win_rate: Optional[float] = None
    leaderboard_type: str = "crypto"  # "crypto" or "all"
    time_period: str = "monthly"  # "monthly", "weekly", "all"
    scraped_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["scraped_at"] = self.scraped_at.isoformat()
        return d


@dataclass
class TraderTrade:
    """A single trade from a top trader."""

    trader_address: str
    market_id: str
    market_question: str
    outcome: str  # "Yes" or "No"
    side: str  # "buy" or "sell"
    price: float  # 0-1
    size: float  # USD amount
    timestamp: datetime
    asset_ticker: Optional[str] = None  # BTC, ETH, SOL, XRP

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d


@dataclass
class TraderPosition:
    """Current position of a trader in a market."""

    trader_address: str
    market_id: str
    market_question: str
    outcome: str
    shares: float
    avg_price: float
    current_value: float
    pnl: float
    asset_ticker: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# Top Trader Downloader
# ─────────────────────────────────────────────────────────────────────────────


class TopTraderDownloader:
    """
    Downloads top trader data from Polymarket for ML training.

    Polymarket APIs:
    - Leaderboard: https://data-api.polymarket.com/leaderboard
    - Profile trades: https://data-api.polymarket.com/trades?maker=<address>
    - Profile positions: https://data-api.polymarket.com/positions?user=<address>
    """

    POLYMARKET_DATA_API = "https://data-api.polymarket.com"
    POLYMARKET_CLOB_API = "https://clob.polymarket.com"
    GAMMA_API = "https://gamma-api.polymarket.com"

    # 15M crypto market keywords
    CRYPTO_KEYWORDS = ["15M", "BTC", "ETH", "SOL", "XRP", "Bitcoin", "Ethereum", "Solana"]

    def __init__(
        self,
        data_dir: Path | str = "./trader_data",
        max_traders: int = 50,
        max_trades_per_trader: int = 500,
    ):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.max_traders = max_traders
        self.max_trades_per_trader = max_trades_per_trader

        self._client: Optional[httpx.AsyncClient] = None
        self._top_traders: list[TopTrader] = []
        self._trades: list[TraderTrade] = []
        self._positions: list[TraderPosition] = []

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                timeout=30.0,
                headers={
                    "User-Agent": "Mozilla/5.0 (compatible; PredictionOracle/1.0)",
                    "Accept": "application/json",
                },
            )
        return self._client

    async def close(self):
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    # ─────────────────────────────────────────────────────────────────────────
    # Leaderboard Scraping
    # ─────────────────────────────────────────────────────────────────────────

    async def fetch_leaderboard(
        self,
        leaderboard_type: str = "crypto",
        time_period: str = "monthly",
        limit: int = 50,
    ) -> list[TopTrader]:
        """
        Fetch top traders by analyzing large crypto trades.

        Since Polymarket doesn't expose a public leaderboard API,
        we identify whales by their large trade activity in crypto markets.
        """
        client = await self._get_client()

        try:
            # Fetch recent trades and identify large crypto traders
            url = f"{self.POLYMARKET_DATA_API}/trades"
            all_traders: dict[str, dict] = {}

            # Fetch multiple batches to get more data
            for offset in range(0, 500, 100):
                params = {"limit": 100, "offset": offset}
                resp = await client.get(url, params=params)

                if resp.status_code != 200:
                    break

                trades = resp.json()
                if not trades:
                    break

                for trade in trades:
                    title = trade.get("title", "")
                    # Filter for crypto trades
                    if not any(kw.lower() in title.lower() for kw in self.CRYPTO_KEYWORDS):
                        continue

                    wallet = trade.get("proxyWallet", "")
                    if not wallet:
                        continue

                    size = float(trade.get("size", 0))

                    if wallet not in all_traders:
                        all_traders[wallet] = {
                            "address": wallet,
                            "total_volume": 0,
                            "trade_count": 0,
                            "max_trade": 0,
                        }

                    all_traders[wallet]["total_volume"] += size
                    all_traders[wallet]["trade_count"] += 1
                    all_traders[wallet]["max_trade"] = max(
                        all_traders[wallet]["max_trade"], size
                    )

                await asyncio.sleep(0.2)  # Rate limit

            # Rank by total volume
            ranked = sorted(
                all_traders.values(),
                key=lambda x: x["total_volume"],
                reverse=True,
            )[:limit]

            traders = []
            for i, entry in enumerate(ranked, 1):
                trader = TopTrader(
                    address=entry["address"],
                    rank=i,
                    volume=entry["total_volume"],
                    markets_traded=entry["trade_count"],
                    leaderboard_type=leaderboard_type,
                    time_period=time_period,
                )
                traders.append(trader)

            logger.info(f"Identified {len(traders)} top crypto traders by volume")
            return traders

        except Exception as e:
            logger.error(f"Failed to fetch leaderboard: {e}")
            return []

    async def fetch_all_leaderboards(self) -> list[TopTrader]:
        """Fetch traders from all relevant leaderboards."""
        all_traders: dict[str, TopTrader] = {}

        # Fetch different leaderboard combinations
        configs = [
            ("crypto", "monthly"),
            ("crypto", "all"),
            ("all", "monthly"),  # Also get general top traders who might trade crypto
        ]

        for lb_type, period in configs:
            traders = await self.fetch_leaderboard(lb_type, period, limit=30)
            for t in traders:
                # Dedupe by address, prefer higher rank in crypto leaderboard
                if t.address not in all_traders:
                    all_traders[t.address] = t
                elif t.leaderboard_type == "crypto" and all_traders[t.address].leaderboard_type != "crypto":
                    all_traders[t.address] = t

            await asyncio.sleep(0.5)  # Rate limiting

        self._top_traders = list(all_traders.values())
        logger.info(f"Total unique traders: {len(self._top_traders)}")
        return self._top_traders

    # ─────────────────────────────────────────────────────────────────────────
    # Trade History Fetching
    # ─────────────────────────────────────────────────────────────────────────

    async def fetch_trader_trades(
        self,
        address: str,
        crypto_only: bool = True,
        limit: int = 500,
    ) -> list[TraderTrade]:
        """
        Fetch trade history for a specific trader.

        Args:
            address: Wallet address of the trader
            crypto_only: Filter to only crypto-related markets
            limit: Max trades to fetch
        """
        client = await self._get_client()

        try:
            # Polymarket trades API - filter by user
            url = f"{self.POLYMARKET_DATA_API}/trades"
            params = {
                "user": address,
                "limit": min(limit, self.max_trades_per_trader),
            }

            resp = await client.get(url, params=params)
            
            # If user param doesn't work, fall back to getting all trades
            if resp.status_code == 400:
                # Try without user filter and filter locally
                params = {"limit": 200}
                resp = await client.get(url, params=params)
            
            if resp.status_code != 200:
                logger.warning(f"Trades API returned {resp.status_code}")
                return []
                
            data = resp.json()

            trades = []
            for entry in data:
                # Skip if not from this trader (when fetching all)
                wallet = entry.get("proxyWallet", "")
                if "user" not in params and wallet != address:
                    continue
                    
                title = entry.get("title", "")
                market_id = entry.get("conditionId", "")

                # Filter for crypto markets if requested
                if crypto_only:
                    is_crypto = any(kw.lower() in title.lower() for kw in self.CRYPTO_KEYWORDS)
                    if not is_crypto:
                        continue

                # Determine asset ticker from title
                asset = self._extract_asset_from_question(title)

                try:
                    ts_str = entry.get("timestamp", "")
                    if ts_str:
                        timestamp = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
                    else:
                        timestamp = datetime.now(timezone.utc)
                except:
                    timestamp = datetime.now(timezone.utc)

                trade = TraderTrade(
                    trader_address=wallet or address,
                    market_id=market_id,
                    market_question=title,
                    outcome=entry.get("asset", "Yes"),  # asset field contains "Yes"/"No"
                    side=entry.get("side", "buy").lower(),
                    price=float(entry.get("price", 0)),
                    size=float(entry.get("size", 0)),
                    timestamp=timestamp,
                    asset_ticker=asset,
                )
                trades.append(trade)

            logger.info(f"Fetched {len(trades)} trades for {address[:10]}...")
            return trades

        except Exception as e:
            logger.error(f"Failed to fetch trades for {address[:10]}: {e}")
            return []

    async def fetch_trader_positions(
        self,
        address: str,
        crypto_only: bool = True,
    ) -> list[TraderPosition]:
        """Fetch current positions for a trader."""
        client = await self._get_client()

        try:
            url = f"{self.POLYMARKET_DATA_API}/positions"
            params = {"user": address}

            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()

            positions = []
            for entry in data:
                question = entry.get("market", {}).get("question", "")
                market_id = entry.get("market", {}).get("conditionId", "")

                if crypto_only:
                    is_crypto = any(kw.lower() in question.lower() for kw in self.CRYPTO_KEYWORDS)
                    if not is_crypto:
                        continue

                asset = self._extract_asset_from_question(question)

                position = TraderPosition(
                    trader_address=address,
                    market_id=market_id,
                    market_question=question,
                    outcome=entry.get("outcome", "Yes"),
                    shares=float(entry.get("size", entry.get("shares", 0))),
                    avg_price=float(entry.get("avgPrice", entry.get("averagePrice", 0))),
                    current_value=float(entry.get("currentValue", entry.get("value", 0))),
                    pnl=float(entry.get("pnl", entry.get("profit", 0))),
                    asset_ticker=asset,
                )
                positions.append(position)

            logger.info(f"Fetched {len(positions)} positions for {address[:10]}...")
            return positions

        except Exception as e:
            logger.error(f"Failed to fetch positions for {address[:10]}: {e}")
            return []

    def _extract_asset_from_question(self, question: str) -> Optional[str]:
        """Extract crypto asset ticker from market question."""
        q_lower = question.lower()
        if "btc" in q_lower or "bitcoin" in q_lower:
            return "BTC"
        elif "eth" in q_lower or "ethereum" in q_lower:
            return "ETH"
        elif "sol" in q_lower or "solana" in q_lower:
            return "SOL"
        elif "xrp" in q_lower or "ripple" in q_lower:
            return "XRP"
        return None

    # ─────────────────────────────────────────────────────────────────────────
    # Bulk Download
    # ─────────────────────────────────────────────────────────────────────────

    async def download_all(
        self,
        save_to_disk: bool = True,
    ) -> dict:
        """
        Download all top traders and their trade history.

        Returns:
            dict with "traders", "trades", "positions", "summary"
        """
        logger.info("Starting bulk download of top trader data...")

        # Step 1: Fetch all leaderboards
        traders = await self.fetch_all_leaderboards()
        logger.info(f"Found {len(traders)} unique top traders")

        # Step 2: Fetch trades and positions for each trader
        all_trades: list[TraderTrade] = []
        all_positions: list[TraderPosition] = []

        for i, trader in enumerate(traders[:self.max_traders]):
            logger.info(f"Processing trader {i+1}/{min(len(traders), self.max_traders)}: {trader.address[:10]}...")

            # Fetch trades
            trades = await self.fetch_trader_trades(trader.address, crypto_only=True)
            all_trades.extend(trades)

            # Fetch positions
            positions = await self.fetch_trader_positions(trader.address, crypto_only=True)
            all_positions.extend(positions)

            # Rate limiting
            await asyncio.sleep(0.3)

        self._trades = all_trades
        self._positions = all_positions

        # Summary stats
        summary = {
            "total_traders": len(traders),
            "total_trades": len(all_trades),
            "total_positions": len(all_positions),
            "trades_by_asset": self._count_by_asset(all_trades),
            "download_timestamp": datetime.now(timezone.utc).isoformat(),
        }

        logger.info(f"Download complete: {summary}")

        # Save to disk
        if save_to_disk:
            await self._save_data(traders, all_trades, all_positions, summary)

        return {
            "traders": traders,
            "trades": all_trades,
            "positions": all_positions,
            "summary": summary,
        }

    def _count_by_asset(self, trades: list[TraderTrade]) -> dict[str, int]:
        """Count trades by asset."""
        counts = {"BTC": 0, "ETH": 0, "SOL": 0, "XRP": 0, "OTHER": 0}
        for t in trades:
            if t.asset_ticker in counts:
                counts[t.asset_ticker] += 1
            else:
                counts["OTHER"] += 1
        return counts

    async def _save_data(
        self,
        traders: list[TopTrader],
        trades: list[TraderTrade],
        positions: list[TraderPosition],
        summary: dict,
    ):
        """Save downloaded data to JSON files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save traders
        traders_file = self.data_dir / f"top_traders_{timestamp}.json"
        with open(traders_file, "w") as f:
            json.dump([t.to_dict() for t in traders], f, indent=2)
        logger.info(f"Saved {len(traders)} traders to {traders_file}")

        # Save trades
        trades_file = self.data_dir / f"trader_trades_{timestamp}.json"
        with open(trades_file, "w") as f:
            json.dump([t.to_dict() for t in trades], f, indent=2)
        logger.info(f"Saved {len(trades)} trades to {trades_file}")

        # Save positions
        positions_file = self.data_dir / f"trader_positions_{timestamp}.json"
        with open(positions_file, "w") as f:
            json.dump([t.to_dict() for t in positions], f, indent=2)
        logger.info(f"Saved {len(positions)} positions to {positions_file}")

        # Save summary
        summary_file = self.data_dir / f"summary_{timestamp}.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved summary to {summary_file}")

        # Also save a latest symlink/copy
        latest_file = self.data_dir / "latest_trades.json"
        with open(latest_file, "w") as f:
            json.dump([t.to_dict() for t in trades], f, indent=2)

    # ─────────────────────────────────────────────────────────────────────────
    # ML Training Data Preparation
    # ─────────────────────────────────────────────────────────────────────────

    def prepare_ml_training_data(
        self,
        trades: Optional[list[TraderTrade]] = None,
    ) -> list[dict]:
        """
        Prepare trade data for ML training.

        Converts trades into feature vectors suitable for training.
        """
        trades = trades or self._trades

        training_data = []
        for trade in trades:
            if not trade.asset_ticker:
                continue

            features = {
                "asset": trade.asset_ticker,
                "price": trade.price,
                "size": trade.size,
                "side": 1 if trade.side == "buy" else -1,
                "outcome": 1 if trade.outcome.lower() == "yes" else 0,
                "hour": trade.timestamp.hour,
                "day_of_week": trade.timestamp.weekday(),
                # Direction implied by outcome + side
                # If buying "Yes" at low price = bullish
                # If selling "Yes" at high price = bearish
                "bullish_signal": 1 if (trade.outcome.lower() == "yes" and trade.side == "buy") else 0,
                "bearish_signal": 1 if (trade.outcome.lower() == "no" and trade.side == "buy") else 0,
            }
            training_data.append(features)

        return training_data

    def get_whale_consensus(self, asset: str = "BTC") -> dict:
        """
        Calculate whale consensus for an asset from recent trades.

        Returns:
            dict with bullish_pct, bearish_pct, total_volume, confidence
        """
        asset_trades = [t for t in self._trades if t.asset_ticker == asset]

        if not asset_trades:
            return {
                "bullish_pct": 0.5,
                "bearish_pct": 0.5,
                "total_volume": 0,
                "confidence": 0,
                "trade_count": 0,
            }

        # Count bullish vs bearish trades weighted by size
        bullish_vol = 0.0
        bearish_vol = 0.0

        for trade in asset_trades:
            if trade.outcome.lower() == "yes" and trade.side == "buy":
                bullish_vol += trade.size
            elif trade.outcome.lower() == "no" and trade.side == "buy":
                bearish_vol += trade.size
            elif trade.outcome.lower() == "yes" and trade.side == "sell":
                bearish_vol += trade.size
            elif trade.outcome.lower() == "no" and trade.side == "sell":
                bullish_vol += trade.size

        total = bullish_vol + bearish_vol
        if total == 0:
            return {
                "bullish_pct": 0.5,
                "bearish_pct": 0.5,
                "total_volume": 0,
                "confidence": 0,
                "trade_count": len(asset_trades),
            }

        bullish_pct = bullish_vol / total
        bearish_pct = bearish_vol / total

        # Confidence based on volume and consensus strength
        consensus_strength = abs(bullish_pct - 0.5) * 2  # 0-1
        volume_factor = min(1.0, total / 100000)  # Normalize by $100k
        confidence = consensus_strength * volume_factor

        return {
            "bullish_pct": bullish_pct,
            "bearish_pct": bearish_pct,
            "total_volume": total,
            "confidence": confidence,
            "trade_count": len(asset_trades),
        }


# ─────────────────────────────────────────────────────────────────────────────
# CLI / Demo
# ─────────────────────────────────────────────────────────────────────────────


async def demo():
    """Demo the top trader downloader."""
    downloader = TopTraderDownloader(
        data_dir="./trader_data",
        max_traders=20,
        max_trades_per_trader=100,
    )

    try:
        # Download all data
        result = await downloader.download_all(save_to_disk=True)

        print("\n" + "=" * 60)
        print("TOP TRADER DOWNLOAD COMPLETE")
        print("=" * 60)
        print(f"Traders: {result['summary']['total_traders']}")
        print(f"Trades: {result['summary']['total_trades']}")
        print(f"Positions: {result['summary']['total_positions']}")
        print(f"\nTrades by asset: {result['summary']['trades_by_asset']}")

        # Show whale consensus
        print("\n" + "-" * 60)
        print("WHALE CONSENSUS")
        print("-" * 60)
        for asset in ["BTC", "ETH", "SOL", "XRP"]:
            consensus = downloader.get_whale_consensus(asset)
            direction = "BULLISH" if consensus["bullish_pct"] > 0.5 else "BEARISH"
            print(
                f"{asset}: {direction} {consensus['bullish_pct']*100:.1f}% bullish, "
                f"${consensus['total_volume']:,.0f} volume, "
                f"{consensus['trade_count']} trades"
            )

        # Show training data sample
        training_data = downloader.prepare_ml_training_data()
        print(f"\nML Training samples: {len(training_data)}")
        if training_data:
            print(f"Sample: {training_data[0]}")

    finally:
        await downloader.close()


if __name__ == "__main__":
    asyncio.run(demo())
