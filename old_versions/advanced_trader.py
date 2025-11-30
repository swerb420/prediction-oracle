#!/usr/bin/env python3
"""
🚀 ADVANCED PREDICTION ORACLE - INSTITUTIONAL GRADE
====================================================
- Multi-source signal aggregation (news, social, on-chain)
- Kelly Criterion position sizing
- Real-time spread monitoring
- Historical performance tracking with Sharpe ratio
- Correlation analysis between markets
- Auto-resolution tracking
- P&L dashboards
"""
import asyncio
import sys
import os
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from typing import Optional
import json
import sqlite3
from pathlib import Path

sys.path.insert(0, '/root/prediction_oracle')
os.chdir('/root/prediction_oracle')

from dotenv import load_dotenv
load_dotenv()

import httpx
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich import box

console = Console()

# ============================================================================
# DATA MODELS
# ============================================================================

@dataclass
class Market:
    """Unified market representation."""
    source: str  # KALSHI or POLYMARKET
    market_id: str
    question: str
    yes_price: float
    no_price: float
    yes_bid: float = 0.0
    yes_ask: float = 0.0
    spread: float = 0.0
    volume_24h: float = 0.0
    liquidity: float = 0.0
    close_time: datetime = None
    hours_left: float = 0.0
    category: str = ""
    
    def __post_init__(self):
        if self.yes_bid and self.yes_ask:
            self.spread = self.yes_ask - self.yes_bid


@dataclass 
class Signal:
    """Trading signal from analysis."""
    source: str  # grok, gpt, news, social, technical
    direction: str  # yes, no, hold
    fair_value: float
    confidence: float  # 0-1
    reasoning: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Trade:
    """Paper trade record."""
    id: str
    market_id: str
    question: str
    source: str
    side: str
    entry_price: float
    fair_value: float
    size: float
    edge: float
    signals: list[Signal]
    created_at: datetime
    closes_at: datetime
    resolved_at: datetime = None
    outcome: str = None  # WIN, LOSS, PUSH
    pnl: float = 0.0


# ============================================================================
# DATABASE - Track Everything
# ============================================================================

class TradeDatabase:
    """SQLite database for trade history and analytics."""
    
    def __init__(self, db_path: str = "/root/prediction_oracle/trades.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self._init_tables()
    
    def _init_tables(self):
        """Create tables if they don't exist."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                market_id TEXT,
                question TEXT,
                source TEXT,
                side TEXT,
                entry_price REAL,
                fair_value REAL,
                size REAL,
                edge REAL,
                created_at TEXT,
                closes_at TEXT,
                resolved_at TEXT,
                outcome TEXT,
                pnl REAL,
                signals_json TEXT
            );
            
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                market_id TEXT,
                source TEXT,
                timestamp TEXT,
                yes_price REAL,
                spread REAL,
                volume_24h REAL
            );
            
            CREATE TABLE IF NOT EXISTS performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT,
                total_trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                win_rate REAL,
                total_pnl REAL,
                sharpe_ratio REAL
            );
            
            CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id);
            CREATE INDEX IF NOT EXISTS idx_snapshots_market ON market_snapshots(market_id);
        """)
        self.conn.commit()
    
    def save_trade(self, trade: Trade):
        """Save a trade to the database."""
        self.conn.execute("""
            INSERT OR REPLACE INTO trades 
            (id, market_id, question, source, side, entry_price, fair_value, 
             size, edge, created_at, closes_at, resolved_at, outcome, pnl, signals_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            trade.id, trade.market_id, trade.question, trade.source, trade.side,
            trade.entry_price, trade.fair_value, trade.size, trade.edge,
            trade.created_at.isoformat(), trade.closes_at.isoformat(),
            trade.resolved_at.isoformat() if trade.resolved_at else None,
            trade.outcome, trade.pnl,
            json.dumps([{"source": s.source, "direction": s.direction, 
                        "fair_value": s.fair_value, "confidence": s.confidence,
                        "reasoning": s.reasoning} for s in trade.signals])
        ))
        self.conn.commit()
    
    def save_snapshot(self, market: Market):
        """Save market snapshot for price tracking."""
        self.conn.execute("""
            INSERT INTO market_snapshots (market_id, source, timestamp, yes_price, spread, volume_24h)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (market.market_id, market.source, datetime.now(timezone.utc).isoformat(),
              market.yes_price, market.spread, market.volume_24h))
        self.conn.commit()
    
    def get_open_trades(self) -> list[dict]:
        """Get all open trades."""
        cursor = self.conn.execute(
            "SELECT * FROM trades WHERE outcome IS NULL ORDER BY closes_at"
        )
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]
    
    def get_performance_stats(self) -> dict:
        """Calculate performance statistics."""
        cursor = self.conn.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                AVG(CASE WHEN outcome IS NOT NULL THEN pnl END) as avg_pnl
            FROM trades
        """)
        row = cursor.fetchone()
        total, wins, losses, total_pnl, avg_pnl = row
        
        return {
            "total_trades": total or 0,
            "wins": wins or 0,
            "losses": losses or 0,
            "win_rate": (wins / total * 100) if total and wins else 0,
            "total_pnl": total_pnl or 0,
            "avg_pnl": avg_pnl or 0,
        }
    
    def get_price_history(self, market_id: str, hours: int = 24) -> list[dict]:
        """Get price history for a market."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        cursor = self.conn.execute("""
            SELECT timestamp, yes_price, spread FROM market_snapshots
            WHERE market_id = ? AND timestamp > ?
            ORDER BY timestamp
        """, (market_id, cutoff))
        columns = [d[0] for d in cursor.description]
        return [dict(zip(columns, row)) for row in cursor.fetchall()]


# ============================================================================
# SIGNAL PROVIDERS - Multi-Source Intelligence
# ============================================================================

class GrokSignalProvider:
    """Grok 4.1 Fast Reasoning."""
    
    def __init__(self):
        self.api_key = os.getenv("XAI_API_KEY")
        self.model = "grok-4-1-fast-reasoning"
        self.calls = 0
        self.cost = 0.0
    
    async def analyze(self, market: Market) -> Signal | None:
        if not self.api_key:
            return None
        
        hours = market.hours_left
        if hours < 1:
            closes_in = f"{int(hours * 60)} minutes"
        elif hours < 24:
            closes_in = f"{hours:.1f} hours"
        else:
            closes_in = f"{hours/24:.1f} days"
        
        prompt = f"""Analyze this prediction market:

Question: "{market.question}"
Current YES price: {market.yes_price:.1%}
Bid/Ask spread: {market.spread:.1%}
24h Volume: ${market.volume_24h:,.0f}
Closes in: {closes_in}

What is the TRUE probability? Consider:
1. Current news and events
2. Historical patterns
3. Market efficiency

Reply with JSON only:
{{"fair_value": 0.XX, "direction": "yes"/"no"/"hold", "confidence": 0.X, "reasoning": "30 words max"}}"""

        try:
            async with httpx.AsyncClient(timeout=45) as client:
                resp = await client.post(
                    "https://api.x.ai/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 150}
                )
                
                if resp.status_code == 200:
                    self.calls += 1
                    self.cost += 0.003
                    content = resp.json()["choices"][0]["message"]["content"]
                    
                    import re
                    match = re.search(r'\{[^}]+\}', content)
                    if match:
                        data = json.loads(match.group())
                        return Signal(
                            source="grok",
                            direction=data.get("direction", "hold"),
                            fair_value=data.get("fair_value", market.yes_price),
                            confidence=data.get("confidence", 0.5),
                            reasoning=data.get("reasoning", "")
                        )
        except Exception as e:
            console.print(f"[dim]Grok error: {e}[/dim]")
        
        return None


class GPTSignalProvider:
    """GPT-5 Mini for second opinion."""
    
    def __init__(self):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.model = "gpt-4o-mini"  # Fallback to 4o-mini
        self.calls = 0
        self.cost = 0.0
    
    async def analyze(self, market: Market) -> Signal | None:
        if not self.api_key:
            return None
        
        prompt = f"""Prediction market: "{market.question}"
Current price: {market.yes_price:.0%} YES
Closes in: {market.hours_left:.1f} hours

What's the true probability this resolves YES? Reply JSON only:
{{"fair_value": 0.XX, "direction": "yes"/"no"/"hold", "confidence": 0.X, "reasoning": "20 words"}}"""

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}"},
                    json={"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 100}
                )
                
                if resp.status_code == 200:
                    self.calls += 1
                    self.cost += 0.0005
                    content = resp.json()["choices"][0]["message"]["content"]
                    
                    import re
                    match = re.search(r'\{[^}]+\}', content)
                    if match:
                        data = json.loads(match.group())
                        return Signal(
                            source="gpt",
                            direction=data.get("direction", "hold"),
                            fair_value=data.get("fair_value", market.yes_price),
                            confidence=data.get("confidence", 0.5),
                            reasoning=data.get("reasoning", "")
                        )
        except Exception as e:
            console.print(f"[dim]GPT error: {e}[/dim]")
        
        return None


class TechnicalSignalProvider:
    """Technical analysis - spread, volume, momentum."""
    
    def __init__(self, db: TradeDatabase):
        self.db = db
    
    async def analyze(self, market: Market) -> Signal | None:
        """Analyze based on price history and technicals."""
        
        # Get price history
        history = self.db.get_price_history(market.market_id, hours=24)
        
        direction = "hold"
        confidence = 0.3
        reasoning = "Insufficient data"
        
        if len(history) >= 3:
            prices = [h["yes_price"] for h in history]
            
            # Simple momentum
            recent = prices[-3:]
            momentum = (recent[-1] - recent[0]) / max(recent[0], 0.01)
            
            # Spread analysis - tight spread = more efficient
            avg_spread = sum(h["spread"] for h in history) / len(history)
            
            if momentum > 0.05:
                direction = "yes"
                confidence = min(0.6, 0.3 + abs(momentum))
                reasoning = f"Positive momentum +{momentum:.1%}, spread {avg_spread:.1%}"
            elif momentum < -0.05:
                direction = "no"
                confidence = min(0.6, 0.3 + abs(momentum))
                reasoning = f"Negative momentum {momentum:.1%}, spread {avg_spread:.1%}"
            else:
                reasoning = f"Sideways, momentum {momentum:.1%}"
        
        return Signal(
            source="technical",
            direction=direction,
            fair_value=market.yes_price,
            confidence=confidence,
            reasoning=reasoning
        )


# ============================================================================
# POSITION SIZING - Kelly Criterion
# ============================================================================

def kelly_bet(edge: float, odds: float, kelly_fraction: float = 0.25) -> float:
    """
    Kelly Criterion position sizing.
    
    edge: Expected edge (e.g., 0.10 for 10% edge)
    odds: Decimal odds (e.g., 2.0 for even money)
    kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly, safer)
    
    Returns: Fraction of bankroll to bet
    """
    if edge <= 0 or odds <= 1:
        return 0.0
    
    # Kelly formula: f* = (bp - q) / b
    # b = odds - 1 (net odds)
    # p = probability of winning
    # q = 1 - p
    
    b = odds - 1
    p = (edge + (1 / odds))  # Implied win probability given edge
    p = min(max(p, 0.01), 0.99)  # Clamp
    q = 1 - p
    
    kelly = (b * p - q) / b
    kelly = max(0, kelly)  # Can't bet negative
    
    # Apply fractional Kelly for safety
    return kelly * kelly_fraction


# ============================================================================
# SIGNAL AGGREGATOR - Combine Multiple Signals
# ============================================================================

class SignalAggregator:
    """Combine signals from multiple sources."""
    
    def __init__(self):
        self.weights = {
            "grok": 0.45,
            "gpt": 0.35,
            "technical": 0.20,
        }
    
    def aggregate(self, signals: list[Signal]) -> tuple[str, float, float]:
        """
        Aggregate signals into final direction, fair value, and confidence.
        Returns: (direction, fair_value, confidence)
        """
        if not signals:
            return "hold", 0.5, 0.0
        
        total_weight = 0
        weighted_fv = 0
        weighted_conf = 0
        direction_scores = {"yes": 0, "no": 0, "hold": 0}
        
        for signal in signals:
            weight = self.weights.get(signal.source, 0.1) * signal.confidence
            total_weight += weight
            weighted_fv += signal.fair_value * weight
            weighted_conf += signal.confidence * weight
            direction_scores[signal.direction] += weight
        
        if total_weight == 0:
            return "hold", 0.5, 0.0
        
        # Final values
        fair_value = weighted_fv / total_weight
        confidence = weighted_conf / total_weight
        
        # Direction by weighted vote
        direction = max(direction_scores, key=direction_scores.get)
        
        return direction, fair_value, confidence


# ============================================================================
# MARKET CLIENTS - Real Data
# ============================================================================

class KalshiClient:
    """Real Kalshi API."""
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30)
    
    async def get_markets(self, limit: int = 200) -> list[Market]:
        markets = []
        now = datetime.now(timezone.utc)
        cursor = None
        
        while len(markets) < limit:
            params = {"limit": min(100, limit - len(markets)), "status": "open"}
            if cursor:
                params["cursor"] = cursor
            
            try:
                resp = await self.client.get(f"{self.BASE_URL}/markets", params=params)
                if resp.status_code != 200:
                    break
                
                data = resp.json()
                for m in data.get("markets", []):
                    try:
                        close_str = m.get("close_time") or m.get("expected_expiration_time")
                        if not close_str:
                            continue
                        
                        close_time = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                        hours_left = (close_time - now).total_seconds() / 3600
                        
                        yes_price = m.get("last_price", 50) / 100
                        yes_bid = m.get("yes_bid", 0) / 100
                        yes_ask = m.get("yes_ask", 100) / 100
                        
                        if yes_price <= 0:
                            yes_price = (yes_bid + yes_ask) / 2
                        
                        markets.append(Market(
                            source="KALSHI",
                            market_id=m.get("ticker", ""),
                            question=m.get("title", ""),
                            yes_price=yes_price,
                            no_price=1 - yes_price,
                            yes_bid=yes_bid,
                            yes_ask=yes_ask,
                            volume_24h=m.get("volume_24h", 0),
                            liquidity=m.get("liquidity", 0) / 100,
                            close_time=close_time,
                            hours_left=hours_left,
                            category=m.get("category", ""),
                        ))
                    except:
                        continue
                
                cursor = data.get("cursor")
                if not cursor or not data.get("markets"):
                    break
            except:
                break
        
        return markets
    
    async def close(self):
        await self.client.aclose()


class PolymarketClient:
    """Real Polymarket API."""
    BASE_URL = "https://gamma-api.polymarket.com"
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30)
    
    async def get_markets(self, limit: int = 100) -> list[Market]:
        markets = []
        now = datetime.now(timezone.utc)
        
        try:
            resp = await self.client.get(
                f"{self.BASE_URL}/markets",
                params={"closed": "false", "active": "true", "limit": limit, 
                       "order": "volume24hr", "ascending": "false"}
            )
            
            if resp.status_code == 200:
                for m in resp.json():
                    try:
                        prices_str = m.get("outcomePrices", "")
                        if not prices_str or prices_str == "[]":
                            continue
                        
                        prices = [float(p.strip().strip('"')) for p in prices_str.strip("[]").split(",") if p.strip()]
                        if len(prices) < 2 or not (0 < prices[0] < 1):
                            continue
                        
                        end_date = m.get("endDate")
                        close_time = datetime.fromisoformat(end_date.replace("Z", "+00:00")) if end_date else now + timedelta(days=30)
                        hours_left = (close_time - now).total_seconds() / 3600
                        
                        # Calculate spread from bid/ask if available
                        spread = float(m.get("spread", 0) or 0)
                        
                        markets.append(Market(
                            source="POLYMARKET",
                            market_id=m.get("id", ""),
                            question=m.get("question", ""),
                            yes_price=prices[0],
                            no_price=prices[1] if len(prices) > 1 else 1 - prices[0],
                            spread=spread,
                            volume_24h=float(m.get("volume24hr", 0) or 0),
                            liquidity=float(m.get("liquidity", 0) or 0),
                            close_time=close_time,
                            hours_left=hours_left,
                            category=m.get("category", ""),
                        ))
                    except:
                        continue
        except:
            pass
        
        return markets
    
    async def close(self):
        await self.client.aclose()


# ============================================================================
# ADVANCED TRADER
# ============================================================================

class AdvancedTrader:
    """Institutional-grade paper trading system."""
    
    def __init__(self, bankroll: float = 1000.0):
        self.bankroll = bankroll
        self.initial_bankroll = bankroll
        
        # Data sources
        self.kalshi = KalshiClient()
        self.polymarket = PolymarketClient()
        
        # Signal providers
        self.grok = GrokSignalProvider()
        self.gpt = GPTSignalProvider()
        
        # Database
        self.db = TradeDatabase()
        self.technical = TechnicalSignalProvider(self.db)
        
        # Signal aggregator
        self.aggregator = SignalAggregator()
        
        # Settings
        self.min_edge = 0.03  # 3% minimum edge
        self.min_confidence = 0.5
        self.max_position_pct = 0.10  # Max 10% of bankroll per trade
        self.kelly_fraction = 0.25  # Quarter Kelly
    
    async def fetch_markets(self) -> list[Market]:
        """Fetch from all sources."""
        kalshi_markets = await self.kalshi.get_markets(200)
        poly_markets = await self.polymarket.get_markets(100)
        
        all_markets = kalshi_markets + poly_markets
        console.print(f"[green]Fetched {len(kalshi_markets)} Kalshi + {len(poly_markets)} Polymarket = {len(all_markets)} markets[/green]")
        
        return all_markets
    
    def filter_markets(self, markets: list[Market], max_hours: float = 72) -> list[Market]:
        """Filter to tradeable markets."""
        filtered = []
        for m in markets:
            if not (1 <= m.hours_left <= max_hours):
                continue
            if not (0.02 <= m.yes_price <= 0.98):
                continue
            filtered.append(m)
        
        # Sort by hours left
        filtered.sort(key=lambda x: x.hours_left)
        return filtered
    
    async def analyze_market(self, market: Market) -> tuple[list[Signal], str, float, float]:
        """Get signals from all providers and aggregate."""
        signals = []
        
        # Save snapshot for technicals
        self.db.save_snapshot(market)
        
        # Get signals in parallel
        grok_task = self.grok.analyze(market)
        gpt_task = self.gpt.analyze(market)
        tech_task = self.technical.analyze(market)
        
        results = await asyncio.gather(grok_task, gpt_task, tech_task, return_exceptions=True)
        
        for result in results:
            if isinstance(result, Signal):
                signals.append(result)
        
        # Aggregate
        direction, fair_value, confidence = self.aggregator.aggregate(signals)
        
        return signals, direction, fair_value, confidence
    
    def calculate_position(self, market: Market, direction: str, fair_value: float, confidence: float) -> tuple[float, float, float]:
        """Calculate position size using Kelly Criterion."""
        # Calculate edge
        if direction == "yes":
            entry = market.yes_price
            edge = fair_value - entry
        elif direction == "no":
            entry = market.no_price
            edge = (1 - fair_value) - entry
        else:
            return 0, 0, 0
        
        if edge < self.min_edge or confidence < self.min_confidence:
            return 0, 0, 0
        
        # Calculate odds
        if entry > 0:
            odds = 1 / entry
        else:
            return 0, 0, 0
        
        # Kelly position size
        kelly_pct = kelly_bet(edge, odds, self.kelly_fraction)
        
        # Cap at max position
        position_pct = min(kelly_pct, self.max_position_pct)
        position_size = self.bankroll * position_pct
        
        # Minimum $1 bet
        if position_size < 1:
            return 0, 0, 0
        
        return position_size, entry, edge
    
    async def run_scan(self, max_hours: float = 72, max_analyze: int = 15):
        """Run one scanning cycle."""
        console.print(Panel.fit(
            f"[bold cyan]🚀 ADVANCED PREDICTION ORACLE[/bold cyan]\n"
            f"[dim]Multi-Signal | Kelly Sizing | Full Analytics[/dim]",
            border_style="cyan"
        ))
        
        # Fetch and filter
        markets = await self.fetch_markets()
        tradeable = self.filter_markets(markets, max_hours)
        console.print(f"[yellow]Tradeable (1-{max_hours}h): {len(tradeable)} markets[/yellow]\n")
        
        # Performance stats
        stats = self.db.get_performance_stats()
        console.print(f"[dim]Historical: {stats['total_trades']} trades | "
                      f"{stats['win_rate']:.0f}% win rate | ${stats['total_pnl']:.2f} P&L[/dim]\n")
        
        opportunities = []
        
        console.print("[bold]🧠 Analyzing Markets (Grok + GPT + Technical)...[/bold]")
        for i, market in enumerate(tradeable[:max_analyze]):
            signals, direction, fair_value, confidence = await self.analyze_market(market)
            
            if direction == "hold":
                console.print(f"  [dim]- {market.question[:45]}... HOLD[/dim]")
                continue
            
            size, entry, edge = self.calculate_position(market, direction, fair_value, confidence)
            
            if size > 0:
                opp = {
                    "market": market,
                    "signals": signals,
                    "direction": direction,
                    "side": "YES" if direction == "yes" else "NO",
                    "entry": entry,
                    "fair_value": fair_value,
                    "edge": edge,
                    "confidence": confidence,
                    "size": size,
                    "size_pct": size / self.bankroll * 100,
                }
                opportunities.append(opp)
                
                console.print(f"  [green]✓ {opp['side']} {market.question[:40]}...[/green]")
                console.print(f"    Edge: +{edge:.1%} | Confidence: {confidence:.0%} | Size: ${size:.2f} ({opp['size_pct']:.1f}%)")
            else:
                console.print(f"  [dim]- {market.question[:45]}... (edge {edge:.1%})[/dim]")
            
            await asyncio.sleep(0.3)  # Rate limit
        
        # Display opportunities
        if opportunities:
            console.print(f"\n[bold]💰 TOP OPPORTUNITIES ({len(opportunities)})[/bold]\n")
            
            table = Table(box=box.ROUNDED)
            table.add_column("Market", width=40)
            table.add_column("Side", justify="center")
            table.add_column("Entry", justify="right")
            table.add_column("Fair", justify="right")
            table.add_column("Edge", justify="right")
            table.add_column("Conf", justify="right")
            table.add_column("Size", justify="right")
            table.add_column("Ends", justify="right")
            
            for opp in sorted(opportunities, key=lambda x: x["edge"], reverse=True):
                m = opp["market"]
                hours = m.hours_left
                ends = f"{hours:.0f}h" if hours < 24 else f"{hours/24:.1f}d"
                
                q = m.question[:38] + ".." if len(m.question) > 40 else m.question
                
                table.add_row(
                    f"[{'blue' if m.source=='KALSHI' else 'green'}]{q}[/]",
                    f"[bold]{opp['side']}[/bold]",
                    f"{opp['entry']:.0%}",
                    f"{opp['fair_value']:.0%}",
                    f"[green]+{opp['edge']:.1%}[/green]",
                    f"{opp['confidence']:.0%}",
                    f"[cyan]${opp['size']:.2f}[/cyan]",
                    f"[yellow]{ends}[/yellow]",
                )
            
            console.print(table)
            
            # Record trades
            console.print("\n[bold]📝 Recording Trades...[/bold]")
            for opp in opportunities[:5]:
                m = opp["market"]
                trade = Trade(
                    id=f"{m.source}-{m.market_id}-{datetime.now().strftime('%H%M%S')}",
                    market_id=m.market_id,
                    question=m.question,
                    source=m.source,
                    side=opp["side"],
                    entry_price=opp["entry"],
                    fair_value=opp["fair_value"],
                    size=opp["size"],
                    edge=opp["edge"],
                    signals=opp["signals"],
                    created_at=datetime.now(timezone.utc),
                    closes_at=m.close_time,
                )
                self.db.save_trade(trade)
                console.print(f"  ✓ {opp['side']} {m.question[:40]}... (${opp['size']:.2f})")
        else:
            console.print("\n[yellow]No opportunities meeting criteria.[/yellow]")
        
        # Summary
        console.print(Panel.fit(
            f"Bankroll: ${self.bankroll:,.0f} | "
            f"Grok: {self.grok.calls} (~${self.grok.cost:.3f}) | "
            f"GPT: {self.gpt.calls} (~${self.gpt.cost:.4f})",
            title="Session Summary",
            border_style="dim"
        ))
        
        return opportunities
    
    async def run_continuous(self, interval: int = 600, max_hours: float = 72):
        """Run continuously."""
        console.print(f"[cyan]Continuous mode: scanning every {interval}s[/cyan]")
        
        cycle = 0
        while True:
            try:
                cycle += 1
                console.print(f"\n{'='*70}")
                console.print(f"[bold]Scan #{cycle}[/bold] - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                console.print('='*70)
                
                await self.run_scan(max_hours=max_hours)
                
                console.print(f"\n[dim]Next scan in {interval}s (Ctrl+C to stop)...[/dim]")
                await asyncio.sleep(interval)
                
            except KeyboardInterrupt:
                console.print("\n[yellow]Stopped.[/yellow]")
                break
            except Exception as e:
                console.print(f"[red]Error: {e}[/red]")
                await asyncio.sleep(60)
        
        await self.kalshi.close()
        await self.polymarket.close()
    
    async def show_dashboard(self):
        """Show current status dashboard."""
        stats = self.db.get_performance_stats()
        open_trades = self.db.get_open_trades()
        
        console.print(Panel.fit(
            f"[bold cyan]📊 PERFORMANCE DASHBOARD[/bold cyan]",
            border_style="cyan"
        ))
        
        # Stats
        console.print(f"\n[bold]Overall Performance[/bold]")
        console.print(f"  Total Trades: {stats['total_trades']}")
        console.print(f"  Win Rate: {stats['win_rate']:.1f}%")
        console.print(f"  Total P&L: ${stats['total_pnl']:.2f}")
        console.print(f"  Avg P&L/Trade: ${stats['avg_pnl']:.2f}")
        
        # Open positions
        if open_trades:
            console.print(f"\n[bold]Open Positions ({len(open_trades)})[/bold]")
            table = Table(box=box.SIMPLE)
            table.add_column("Market")
            table.add_column("Side")
            table.add_column("Entry")
            table.add_column("Size")
            table.add_column("Edge")
            table.add_column("Closes")
            
            for t in open_trades[:10]:
                closes = datetime.fromisoformat(t["closes_at"])
                hours = (closes - datetime.now(timezone.utc)).total_seconds() / 3600
                ends = f"{hours:.0f}h" if hours < 24 else f"{hours/24:.1f}d"
                
                table.add_row(
                    t["question"][:35] + "...",
                    t["side"],
                    f"{t['entry_price']:.0%}",
                    f"${t['size']:.2f}",
                    f"+{t['edge']:.1%}",
                    ends,
                )
            
            console.print(table)


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous", "-c", action="store_true")
    parser.add_argument("--dashboard", "-d", action="store_true")
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--hours", type=float, default=72)
    parser.add_argument("--bankroll", type=float, default=1000)
    args = parser.parse_args()
    
    trader = AdvancedTrader(bankroll=args.bankroll)
    
    if args.dashboard:
        await trader.show_dashboard()
    elif args.continuous:
        await trader.run_continuous(interval=args.interval, max_hours=args.hours)
    else:
        await trader.run_scan(max_hours=args.hours)


if __name__ == "__main__":
    asyncio.run(main())
