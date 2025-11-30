#!/usr/bin/env python3
"""
🎯 PREDICTION ORACLE PRO
========================
- Deep market scanning (30 days)
- Multi-tier confidence system
- Sure-thing finder (90%+ confidence)
- Quick flip finder (high edge, ending soon)
- Value bet finder (great odds, medium confidence)
- Telegram alerts
- Paper trading with P&L tracking
"""
import asyncio
import httpx
import json
import os
import re
import sqlite3
from datetime import datetime, timezone, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# ============================================================================
# CONFIG
# ============================================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
XAI_API_KEY = os.getenv("XAI_API_KEY", "")
DB_PATH = "/root/prediction_oracle/oracle.db"

# ============================================================================
# DATABASE
# ============================================================================

def init_db():
    """Initialize SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT,
            market_id TEXT,
            question TEXT,
            direction TEXT,
            entry_price REAL,
            fair_value REAL,
            edge REAL,
            confidence TEXT,
            bet_size REAL,
            potential_payout REAL,
            reason TEXT,
            hours_left REAL,
            created_at TEXT,
            closes_at TEXT,
            resolved_at TEXT,
            outcome TEXT,
            pnl REAL,
            category TEXT
        );
        
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            markets_scanned INTEGER,
            winners_found INTEGER,
            categories TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_trades_open ON trades(outcome);
        CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id);
    """)
    conn.commit()
    return conn


def save_trade(conn, trade: dict):
    """Save a trade to database."""
    conn.execute("""
        INSERT INTO trades (source, market_id, question, direction, entry_price, 
            fair_value, edge, confidence, bet_size, potential_payout, reason,
            hours_left, created_at, closes_at, category)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        trade["source"], trade["market_id"], trade["question"], trade["direction"],
        trade["entry_price"], trade["fair_value"], trade["edge"], trade["confidence"],
        trade.get("bet_size", 5), trade.get("payout", 0), trade["reason"],
        trade["hours_left"], datetime.now(timezone.utc).isoformat(),
        trade.get("closes_at", ""), trade.get("category", "")
    ))
    conn.commit()


def get_open_trades(conn) -> list:
    """Get all open trades."""
    cursor = conn.execute("SELECT * FROM trades WHERE outcome IS NULL ORDER BY closes_at")
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]


def get_stats(conn) -> dict:
    """Get performance stats."""
    cursor = conn.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN outcome IS NOT NULL THEN pnl ELSE 0 END) as total_pnl,
            COUNT(CASE WHEN outcome IS NULL THEN 1 END) as open_trades
        FROM trades
    """)
    row = cursor.fetchone()
    return {
        "total": row[0] or 0,
        "wins": row[1] or 0,
        "losses": row[2] or 0,
        "total_pnl": row[3] or 0,
        "open_trades": row[4] or 0,
        "win_rate": (row[1] / (row[1] + row[2]) * 100) if (row[1] and row[2]) else 0
    }


# ============================================================================
# TELEGRAM
# ============================================================================

async def send_telegram(message: str, parse_mode: str = "HTML"):
    """Send message to Telegram."""
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return False
    
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.post(
                f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
                json={
                    "chat_id": TELEGRAM_CHAT_ID,
                    "text": message,
                    "parse_mode": parse_mode
                }
            )
            return resp.status_code == 200
    except:
        return False


async def alert_winner(trade: dict):
    """Send Telegram alert for a winner."""
    emoji = "🔥" if trade["edge"] >= 0.10 else "✅" if trade["edge"] >= 0.05 else "📊"
    conf_emoji = "🎯" if trade["confidence"] == "HIGH" else "📈" if trade["confidence"] == "MEDIUM" else "📉"
    
    hrs = trade["hours_left"]
    ends = f"{hrs:.0f}h" if hrs < 24 else f"{hrs/24:.1f}d"
    
    msg = f"""{emoji} <b>NEW TRADE SIGNAL</b>

<b>{trade['direction']}</b> @ {trade['entry_price']:.0%} → Fair: {trade['fair_value']:.0%}
Edge: <b>+{trade['edge']:.1%}</b> {conf_emoji} {trade['confidence']}

📋 {trade['question'][:80]}

💰 $5 → ${trade['payout']:.2f} (+${trade['payout']-5:.2f})
⏰ Ends: {ends}
🏷️ {trade['category']}

💡 {trade['reason']}"""
    
    await send_telegram(msg)


async def send_daily_summary(stats: dict, trades: list):
    """Send daily summary."""
    msg = f"""📊 <b>DAILY SUMMARY</b>

📈 Performance:
• Total Trades: {stats['total']}
• Win Rate: {stats['win_rate']:.0f}%
• P&L: ${stats['total_pnl']:.2f}

📍 Open Positions: {stats['open_trades']}

🔥 Today's Picks: {len(trades)}"""
    
    for t in trades[:5]:
        msg += f"\n• {t['direction']} {t['question'][:30]}... (+{t['edge']:.0%})"
    
    await send_telegram(msg)


# ============================================================================
# MARKET FETCHERS
# ============================================================================

async def fetch_kalshi(max_pages: int = 5) -> list:
    """Fetch Kalshi markets."""
    markets = []
    now = datetime.now(timezone.utc)
    
    async with httpx.AsyncClient(timeout=30) as client:
        cursor = None
        for _ in range(max_pages):
            params = {"limit": 100, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            
            try:
                resp = await client.get(
                    "https://api.elections.kalshi.com/trade-api/v2/markets",
                    params=params
                )
                if resp.status_code != 200:
                    break
                
                data = resp.json()
                for m in data.get("markets", []):
                    try:
                        close_str = m.get("close_time") or m.get("expected_expiration_time")
                        if not close_str:
                            continue
                        
                        close = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                        hours = (close - now).total_seconds() / 3600
                        price = m.get("last_price", 50) / 100
                        
                        if 0.03 <= price <= 0.97 and hours > 0:
                            markets.append({
                                "src": "KALSHI",
                                "id": m.get("ticker", ""),
                                "q": m.get("title", ""),
                                "price": price,
                                "hours": hours,
                                "vol": m.get("volume_24h", 0),
                                "cat": m.get("category", ""),
                                "close_time": close.isoformat()
                            })
                    except:
                        pass
                
                cursor = data.get("cursor")
                if not cursor:
                    break
            except:
                break
    
    return markets


async def fetch_polymarket(limit: int = 100) -> list:
    """Fetch Polymarket markets."""
    markets = []
    now = datetime.now(timezone.utc)
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={"closed": "false", "active": "true", "limit": limit,
                       "order": "volume24hr", "ascending": "false"}
            )
            
            if resp.status_code == 200:
                for m in resp.json():
                    try:
                        prices_str = m.get("outcomePrices", "")
                        if not prices_str:
                            continue
                        
                        prices = [float(p.strip().strip('"')) for p in prices_str.strip("[]").split(",") if p.strip()]
                        if len(prices) < 2 or not (0.03 < prices[0] < 0.97):
                            continue
                        
                        end = m.get("endDate")
                        close = datetime.fromisoformat(end.replace("Z", "+00:00")) if end else now + timedelta(days=90)
                        hours = (close - now).total_seconds() / 3600
                        
                        if hours > 0:
                            markets.append({
                                "src": "POLY",
                                "id": m.get("id", ""),
                                "q": m.get("question", ""),
                                "price": prices[0],
                                "hours": hours,
                                "vol": float(m.get("volume24hr", 0) or 0),
                                "cat": m.get("category", ""),
                                "close_time": close.isoformat()
                            })
                    except:
                        pass
        except:
            pass
    
    return markets


# ============================================================================
# GROK ANALYSIS - SMART BATCHING
# ============================================================================

async def analyze_batch(markets: list, analysis_type: str = "standard") -> list:
    """
    Analyze a batch of markets with Grok.
    analysis_type: "quick_flip", "sure_thing", "value_bet", "standard"
    """
    if not XAI_API_KEY or not markets:
        return []
    
    # Build context-aware prompt based on analysis type
    if analysis_type == "quick_flip":
        context = "Focus on markets ending SOON (<24h). Find mispriced markets where you have high conviction. Sports/events preferred."
    elif analysis_type == "sure_thing":
        context = "Find markets that are ALMOST CERTAIN to resolve one way. Look for 90%+ probability events priced wrong. Be very selective."
    elif analysis_type == "value_bet":
        context = "Find VALUE - markets where odds are significantly better than true probability. Higher edge more important than confidence."
    else:
        context = "Standard analysis - find any mispriced markets with 5%+ edge."
    
    markets_text = "\n".join([
        f"{i}. \"{m['q']}\" - {m['price']:.0%} YES, ends {m['hours']:.0f}h, vol ${m['vol']:,.0f}"
        for i, m in enumerate(markets[:12], 1)  # Max 12 per batch
    ])
    
    prompt = f"""You are an expert prediction market analyst. {context}

Markets to analyze:
{markets_text}

For each market, determine:
1. TRUE probability (fair value)
2. Direction: YES, NO, or SKIP
3. Confidence: HIGH (80%+), MEDIUM (60-80%), LOW (<60%)
4. Brief reasoning

Reply as JSON array only:
[{{"m": 1, "fv": 0.XX, "d": "YES/NO/SKIP", "c": "HIGH/MEDIUM/LOW", "r": "reason"}}]

Be accurate. Only recommend trades with real edge."""

    try:
        async with httpx.AsyncClient(timeout=90) as client:
            resp = await client.post(
                "https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_API_KEY}"},
                json={
                    "model": "grok-4-1-fast-reasoning",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800
                }
            )
            
            if resp.status_code != 200:
                print(f"Grok error: {resp.status_code}")
                return []
            
            content = resp.json()["choices"][0]["message"]["content"]
            
            # Parse JSON
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if not match:
                return []
            
            analyses = json.loads(match.group())
            results = []
            
            for a in analyses:
                idx = a.get("m", 1) - 1
                if idx >= len(markets):
                    continue
                
                m = markets[idx]
                fv = float(a.get("fv", m["price"]))
                d = a.get("d", "SKIP").upper()
                c = a.get("c", "LOW").upper()
                r = a.get("r", "")
                
                if d == "SKIP":
                    continue
                
                # Calculate edge
                if d == "YES":
                    edge = fv - m["price"]
                    entry = m["price"]
                else:  # NO
                    edge = m["price"] - fv
                    entry = 1 - m["price"]
                
                # Minimum edge threshold
                min_edge = 0.03 if c == "HIGH" else 0.05 if c == "MEDIUM" else 0.08
                
                if edge >= min_edge:
                    payout = 5 / entry if entry > 0 else 0
                    
                    results.append({
                        "source": m["src"],
                        "market_id": m["id"],
                        "question": m["q"],
                        "direction": d,
                        "entry_price": entry,
                        "fair_value": fv,
                        "edge": edge,
                        "confidence": c,
                        "reason": r,
                        "hours_left": m["hours"],
                        "volume": m["vol"],
                        "category": m["cat"] or analysis_type,
                        "payout": payout,
                        "closes_at": m.get("close_time", ""),
                        "analysis_type": analysis_type
                    })
            
            return results
            
    except Exception as e:
        print(f"Analysis error: {e}")
        return []


# ============================================================================
# SMART SCANNER
# ============================================================================

class SmartScanner:
    """Intelligent market scanner with multiple strategies."""
    
    def __init__(self):
        self.db = init_db()
        self.all_markets = []
        self.winners = []
    
    async def fetch_all_markets(self):
        """Fetch from all sources."""
        print("📡 Fetching markets...")
        
        kalshi = await fetch_kalshi(max_pages=5)  # 500 markets
        poly = await fetch_polymarket(limit=100)
        
        self.all_markets = kalshi + poly
        print(f"  Kalshi: {len(kalshi)} | Polymarket: {len(poly)} | Total: {len(self.all_markets)}")
    
    def filter_by_strategy(self, strategy: str) -> list:
        """Filter markets by strategy."""
        if strategy == "quick_flip":
            # Ending in <24h, contested prices, good volume
            return [m for m in self.all_markets 
                    if m["hours"] <= 24 and 0.25 <= m["price"] <= 0.75 and m["vol"] > 10000][:12]
        
        elif strategy == "sure_thing":
            # Extreme prices that might be wrong, any timeframe
            return [m for m in self.all_markets 
                    if (m["price"] <= 0.15 or m["price"] >= 0.85) and m["vol"] > 20000][:12]
        
        elif strategy == "value_bet":
            # Medium timeframe (1-7 days), contested, high volume
            return [m for m in self.all_markets 
                    if 24 <= m["hours"] <= 168 and 0.30 <= m["price"] <= 0.70 and m["vol"] > 50000][:12]
        
        elif strategy == "long_term":
            # 7-30 days out, any price, good volume
            return [m for m in self.all_markets 
                    if 168 <= m["hours"] <= 720 and m["vol"] > 30000][:12]
        
        else:
            # Standard - ending in 7 days, good candidates
            return sorted(
                [m for m in self.all_markets if m["hours"] <= 168],
                key=lambda x: (-x["vol"], x["hours"])
            )[:12]
    
    async def run_full_scan(self, strategies: list = None):
        """Run full multi-strategy scan."""
        if strategies is None:
            strategies = ["quick_flip", "sure_thing", "value_bet", "standard"]
        
        await self.fetch_all_markets()
        
        if not self.all_markets:
            print("❌ No markets fetched")
            return []
        
        self.winners = []
        
        for strategy in strategies:
            candidates = self.filter_by_strategy(strategy)
            if not candidates:
                print(f"\n⚪ {strategy}: No candidates")
                continue
            
            print(f"\n🔍 Analyzing {strategy} ({len(candidates)} markets)...")
            
            results = await analyze_batch(candidates, strategy)
            
            if results:
                print(f"  ✅ Found {len(results)} winners:")
                for r in results:
                    emoji = "🔥" if r["edge"] >= 0.10 else "✅" if r["edge"] >= 0.05 else "📊"
                    hrs = r["hours_left"]
                    ends = f"{hrs:.0f}h" if hrs < 24 else f"{hrs/24:.1f}d"
                    print(f"    {emoji} {r['direction']} {r['question'][:45]}...")
                    print(f"       Edge: +{r['edge']:.1%} | {r['confidence']} | Ends: {ends}")
                
                self.winners.extend(results)
            else:
                print(f"  📉 No winners in {strategy}")
            
            await asyncio.sleep(1)  # Rate limit between batches
        
        return self.winners
    
    async def save_and_alert(self):
        """Save winners and send alerts."""
        if not self.winners:
            return
        
        # Sort by edge
        self.winners.sort(key=lambda x: x["edge"], reverse=True)
        
        # Save to JSON
        with open("/root/prediction_oracle/winners.json", "w") as f:
            json.dump(self.winners, f, indent=2, default=str)
        
        # Save to database
        for w in self.winners:
            save_trade(self.db, w)
        
        print(f"\n💾 Saved {len(self.winners)} winners")
        
        # Send Telegram alerts for top picks
        top_picks = [w for w in self.winners if w["edge"] >= 0.05 or w["confidence"] == "HIGH"]
        for w in top_picks[:5]:
            await alert_winner(w)
            await asyncio.sleep(0.5)
        
        if TELEGRAM_BOT_TOKEN:
            print(f"📱 Sent {min(len(top_picks), 5)} Telegram alerts")
    
    def display_results(self):
        """Display results summary."""
        print("\n" + "=" * 70)
        print("🏆 FINAL RESULTS")
        print("=" * 70)
        
        if not self.winners:
            print("No winners found")
            return
        
        # Group by category
        by_cat = {}
        for w in self.winners:
            cat = w.get("analysis_type", "standard")
            if cat not in by_cat:
                by_cat[cat] = []
            by_cat[cat].append(w)
        
        for cat, trades in by_cat.items():
            print(f"\n📁 {cat.upper()} ({len(trades)} trades)")
            print("-" * 50)
            
            for t in sorted(trades, key=lambda x: x["edge"], reverse=True)[:5]:
                emoji = "🔥" if t["edge"] >= 0.10 else "✅"
                hrs = t["hours_left"]
                ends = f"{hrs:.0f}h" if hrs < 24 else f"{hrs/24:.1f}d"
                
                print(f"\n{emoji} {t['direction']} @ {t['entry_price']:.0%} → Fair: {t['fair_value']:.0%}")
                print(f"   Edge: +{t['edge']:.1%} | Conf: {t['confidence']} | $5→${t['payout']:.2f}")
                print(f"   [{t['source']}] {t['question'][:55]}")
                print(f"   Ends: {ends} | {t['reason']}")
        
        # Stats
        stats = get_stats(self.db)
        print(f"\n📊 PORTFOLIO STATUS")
        print(f"   Open Trades: {stats['open_trades']}")
        print(f"   Total P&L: ${stats['total_pnl']:.2f}")
        print(f"   Win Rate: {stats['win_rate']:.0f}%")


# ============================================================================
# MAIN
# ============================================================================

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prediction Oracle Pro")
    parser.add_argument("--quick", action="store_true", help="Quick flip scan only (<24h)")
    parser.add_argument("--sure", action="store_true", help="Sure thing scan only")
    parser.add_argument("--value", action="store_true", help="Value bet scan only")
    parser.add_argument("--long", action="store_true", help="Include long-term (7-30d)")
    parser.add_argument("--all", action="store_true", help="All strategies including long-term")
    parser.add_argument("--status", action="store_true", help="Show portfolio status")
    args = parser.parse_args()
    
    scanner = SmartScanner()
    
    if args.status:
        stats = get_stats(scanner.db)
        trades = get_open_trades(scanner.db)
        print("📊 PORTFOLIO STATUS")
        print(f"Open: {stats['open_trades']} | P&L: ${stats['total_pnl']:.2f} | Win: {stats['win_rate']:.0f}%")
        for t in trades[:10]:
            print(f"  • {t['direction']} {t['question'][:40]}... (+{t['edge']:.0%})")
        return
    
    # Determine strategies
    strategies = []
    if args.quick:
        strategies = ["quick_flip"]
    elif args.sure:
        strategies = ["sure_thing"]
    elif args.value:
        strategies = ["value_bet"]
    elif args.all:
        strategies = ["quick_flip", "sure_thing", "value_bet", "long_term"]
    elif args.long:
        strategies = ["quick_flip", "value_bet", "long_term"]
    else:
        strategies = ["quick_flip", "sure_thing", "value_bet"]  # Default
    
    print("🎯 PREDICTION ORACLE PRO")
    print(f"Strategies: {', '.join(strategies)}")
    print("=" * 70)
    
    await scanner.run_full_scan(strategies)
    await scanner.save_and_alert()
    scanner.display_results()
    
    print(f"\n💰 Estimated API cost: ~${len(strategies) * 0.003:.3f}")


if __name__ == "__main__":
    asyncio.run(main())
