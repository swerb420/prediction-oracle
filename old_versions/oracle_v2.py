#!/usr/bin/env python3
"""
🎯 PREDICTION ORACLE V2 - SMART ALERTS
======================================
- Tiered betting strategy (scalps, value, longshots, whales)
- Rich Telegram notifications with full trade details
- Smart position sizing based on confidence & edge
- Manual alert for high-conviction plays
- SQLite tracking with full analytics
"""
import asyncio
import sqlite3
import httpx
import json
import os
import re
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

load_dotenv("/root/prediction_oracle/.env")

# ============================================================================
# CONFIG
# ============================================================================

DB_PATH = "/root/prediction_oracle/oracle_v2.db"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
XAI_KEY = os.getenv("XAI_API_KEY", "")

# Bankroll & Sizing
BANKROLL = 1000.0
BASE_BET = 5.0

# Bet tiers
TIERS = {
    "WHALE": {  # High confidence, high edge - ALERT FOR MANUAL BET
        "min_edge": 0.08,
        "min_confidence": "HIGH",
        "bet_mult": 4.0,  # $20 suggested
        "emoji": "🐋",
        "alert_manual": True,
    },
    "SCALP": {  # Sure things - small guaranteed profits
        "min_edge": 0.03,
        "min_confidence": "HIGH",
        "max_odds": 1.5,  # Only bet on heavy favorites
        "bet_mult": 2.0,  # $10
        "emoji": "💰",
        "alert_manual": False,
    },
    "VALUE": {  # Good edge, medium confidence
        "min_edge": 0.05,
        "min_confidence": "MEDIUM",
        "bet_mult": 1.5,  # $7.50
        "emoji": "📈",
        "alert_manual": False,
    },
    "LONGSHOT": {  # High edge, lower prices - big potential
        "min_edge": 0.10,
        "min_price": 0.10,  # Entry at 10-35%
        "max_price": 0.35,
        "bet_mult": 1.0,  # $5
        "emoji": "🎰",
        "alert_manual": False,
    },
}

# ============================================================================
# DATABASE
# ============================================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tier TEXT,
            source TEXT,
            market_id TEXT UNIQUE,
            question TEXT,
            direction TEXT,
            entry_price REAL,
            fair_value REAL,
            edge REAL,
            confidence TEXT,
            bet_size REAL,
            potential_payout REAL,
            potential_profit REAL,
            risk_reward REAL,
            reason TEXT,
            hours_left REAL,
            volume REAL,
            created_at TEXT,
            closes_at TEXT,
            resolved_at TEXT,
            outcome TEXT,
            pnl REAL
        );
        
        CREATE TABLE IF NOT EXISTS daily_stats (
            date TEXT PRIMARY KEY,
            trades_opened INTEGER,
            trades_closed INTEGER,
            wins INTEGER,
            losses INTEGER,
            pnl REAL,
            best_trade TEXT,
            worst_trade TEXT
        );
        
        CREATE INDEX IF NOT EXISTS idx_trades_tier ON trades(tier);
        CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome);
    """)
    conn.commit()
    return conn

def save_trade(db, t: dict) -> bool:
    """Save trade, return True if new."""
    try:
        db.execute("""
            INSERT INTO trades (tier, source, market_id, question, direction, 
                entry_price, fair_value, edge, confidence, bet_size, potential_payout,
                potential_profit, risk_reward, reason, hours_left, volume, created_at, closes_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            t["tier"], t["source"], t["market_id"], t["question"], t["direction"],
            t["entry_price"], t["fair_value"], t["edge"], t["confidence"],
            t["bet_size"], t["payout"], t["profit"], t["rr"],
            t["reason"], t["hours_left"], t.get("volume", 0),
            datetime.now(timezone.utc).isoformat(), t.get("closes_at", "")
        ))
        db.commit()
        return True
    except sqlite3.IntegrityError:
        return False  # Already exists

def get_open_trades(db) -> list:
    cursor = db.execute("""
        SELECT * FROM trades WHERE outcome IS NULL ORDER BY created_at DESC
    """)
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]

def get_trades_by_tier(db) -> dict:
    """Get open trades grouped by tier."""
    trades = get_open_trades(db)
    by_tier = {}
    for t in trades:
        tier = t.get("tier", "VALUE")
        if tier not in by_tier:
            by_tier[tier] = []
        by_tier[tier].append(t)
    return by_tier

def resolve_trade(db, market_id: str, outcome: str, pnl: float):
    db.execute("""
        UPDATE trades SET outcome = ?, pnl = ?, resolved_at = ?
        WHERE market_id = ?
    """, (outcome, pnl, datetime.now(timezone.utc).isoformat(), market_id))
    db.commit()

def get_stats(db) -> dict:
    cursor = db.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN outcome IS NOT NULL THEN pnl ELSE 0 END) as realized_pnl,
            SUM(CASE WHEN outcome IS NULL THEN bet_size ELSE 0 END) as at_risk,
            SUM(CASE WHEN outcome IS NULL THEN potential_profit ELSE 0 END) as potential_profit,
            COUNT(CASE WHEN outcome IS NULL THEN 1 END) as open_count
        FROM trades
    """)
    r = cursor.fetchone()
    return {
        "total": r[0] or 0, "wins": r[1] or 0, "losses": r[2] or 0,
        "realized_pnl": r[3] or 0, "at_risk": r[4] or 0,
        "potential_profit": r[5] or 0, "open": r[6] or 0,
        "win_rate": (r[1]/(r[1]+r[2])*100) if r[1] and r[2] else 0
    }

def get_tier_stats(db) -> dict:
    """Get stats by tier."""
    cursor = db.execute("""
        SELECT tier,
            COUNT(*) as total,
            SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome IS NOT NULL THEN pnl ELSE 0 END) as pnl
        FROM trades GROUP BY tier
    """)
    stats = {}
    for row in cursor.fetchall():
        tier, total, wins, pnl = row
        stats[tier] = {"total": total, "wins": wins or 0, "pnl": pnl or 0,
                       "win_rate": (wins/total*100) if wins and total else 0}
    return stats

# ============================================================================
# TELEGRAM - RICH MESSAGES
# ============================================================================

async def telegram(msg: str, urgent: bool = False):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        print(f"[TG] {msg[:100]}...")
        return
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            payload = {"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "HTML"}
            if urgent:
                payload["disable_notification"] = False
            await c.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage", json=payload)
    except Exception as e:
        print(f"Telegram error: {e}")

async def alert_new_trade(t: dict):
    """Send detailed trade alert."""
    tier_info = TIERS.get(t["tier"], {})
    emoji = tier_info.get("emoji", "📊")
    urgent = tier_info.get("alert_manual", False)
    
    # Time formatting
    hrs = t["hours_left"]
    if hrs < 1:
        time_str = f"{int(hrs*60)}m"
    elif hrs < 24:
        time_str = f"{hrs:.1f}h"
    else:
        time_str = f"{hrs/24:.1f}d"
    
    # Build message
    msg = f"""{emoji} <b>{'🚨 WHALE ALERT - MANUAL BET!' if urgent else 'NEW TRADE'}</b> {emoji}

<b>Tier:</b> {t['tier']}
<b>Direction:</b> {t['direction']}

📋 <b>{t['question'][:70]}</b>

┌─────────────────────────
│ 💵 <b>Entry:</b> {t['entry_price']:.1%}
│ 🎯 <b>Fair Value:</b> {t['fair_value']:.1%}  
│ 📊 <b>Edge:</b> +{t['edge']:.1%}
│ 🔒 <b>Confidence:</b> {t['confidence']}
├─────────────────────────
│ 💰 <b>Bet Size:</b> ${t['bet_size']:.2f}
│ 🏆 <b>Win Payout:</b> ${t['payout']:.2f}
│ ✅ <b>Profit if Win:</b> +${t['profit']:.2f}
│ ❌ <b>Loss if Wrong:</b> -${t['bet_size']:.2f}
│ ⚖️ <b>Risk:Reward:</b> 1:{t['rr']:.1f}
├─────────────────────────
│ ⏰ <b>Closes:</b> {time_str}
│ 📈 <b>Volume:</b> ${t.get('volume', 0):,.0f}
│ 🏷️ <b>Source:</b> {t['source']}
└─────────────────────────

💡 <i>{t['reason']}</i>"""

    if urgent:
        msg += "\n\n🚨 <b>HIGH CONVICTION - CONSIDER LARGER BET!</b> 🚨"
    
    await telegram(msg, urgent=urgent)

async def alert_resolution(t: dict, outcome: str, pnl: float):
    """Send resolution alert."""
    emoji = "🎉" if outcome == "WIN" else "❌"
    
    msg = f"""{emoji} <b>TRADE RESOLVED: {outcome}</b> {emoji}

📋 {t['question'][:60]}...

<b>Direction:</b> {t['direction']}
<b>Entry:</b> {t['entry_price']:.1%}
<b>Bet:</b> ${t['bet_size']:.2f}

<b>P&L: {'🟢' if pnl > 0 else '🔴'} ${pnl:+.2f}</b>"""
    
    await telegram(msg)

async def send_portfolio_summary(db):
    """Send full portfolio summary."""
    stats = get_stats(db)
    tier_stats = get_tier_stats(db)
    trades = get_open_trades(db)
    by_tier = get_trades_by_tier(db)
    
    msg = f"""📊 <b>PORTFOLIO SUMMARY</b>

💼 <b>Overview</b>
├ Open Positions: {stats['open']}
├ At Risk: ${stats['at_risk']:.2f}
├ Potential Profit: ${stats['potential_profit']:.2f}
└ Realized P&L: ${stats['realized_pnl']:+.2f}

📈 <b>Performance</b>
├ Total Trades: {stats['total']}
├ Wins: {stats['wins']} | Losses: {stats['losses']}
└ Win Rate: {stats['win_rate']:.0f}%

"""
    
    # Tier breakdown
    msg += "🏷️ <b>By Tier</b>\n"
    for tier, info in tier_stats.items():
        emoji = TIERS.get(tier, {}).get("emoji", "📊")
        msg += f"├ {emoji} {tier}: {info['total']} trades | {info['win_rate']:.0f}% win | ${info['pnl']:+.2f}\n"
    
    # Open positions
    if trades:
        msg += "\n📍 <b>Open Positions</b>\n"
        for t in trades[:8]:
            emoji = TIERS.get(t.get("tier"), {}).get("emoji", "📊")
            hrs = t.get("hours_left", 0)
            time_str = f"{hrs:.0f}h" if hrs < 24 else f"{hrs/24:.1f}d"
            msg += f"\n{emoji} <b>{t['direction']}</b> @ {t['entry_price']:.0%} → {t['fair_value']:.0%}\n"
            msg += f"   {t['question'][:40]}...\n"
            msg += f"   Edge: +{t['edge']:.1%} | ${t['bet_size']:.0f}→${t['potential_payout']:.0f} | {time_str}\n"
    
    await telegram(msg)

# ============================================================================
# MARKET FETCHERS
# ============================================================================

async def fetch_all_markets() -> list:
    """Fetch from Kalshi + Polymarket."""
    markets = []
    now = datetime.now(timezone.utc)
    
    # Kalshi (500 markets)
    async with httpx.AsyncClient(timeout=30) as c:
        cursor = None
        for _ in range(5):
            params = {"limit": 100, "status": "open"}
            if cursor:
                params["cursor"] = cursor
            try:
                resp = await c.get("https://api.elections.kalshi.com/trade-api/v2/markets", params=params)
                if resp.status_code != 200:
                    break
                data = resp.json()
                for m in data.get("markets", []):
                    close_str = m.get("close_time") or m.get("expected_expiration_time")
                    if not close_str:
                        continue
                    try:
                        close = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                        hours = (close - now).total_seconds() / 3600
                        price = m.get("last_price", 50) / 100
                        if 0.03 <= price <= 0.97 and hours > 0:
                            markets.append({
                                "src": "KALSHI", "id": m.get("ticker"), "q": m.get("title"),
                                "price": price, "hours": hours, "vol": m.get("volume_24h", 0),
                                "close_time": close.isoformat()
                            })
                    except:
                        pass
                cursor = data.get("cursor")
                if not cursor:
                    break
            except:
                break
    
    # Polymarket
    async with httpx.AsyncClient(timeout=30) as c:
        try:
            resp = await c.get("https://gamma-api.polymarket.com/markets",
                params={"closed": "false", "active": "true", "limit": 100,
                       "order": "volume24hr", "ascending": "false"})
            if resp.status_code == 200:
                for m in resp.json():
                    try:
                        prices = m.get("outcomePrices", "")
                        if not prices:
                            continue
                        p = [float(x.strip().strip('"')) for x in prices.strip("[]").split(",") if x.strip()]
                        if len(p) < 2 or not (0.03 < p[0] < 0.97):
                            continue
                        end = m.get("endDate")
                        close = datetime.fromisoformat(end.replace("Z", "+00:00")) if end else now + timedelta(days=30)
                        hours = (close - now).total_seconds() / 3600
                        if hours > 0:
                            markets.append({
                                "src": "POLY", "id": m.get("id"), "q": m.get("question"),
                                "price": p[0], "hours": hours, 
                                "vol": float(m.get("volume24hr", 0) or 0),
                                "close_time": close.isoformat()
                            })
                    except:
                        pass
        except:
            pass
    
    return markets

# ============================================================================
# SMART ANALYZER
# ============================================================================

async def analyze_for_tier(markets: list, tier: str) -> list:
    """Analyze markets looking for specific tier opportunities."""
    if not XAI_KEY or not markets:
        return []
    
    tier_info = TIERS.get(tier, {})
    
    # Context for Grok based on tier
    if tier == "WHALE":
        context = "Find VERY HIGH CONFIDENCE plays. Only recommend if you're 80%+ sure. These are for large bets."
    elif tier == "SCALP":
        context = "Find SAFE bets on heavy favorites. High confidence, lower edge is OK. Looking for reliable small wins."
    elif tier == "LONGSHOT":
        context = "Find UNDERVALUED underdogs. Markets priced 10-35% that should be higher. High potential payouts."
    else:  # VALUE
        context = "Find good VALUE bets. Solid edge with reasonable confidence."
    
    markets_text = "\n".join([
        f"{i}. \"{m['q']}\" - {m['price']:.0%} YES, {m['hours']:.0f}h, ${m['vol']:,.0f} vol"
        for i, m in enumerate(markets[:12], 1)
    ])
    
    prompt = f"""You are an expert prediction market analyst. {context}

Markets:
{markets_text}

For each, provide TRUE probability and recommendation.
Be ACCURATE - only recommend trades with REAL edge.

JSON array: [{{"m":1,"fv":0.XX,"d":"YES/NO/SKIP","c":"HIGH/MEDIUM/LOW","r":"10 word reason"}}]"""

    try:
        async with httpx.AsyncClient(timeout=90) as c:
            resp = await c.post("https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_KEY}"},
                json={"model": "grok-4-1-fast-reasoning", 
                      "messages": [{"role": "user", "content": prompt}], 
                      "max_tokens": 800})
            
            if resp.status_code != 200:
                return []
            
            content = resp.json()["choices"][0]["message"]["content"]
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if not match:
                return []
            
            results = []
            for a in json.loads(match.group()):
                idx = a.get("m", 1) - 1
                if idx >= len(markets):
                    continue
                
                m = markets[idx]
                fv = float(a.get("fv", m["price"]))
                d = a.get("d", "SKIP").upper()
                conf = a.get("c", "MEDIUM").upper()
                reason = a.get("r", "")
                
                if d == "SKIP":
                    continue
                
                # Calculate edge and entry
                if d == "YES":
                    edge = fv - m["price"]
                    entry = m["price"]
                else:
                    edge = m["price"] - fv
                    entry = 1 - m["price"]
                
                if entry <= 0 or entry >= 1:
                    continue
                
                # Check tier requirements
                min_edge = tier_info.get("min_edge", 0.03)
                min_conf = tier_info.get("min_confidence", "LOW")
                
                conf_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
                if edge < min_edge:
                    continue
                if conf_rank.get(conf, 0) < conf_rank.get(min_conf, 0):
                    continue
                
                # Additional tier filters
                if tier == "SCALP":
                    odds = 1 / entry
                    if odds > tier_info.get("max_odds", 1.5):
                        continue  # Skip if odds too high
                
                if tier == "LONGSHOT":
                    if not (tier_info.get("min_price", 0) <= entry <= tier_info.get("max_price", 1)):
                        continue
                
                # Calculate bet sizing
                bet_mult = tier_info.get("bet_mult", 1.0)
                bet_size = BASE_BET * bet_mult
                payout = bet_size / entry
                profit = payout - bet_size
                rr = profit / bet_size
                
                results.append({
                    "tier": tier,
                    "source": m["src"],
                    "market_id": m["id"],
                    "question": m["q"],
                    "direction": d,
                    "entry_price": entry,
                    "fair_value": fv,
                    "edge": edge,
                    "confidence": conf,
                    "bet_size": bet_size,
                    "payout": payout,
                    "profit": profit,
                    "rr": rr,
                    "reason": reason,
                    "hours_left": m["hours"],
                    "volume": m["vol"],
                    "closes_at": m.get("close_time", "")
                })
            
            return results
    except Exception as e:
        print(f"Analysis error: {e}")
        return []

# ============================================================================
# SMART SCANNER
# ============================================================================

async def run_smart_scan(db, tiers_to_scan: list = None):
    """Run intelligent multi-tier scan."""
    if tiers_to_scan is None:
        tiers_to_scan = ["WHALE", "SCALP", "VALUE", "LONGSHOT"]
    
    print("📡 Fetching markets...")
    all_markets = await fetch_all_markets()
    print(f"  Found {len(all_markets)} markets")
    
    all_winners = []
    
    for tier in tiers_to_scan:
        tier_info = TIERS.get(tier, {})
        emoji = tier_info.get("emoji", "📊")
        
        # Filter candidates based on tier
        if tier == "SCALP":
            # Heavy favorites for scalping
            candidates = [m for m in all_markets 
                         if (m["price"] >= 0.70 or m["price"] <= 0.30) and m["vol"] > 10000]
        elif tier == "LONGSHOT":
            # Underdogs with volume
            candidates = [m for m in all_markets 
                         if 0.10 <= m["price"] <= 0.35 and m["vol"] > 20000]
        elif tier == "WHALE":
            # High volume, any price
            candidates = [m for m in all_markets if m["vol"] > 50000]
        else:  # VALUE
            # Contested markets
            candidates = [m for m in all_markets 
                         if 0.30 <= m["price"] <= 0.70 and m["vol"] > 20000]
        
        # Sort by volume (most liquid first)
        candidates.sort(key=lambda x: -x["vol"])
        
        if not candidates:
            print(f"  {emoji} {tier}: No candidates")
            continue
        
        print(f"  {emoji} Analyzing {tier} ({len(candidates[:12])} markets)...")
        
        winners = await analyze_for_tier(candidates[:12], tier)
        
        if winners:
            print(f"    ✅ Found {len(winners)} {tier} trades")
            all_winners.extend(winners)
        else:
            print(f"    ❌ No {tier} trades found")
        
        await asyncio.sleep(1)  # Rate limit
    
    # Save and alert
    new_count = 0
    for w in all_winners:
        if save_trade(db, w):
            new_count += 1
            await alert_new_trade(w)
            await asyncio.sleep(0.5)
    
    print(f"\n📊 Scan complete: {new_count} new trades")
    return all_winners

# ============================================================================
# RESOLUTION CHECKER
# ============================================================================

async def check_resolutions(db):
    """Check and resolve completed trades."""
    trades = get_open_trades(db)
    resolved = []
    
    for t in trades:
        # Check if past close time
        if t.get("closes_at"):
            try:
                close = datetime.fromisoformat(t["closes_at"].replace("Z", "+00:00"))
                if datetime.now(timezone.utc) < close - timedelta(minutes=30):
                    continue
            except:
                pass
        
        # Check market status
        result = None
        if t["source"] == "POLY":
            try:
                async with httpx.AsyncClient(timeout=15) as c:
                    resp = await c.get(f"https://gamma-api.polymarket.com/markets/{t['market_id']}")
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get("closed"):
                            prices = data.get("outcomePrices", "")
                            if prices:
                                p = [float(x.strip().strip('"')) for x in prices.strip("[]").split(",")]
                                if p[0] >= 0.99:
                                    result = "YES"
                                elif p[0] <= 0.01:
                                    result = "NO"
            except:
                pass
        else:  # KALSHI
            try:
                async with httpx.AsyncClient(timeout=15) as c:
                    resp = await c.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{t['market_id']}")
                    if resp.status_code == 200:
                        data = resp.json().get("market", {})
                        if data.get("status") == "finalized":
                            result = "YES" if data.get("result") == "yes" else "NO"
            except:
                pass
        
        if result:
            outcome = "WIN" if result == t["direction"] else "LOSS"
            pnl = t["potential_profit"] if outcome == "WIN" else -t["bet_size"]
            
            resolve_trade(db, t["market_id"], outcome, pnl)
            await alert_resolution(t, outcome, pnl)
            resolved.append({**t, "outcome": outcome, "pnl": pnl})
            print(f"  {'🎉' if outcome == 'WIN' else '❌'} {outcome}: {t['question'][:40]}... (${pnl:+.2f})")
    
    return resolved

# ============================================================================
# MAIN
# ============================================================================

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prediction Oracle V2")
    parser.add_argument("--scan", action="store_true", help="Run full scan")
    parser.add_argument("--whale", action="store_true", help="Scan for whale plays only")
    parser.add_argument("--scalp", action="store_true", help="Scan for scalps only")
    parser.add_argument("--longshot", action="store_true", help="Scan for longshots only")
    parser.add_argument("--resolve", action="store_true", help="Check resolutions")
    parser.add_argument("--status", action="store_true", help="Send portfolio summary")
    parser.add_argument("--daemon", action="store_true", help="Run as daemon")
    args = parser.parse_args()
    
    db = init_db()
    
    if args.status:
        await send_portfolio_summary(db)
        return
    
    if args.resolve:
        print("🔍 Checking resolutions...")
        await check_resolutions(db)
        return
    
    if args.whale:
        await run_smart_scan(db, ["WHALE"])
    elif args.scalp:
        await run_smart_scan(db, ["SCALP"])
    elif args.longshot:
        await run_smart_scan(db, ["LONGSHOT"])
    elif args.scan:
        await run_smart_scan(db)
        await send_portfolio_summary(db)
    elif args.daemon:
        print("🤖 Starting daemon...")
        await telegram("🤖 <b>Oracle V2 Daemon Started</b>")
        
        while True:
            try:
                print(f"\n⏰ {datetime.now().strftime('%H:%M')} - Checking resolutions...")
                await check_resolutions(db)
                
                # Scan every 30 min
                print("🔍 Running scan...")
                await run_smart_scan(db)
                
                # Summary every 4 hours
                stats = get_stats(db)
                print(f"📊 Open: {stats['open']} | P&L: ${stats['realized_pnl']:.2f}")
                
                await asyncio.sleep(1800)  # 30 min
                
            except KeyboardInterrupt:
                print("\n🛑 Stopping...")
                await telegram("🛑 Oracle V2 Stopped")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                await asyncio.sleep(300)
    else:
        # Default: full scan
        await run_smart_scan(db)
        await send_portfolio_summary(db)

if __name__ == "__main__":
    asyncio.run(main())
