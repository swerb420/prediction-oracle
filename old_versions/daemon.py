#!/usr/bin/env python3
"""
🤖 PREDICTION ORACLE DAEMON
===========================
Runs continuously, tracks positions, resolves trades, sends alerts.
Uses SQLite for persistence, Telegram for notifications.
"""
import asyncio
import sqlite3
import httpx
import json
import os
import re
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv

load_dotenv("/root/prediction_oracle/.env")

# Config
DB_PATH = "/root/prediction_oracle/oracle.db"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
XAI_KEY = os.getenv("XAI_API_KEY", "")

# ============================================================================
# DATABASE
# ============================================================================

def get_db():
    return sqlite3.connect(DB_PATH)

def get_open_trades(db) -> list:
    cursor = db.execute("""
        SELECT id, source, market_id, question, direction, entry_price, 
               fair_value, edge, confidence, bet_size, potential_payout,
               reason, hours_left, created_at, closes_at, category
        FROM trades WHERE outcome IS NULL
    """)
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in cursor.fetchall()]

def resolve_trade(db, trade_id: int, outcome: str, pnl: float):
    db.execute("""
        UPDATE trades SET outcome = ?, pnl = ?, resolved_at = ?
        WHERE id = ?
    """, (outcome, pnl, datetime.now(timezone.utc).isoformat(), trade_id))
    db.commit()

def get_stats(db) -> dict:
    cursor = db.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN outcome IS NOT NULL THEN pnl ELSE 0 END) as total_pnl,
            COUNT(CASE WHEN outcome IS NULL THEN 1 END) as open_count
        FROM trades
    """)
    r = cursor.fetchone()
    return {"total": r[0], "wins": r[1] or 0, "losses": r[2] or 0, 
            "pnl": r[3] or 0, "open": r[4] or 0,
            "win_rate": (r[1]/(r[1]+r[2])*100) if r[1] and r[2] else 0}

# ============================================================================
# TELEGRAM
# ============================================================================

async def telegram(msg: str):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "HTML"})
    except:
        pass

# ============================================================================
# MARKET RESOLUTION CHECKER
# ============================================================================

async def check_polymarket_resolution(market_id: str) -> dict | None:
    """Check if a Polymarket market has resolved."""
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            resp = await c.get(f"https://gamma-api.polymarket.com/markets/{market_id}")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("closed"):
                    # Get resolution
                    prices = data.get("outcomePrices", "")
                    if prices:
                        p = [float(x.strip().strip('"')) for x in prices.strip("[]").split(",")]
                        if p[0] >= 0.99:
                            return {"resolved": True, "outcome": "YES"}
                        elif p[0] <= 0.01:
                            return {"resolved": True, "outcome": "NO"}
                return {"resolved": False}
    except:
        pass
    return None

async def check_kalshi_resolution(market_id: str) -> dict | None:
    """Check if a Kalshi market has resolved."""
    try:
        async with httpx.AsyncClient(timeout=15) as c:
            resp = await c.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{market_id}")
            if resp.status_code == 200:
                data = resp.json().get("market", {})
                status = data.get("status", "")
                if status == "finalized":
                    result = data.get("result", "")
                    return {"resolved": True, "outcome": "YES" if result == "yes" else "NO"}
                return {"resolved": False}
    except:
        pass
    return None

async def resolve_trades(db):
    """Check and resolve any completed trades."""
    trades = get_open_trades(db)
    resolved = []
    
    for t in trades:
        # Check if market is past close time
        if t["closes_at"]:
            try:
                close = datetime.fromisoformat(t["closes_at"].replace("Z", "+00:00"))
                if datetime.now(timezone.utc) < close:
                    continue  # Not yet closed
            except:
                pass
        
        # Check resolution
        if t["source"] == "POLY":
            result = await check_polymarket_resolution(t["market_id"])
        else:
            result = await check_kalshi_resolution(t["market_id"])
        
        if result and result.get("resolved"):
            market_outcome = result["outcome"]
            trade_direction = t["direction"]
            
            # Did we win?
            if market_outcome == trade_direction:
                outcome = "WIN"
                pnl = t["potential_payout"] - t["bet_size"]
            else:
                outcome = "LOSS"
                pnl = -t["bet_size"]
            
            resolve_trade(db, t["id"], outcome, pnl)
            resolved.append({**t, "outcome": outcome, "pnl": pnl, "market_outcome": market_outcome})
    
    return resolved

# ============================================================================
# SCANNER (reuse from oracle_pro)
# ============================================================================

async def fetch_markets() -> list:
    """Fetch all markets."""
    markets = []
    now = datetime.now(timezone.utc)
    
    # Kalshi
    async with httpx.AsyncClient(timeout=30) as c:
        cursor = None
        for _ in range(3):
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
                        if 0.05 <= price <= 0.95 and 1 <= hours <= 720:
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
                params={"closed": "false", "active": "true", "limit": 100})
            if resp.status_code == 200:
                for m in resp.json():
                    try:
                        prices = m.get("outcomePrices", "")
                        if not prices:
                            continue
                        p = [float(x.strip().strip('"')) for x in prices.strip("[]").split(",") if x.strip()]
                        if len(p) < 2 or not (0.05 < p[0] < 0.95):
                            continue
                        end = m.get("endDate")
                        close = datetime.fromisoformat(end.replace("Z", "+00:00")) if end else now + timedelta(days=30)
                        hours = (close - now).total_seconds() / 3600
                        if 1 <= hours <= 720:
                            markets.append({
                                "src": "POLY", "id": m.get("id"), "q": m.get("question"),
                                "price": p[0], "hours": hours, "vol": float(m.get("volume24hr", 0) or 0),
                                "close_time": close.isoformat()
                            })
                    except:
                        pass
        except:
            pass
    
    return markets

async def analyze_batch(markets: list) -> list:
    """Analyze markets with Grok."""
    if not XAI_KEY or not markets:
        return []
    
    markets_text = "\n".join([
        f"{i}. \"{m['q']}\" - {m['price']:.0%} YES, ends {m['hours']:.0f}h"
        for i, m in enumerate(markets[:10], 1)
    ])
    
    prompt = f"""Analyze for mispricing. For each give true probability.

{markets_text}

JSON only: [{{"m":1,"fv":0.XX,"d":"YES/NO/SKIP","c":"HIGH/MED/LOW","r":"reason"}}]"""

    try:
        async with httpx.AsyncClient(timeout=60) as c:
            resp = await c.post("https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_KEY}"},
                json={"model": "grok-4-1-fast-reasoning", "messages": [{"role": "user", "content": prompt}], "max_tokens": 600})
            
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
                if d == "SKIP":
                    continue
                
                if d == "YES":
                    edge = fv - m["price"]
                    entry = m["price"]
                else:
                    edge = m["price"] - fv
                    entry = 1 - m["price"]
                
                if edge >= 0.03:
                    results.append({
                        "source": m["src"], "market_id": m["id"], "question": m["q"],
                        "direction": d, "entry_price": entry, "fair_value": fv,
                        "edge": edge, "confidence": a.get("c", "MED"),
                        "reason": a.get("r", ""), "hours_left": m["hours"],
                        "bet_size": 5.0, "potential_payout": 5/entry if entry > 0 else 0,
                        "closes_at": m.get("close_time", "")
                    })
            return results
    except:
        return []

def save_trade(db, t: dict):
    # Check if already exists
    cursor = db.execute("SELECT id FROM trades WHERE market_id = ? AND direction = ?", 
                        (t["market_id"], t["direction"]))
    if cursor.fetchone():
        return False  # Already have this trade
    
    db.execute("""
        INSERT INTO trades (source, market_id, question, direction, entry_price,
            fair_value, edge, confidence, bet_size, potential_payout, reason,
            hours_left, created_at, closes_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (t["source"], t["market_id"], t["question"], t["direction"], t["entry_price"],
          t["fair_value"], t["edge"], t["confidence"], t["bet_size"], t["potential_payout"],
          t["reason"], t["hours_left"], datetime.now(timezone.utc).isoformat(), t["closes_at"]))
    db.commit()
    return True

# ============================================================================
# MAIN DAEMON
# ============================================================================

async def run_daemon(scan_interval: int = 1800, resolve_interval: int = 300):
    """
    Main daemon loop.
    - Scans for new opportunities every scan_interval (default 30min)
    - Checks for resolutions every resolve_interval (default 5min)
    """
    print("🤖 PREDICTION ORACLE DAEMON")
    print(f"Scan interval: {scan_interval}s | Resolution check: {resolve_interval}s")
    print("=" * 60)
    
    await telegram("🤖 <b>Oracle Daemon Started</b>\n\nScanning for opportunities...")
    
    db = get_db()
    last_scan = 0
    
    while True:
        try:
            now = asyncio.get_event_loop().time()
            
            # Check resolutions frequently
            resolved = await resolve_trades(db)
            for r in resolved:
                emoji = "🎉" if r["outcome"] == "WIN" else "❌"
                msg = f"""{emoji} <b>TRADE RESOLVED: {r['outcome']}</b>

{r['direction']} {r['question'][:50]}...
Entry: {r['entry_price']:.0%} | Market: {r['market_outcome']}
P&L: <b>${r['pnl']:+.2f}</b>"""
                await telegram(msg)
                print(f"{emoji} Resolved: {r['direction']} {r['question'][:40]}... → {r['outcome']} (${r['pnl']:+.2f})")
            
            # Scan for new opportunities less frequently
            if now - last_scan >= scan_interval:
                print(f"\n🔍 Scanning... ({datetime.now().strftime('%H:%M')})")
                
                markets = await fetch_markets()
                
                # Filter for high-potential
                candidates = [m for m in markets if 0.25 <= m["price"] <= 0.75 or m["vol"] > 50000]
                candidates.sort(key=lambda x: x["hours"])
                
                if candidates:
                    winners = await analyze_batch(candidates[:10])
                    
                    new_trades = 0
                    for w in winners:
                        if save_trade(db, w):
                            new_trades += 1
                            emoji = "🔥" if w["edge"] >= 0.10 else "✅"
                            msg = f"""{emoji} <b>NEW TRADE</b>

<b>{w['direction']}</b> @ {w['entry_price']:.0%} → Fair: {w['fair_value']:.0%}
Edge: +{w['edge']:.1%} | {w['confidence']}

{w['question'][:60]}

💰 $5 → ${w['potential_payout']:.2f}
⏰ {w['hours_left']:.0f}h"""
                            await telegram(msg)
                            print(f"  {emoji} New: {w['direction']} {w['question'][:40]}...")
                    
                    if new_trades:
                        print(f"  Added {new_trades} new trades")
                
                # Send status update
                stats = get_stats(db)
                if stats["open"] > 0:
                    print(f"  Portfolio: {stats['open']} open | P&L: ${stats['pnl']:.2f} | Win: {stats['win_rate']:.0f}%")
                
                last_scan = now
            
            await asyncio.sleep(resolve_interval)
            
        except KeyboardInterrupt:
            print("\n🛑 Stopping daemon...")
            await telegram("🛑 Oracle Daemon Stopped")
            break
        except Exception as e:
            print(f"❌ Error: {e}")
            await asyncio.sleep(60)

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", type=int, default=1800, help="Scan interval (seconds)")
    parser.add_argument("--resolve", type=int, default=300, help="Resolution check interval")
    parser.add_argument("--status", action="store_true", help="Show status and exit")
    args = parser.parse_args()
    
    if args.status:
        db = get_db()
        stats = get_stats(db)
        trades = get_open_trades(db)
        
        print("📊 PORTFOLIO STATUS")
        print(f"Open: {stats['open']} | Wins: {stats['wins']} | Losses: {stats['losses']}")
        print(f"P&L: ${stats['pnl']:.2f} | Win Rate: {stats['win_rate']:.0f}%")
        
        if trades:
            print("\n📈 OPEN POSITIONS:")
            for t in trades:
                print(f"  {t['direction']} {t['question'][:45]}... (+{t['edge']:.1%})")
        return
    
    await run_daemon(scan_interval=args.scan, resolve_interval=args.resolve)

if __name__ == "__main__":
    asyncio.run(main())
