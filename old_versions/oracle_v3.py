#!/usr/bin/env python3
"""
🧠 PREDICTION ORACLE V3 - DUAL LLM CONSENSUS
=============================================
- Grok 4.1 + GPT for consensus
- Track LLM accuracy over time
- Adaptive prompts based on performance
- Conservative mode: Only bet when BOTH agree
- Aggressive mode: Hot longshots with high dual-confidence
- Full paper trading with resolution tracking
"""
import asyncio
import sqlite3
import httpx
import json
import os
import re
from datetime import datetime, timezone, timedelta
from typing import Optional
from dotenv import load_dotenv

load_dotenv("/root/prediction_oracle/.env")

# ============================================================================
# CONFIG
# ============================================================================

DB_PATH = "/root/prediction_oracle/oracle_v3.db"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
XAI_KEY = os.getenv("XAI_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

BASE_BET = 5.0
BANKROLL = 1000.0

# ============================================================================
# DATABASE WITH LLM TRACKING
# ============================================================================

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            tier TEXT,
            source TEXT,
            market_id TEXT,
            question TEXT,
            direction TEXT,
            entry_price REAL,
            bet_size REAL,
            potential_payout REAL,
            potential_profit REAL,
            risk_reward REAL,
            
            -- LLM Analysis
            grok_fair_value REAL,
            grok_direction TEXT,
            grok_confidence TEXT,
            grok_reason TEXT,
            
            gpt_fair_value REAL,
            gpt_direction TEXT,
            gpt_confidence TEXT,
            gpt_reason TEXT,
            
            consensus TEXT,  -- AGREE, DISAGREE, PARTIAL
            consensus_confidence TEXT,  -- ULTRA_HIGH, HIGH, MEDIUM, LOW
            combined_edge REAL,
            
            hours_left REAL,
            volume REAL,
            created_at TEXT,
            closes_at TEXT,
            resolved_at TEXT,
            outcome TEXT,
            actual_result TEXT,  -- What the market resolved to
            pnl REAL,
            
            -- For accuracy tracking
            grok_correct INTEGER,
            gpt_correct INTEGER
        );
        
        CREATE TABLE IF NOT EXISTS llm_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            llm TEXT,
            date TEXT,
            total_predictions INTEGER,
            correct_predictions INTEGER,
            accuracy REAL,
            avg_edge_when_correct REAL,
            avg_edge_when_wrong REAL,
            best_category TEXT,
            worst_category TEXT
        );
        
        CREATE TABLE IF NOT EXISTS prompts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            llm TEXT,
            version INTEGER,
            prompt_template TEXT,
            accuracy REAL,
            created_at TEXT,
            active INTEGER DEFAULT 1
        );
        
        CREATE INDEX IF NOT EXISTS idx_trades_consensus ON trades(consensus);
        CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id, direction);
    """)
    conn.commit()
    return conn

def get_llm_accuracy(db, llm: str) -> dict:
    """Get accuracy stats for an LLM."""
    cursor = db.execute(f"""
        SELECT 
            COUNT(*) as total,
            SUM({llm}_correct) as correct,
            AVG(CASE WHEN {llm}_correct = 1 THEN combined_edge END) as edge_correct,
            AVG(CASE WHEN {llm}_correct = 0 THEN combined_edge END) as edge_wrong
        FROM trades WHERE outcome IS NOT NULL AND {llm}_correct IS NOT NULL
    """)
    r = cursor.fetchone()
    return {
        "total": r[0] or 0,
        "correct": r[1] or 0,
        "accuracy": (r[1]/r[0]*100) if r[0] and r[1] else 0,
        "edge_when_correct": r[2] or 0,
        "edge_when_wrong": r[3] or 0
    }

def get_consensus_accuracy(db) -> dict:
    """Get accuracy when both LLMs agree."""
    cursor = db.execute("""
        SELECT 
            consensus,
            COUNT(*) as total,
            SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(pnl) as pnl
        FROM trades WHERE outcome IS NOT NULL
        GROUP BY consensus
    """)
    results = {}
    for row in cursor.fetchall():
        cons, total, wins, pnl = row
        results[cons] = {
            "total": total, "wins": wins or 0,
            "win_rate": (wins/total*100) if wins and total else 0,
            "pnl": pnl or 0
        }
    return results

# ============================================================================
# TELEGRAM
# ============================================================================

async def telegram(msg: str, urgent: bool = False):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        return
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            await c.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "HTML"})
    except:
        pass

async def alert_trade(t: dict):
    """Send rich trade alert with dual-LLM analysis."""
    tier_emoji = {"CONSERVATIVE": "🛡️", "AGGRESSIVE": "🔥", "LONGSHOT": "🎰", "WHALE": "🐋"}.get(t["tier"], "📊")
    consensus_emoji = {"AGREE": "✅✅", "PARTIAL": "⚠️", "DISAGREE": "❌"}.get(t["consensus"], "")
    
    hrs = t["hours_left"]
    time_str = f"{hrs:.0f}h" if hrs < 24 else f"{hrs/24:.1f}d"
    
    urgent = t["tier"] == "WHALE" or (t["consensus"] == "AGREE" and t["consensus_confidence"] == "ULTRA_HIGH")
    
    msg = f"""{tier_emoji} <b>{'🚨 HIGH CONVICTION ALERT!' if urgent else 'NEW TRADE'}</b>

<b>{t['direction']}</b> @ {t['entry_price']:.1%}
📋 {t['question'][:65]}

{'─'*30}
🧠 <b>DUAL LLM ANALYSIS</b>
{'─'*30}

<b>Grok 4.1:</b> {t['grok_direction']} @ {t['grok_fair_value']:.0%}
   Confidence: {t['grok_confidence']}
   "{t['grok_reason']}"

<b>GPT:</b> {t['gpt_direction']} @ {t['gpt_fair_value']:.0%}
   Confidence: {t['gpt_confidence']}
   "{t['gpt_reason']}"

{consensus_emoji} <b>Consensus:</b> {t['consensus']} ({t['consensus_confidence']})

{'─'*30}
💰 <b>BET DETAILS</b>
{'─'*30}
├ Bet: <b>${t['bet_size']:.2f}</b>
├ Win Payout: ${t['potential_payout']:.2f}
├ Profit if Win: +${t['potential_profit']:.2f}
├ Risk:Reward: 1:{t['risk_reward']:.1f}
├ Combined Edge: +{t['combined_edge']:.1%}
└ Closes: {time_str}

🏷️ Tier: {t['tier']} | Source: {t['source']}"""

    if urgent:
        msg += "\n\n🚨 <b>BOTH LLMs AGREE WITH HIGH CONFIDENCE!</b>"
    
    await telegram(msg, urgent=urgent)

async def send_accuracy_report(db):
    """Send LLM accuracy report."""
    grok = get_llm_accuracy(db, "grok")
    gpt = get_llm_accuracy(db, "gpt")
    consensus = get_consensus_accuracy(db)
    
    msg = f"""📊 <b>LLM ACCURACY REPORT</b>

🧠 <b>Grok 4.1</b>
├ Predictions: {grok['total']}
├ Accuracy: {grok['accuracy']:.1f}%
├ Avg Edge (correct): +{grok['edge_when_correct']:.1%}
└ Avg Edge (wrong): +{grok['edge_when_wrong']:.1%}

🤖 <b>GPT</b>
├ Predictions: {gpt['total']}
├ Accuracy: {gpt['accuracy']:.1f}%
├ Avg Edge (correct): +{gpt['edge_when_correct']:.1%}
└ Avg Edge (wrong): +{gpt['edge_when_wrong']:.1%}

{'─'*30}
<b>CONSENSUS PERFORMANCE</b>
"""
    
    for cons, stats in consensus.items():
        emoji = "✅" if cons == "AGREE" else "⚠️" if cons == "PARTIAL" else "❌"
        msg += f"\n{emoji} {cons}: {stats['win_rate']:.0f}% win ({stats['total']} trades) | ${stats['pnl']:+.2f}"
    
    await telegram(msg)

# ============================================================================
# DUAL LLM ANALYSIS
# ============================================================================

async def analyze_with_grok(markets: list, mode: str = "standard") -> list:
    """Analyze with Grok 4.1 Fast Reasoning."""
    if not XAI_KEY or not markets:
        return []
    
    # Mode-specific prompts
    if mode == "conservative":
        context = "Be VERY conservative. Only recommend bets you're 85%+ confident in. Safety over edge."
    elif mode == "aggressive":
        context = "Look for HIGH EDGE opportunities. Willing to take more risk for bigger rewards."
    elif mode == "longshot":
        context = "Find UNDERVALUED underdogs (10-40% prices). Look for mispriced long shots."
    else:
        context = "Standard analysis. Find mispriced markets with 5%+ edge."
    
    markets_text = "\n".join([
        f"{i}. \"{m['q']}\" - {m['price']:.0%} YES, {m['hours']:.0f}h, ${m['vol']:,.0f}"
        for i, m in enumerate(markets[:10], 1)
    ])
    
    prompt = f"""You are Grok, an expert prediction market analyst. {context}

Markets:
{markets_text}

For each market, determine TRUE probability based on:
- Current events and news
- Historical patterns
- Market efficiency

Reply JSON array ONLY:
[{{"m":1,"fv":0.XX,"d":"YES/NO/SKIP","c":"HIGH/MEDIUM/LOW","r":"reason 15 words max"}}]

Be accurate. Your track record is being measured."""

    try:
        async with httpx.AsyncClient(timeout=90) as c:
            resp = await c.post("https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_KEY}"},
                json={"model": "grok-4-1-fast-reasoning", 
                      "messages": [{"role": "user", "content": prompt}], 
                      "max_tokens": 800})
            
            if resp.status_code != 200:
                print(f"Grok error: {resp.status_code}")
                return []
            
            content = resp.json()["choices"][0]["message"]["content"]
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                return json.loads(match.group())
    except Exception as e:
        print(f"Grok error: {e}")
    return []

async def analyze_with_gpt(markets: list, mode: str = "standard") -> list:
    """Analyze with GPT."""
    if not OPENAI_KEY or not markets:
        return []
    
    if mode == "conservative":
        context = "Be conservative. Only recommend very safe bets with high confidence."
    elif mode == "aggressive":
        context = "Find high edge opportunities, willing to take calculated risks."
    elif mode == "longshot":
        context = "Find undervalued underdogs with big potential."
    else:
        context = "Standard value analysis."
    
    markets_text = "\n".join([
        f"{i}. \"{m['q']}\" - {m['price']:.0%} YES, {m['hours']:.0f}h"
        for i, m in enumerate(markets[:10], 1)
    ])
    
    prompt = f"""Prediction market analyst. {context}

Markets:
{markets_text}

For each give TRUE probability.
JSON array only: [{{"m":1,"fv":0.XX,"d":"YES/NO/SKIP","c":"HIGH/MEDIUM/LOW","r":"reason"}}]"""

    try:
        async with httpx.AsyncClient(timeout=60) as c:
            resp = await c.post("https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                json={"model": "gpt-4o-mini",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 600})
            
            if resp.status_code != 200:
                print(f"GPT error: {resp.status_code}")
                return []
            
            content = resp.json()["choices"][0]["message"]["content"]
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                return json.loads(match.group())
    except Exception as e:
        print(f"GPT error: {e}")
    return []

def combine_analyses(markets: list, grok_results: list, gpt_results: list, mode: str) -> list:
    """Combine both LLM analyses and determine consensus."""
    
    # Index results by market number
    grok_by_m = {a.get("m", 0): a for a in grok_results}
    gpt_by_m = {a.get("m", 0): a for a in gpt_results}
    
    combined = []
    
    for i, m in enumerate(markets[:10], 1):
        grok = grok_by_m.get(i, {})
        gpt = gpt_by_m.get(i, {})
        
        grok_dir = grok.get("d", "SKIP").upper()
        gpt_dir = gpt.get("d", "SKIP").upper()
        
        if grok_dir == "SKIP" and gpt_dir == "SKIP":
            continue
        
        grok_fv = float(grok.get("fv", m["price"]))
        gpt_fv = float(gpt.get("fv", m["price"]))
        grok_conf = grok.get("c", "LOW").upper()
        gpt_conf = gpt.get("c", "LOW").upper()
        
        # Determine consensus
        if grok_dir == gpt_dir and grok_dir != "SKIP":
            consensus = "AGREE"
            direction = grok_dir
            # Average fair values when agreeing
            fair_value = (grok_fv + gpt_fv) / 2
        elif grok_dir == "SKIP":
            consensus = "PARTIAL"
            direction = gpt_dir
            fair_value = gpt_fv
        elif gpt_dir == "SKIP":
            consensus = "PARTIAL"
            direction = grok_dir
            fair_value = grok_fv
        else:
            consensus = "DISAGREE"
            continue  # Skip when they disagree
        
        # Calculate entry and edge
        if direction == "YES":
            entry = m["price"]
            edge = fair_value - entry
        else:
            entry = 1 - m["price"]
            edge = (1 - fair_value) - (1 - m["price"])
            edge = m["price"] - fair_value  # Simplified
        
        if entry <= 0 or entry >= 1 or edge < 0.03:
            continue
        
        # Determine consensus confidence
        conf_rank = {"HIGH": 3, "MEDIUM": 2, "LOW": 1}
        grok_rank = conf_rank.get(grok_conf, 1)
        gpt_rank = conf_rank.get(gpt_conf, 1)
        
        if consensus == "AGREE" and grok_rank >= 3 and gpt_rank >= 3:
            cons_conf = "ULTRA_HIGH"
        elif consensus == "AGREE" and (grok_rank + gpt_rank) >= 5:
            cons_conf = "HIGH"
        elif consensus == "AGREE":
            cons_conf = "MEDIUM"
        else:
            cons_conf = "LOW"
        
        # Determine tier based on mode and consensus
        if mode == "conservative":
            if consensus != "AGREE" or cons_conf not in ["ULTRA_HIGH", "HIGH"]:
                continue  # Skip non-consensus for conservative
            tier = "CONSERVATIVE"
            bet_mult = 2.0 if cons_conf == "ULTRA_HIGH" else 1.5
        elif mode == "aggressive" or mode == "longshot":
            tier = "AGGRESSIVE" if edge >= 0.10 else "LONGSHOT"
            bet_mult = 1.5 if consensus == "AGREE" else 1.0
        else:
            if consensus == "AGREE" and cons_conf == "ULTRA_HIGH" and edge >= 0.08:
                tier = "WHALE"
                bet_mult = 4.0
            elif consensus == "AGREE":
                tier = "VALUE"
                bet_mult = 1.5
            else:
                tier = "SPECULATIVE"
                bet_mult = 1.0
        
        bet_size = BASE_BET * bet_mult
        payout = bet_size / entry if entry > 0 else 0
        profit = payout - bet_size
        rr = profit / bet_size if bet_size > 0 else 0
        
        combined.append({
            "tier": tier,
            "source": m["src"],
            "market_id": m["id"],
            "question": m["q"],
            "direction": direction,
            "entry_price": entry,
            "bet_size": bet_size,
            "potential_payout": payout,
            "potential_profit": profit,
            "risk_reward": rr,
            
            "grok_fair_value": grok_fv,
            "grok_direction": grok_dir,
            "grok_confidence": grok_conf,
            "grok_reason": grok.get("r", ""),
            
            "gpt_fair_value": gpt_fv,
            "gpt_direction": gpt_dir,
            "gpt_confidence": gpt_conf,
            "gpt_reason": gpt.get("r", ""),
            
            "consensus": consensus,
            "consensus_confidence": cons_conf,
            "combined_edge": edge,
            
            "hours_left": m["hours"],
            "volume": m["vol"],
            "closes_at": m.get("close_time", "")
        })
    
    return combined

# ============================================================================
# MARKET FETCHING
# ============================================================================

async def fetch_markets() -> list:
    """Fetch all markets."""
    markets = []
    now = datetime.now(timezone.utc)
    
    # Kalshi
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
                params={"closed": "false", "active": "true", "limit": 100})
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
# RESOLUTION & TRACKING
# ============================================================================

async def check_resolutions(db):
    """Check resolutions and update LLM accuracy."""
    cursor = db.execute("SELECT * FROM trades WHERE outcome IS NULL")
    cols = [d[0] for d in cursor.description]
    trades = [dict(zip(cols, row)) for row in cursor.fetchall()]
    
    resolved = []
    
    for t in trades:
        if t.get("closes_at"):
            try:
                close = datetime.fromisoformat(t["closes_at"].replace("Z", "+00:00"))
                if datetime.now(timezone.utc) < close - timedelta(minutes=30):
                    continue
            except:
                pass
        
        # Check market resolution
        actual_result = None
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
                                    actual_result = "YES"
                                elif p[0] <= 0.01:
                                    actual_result = "NO"
            except:
                pass
        else:  # KALSHI
            try:
                async with httpx.AsyncClient(timeout=15) as c:
                    resp = await c.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{t['market_id']}")
                    if resp.status_code == 200:
                        data = resp.json().get("market", {})
                        if data.get("status") == "finalized":
                            actual_result = "YES" if data.get("result") == "yes" else "NO"
            except:
                pass
        
        if actual_result:
            outcome = "WIN" if actual_result == t["direction"] else "LOSS"
            pnl = t["potential_profit"] if outcome == "WIN" else -t["bet_size"]
            
            # Track LLM accuracy
            grok_correct = 1 if t["grok_direction"] == actual_result else 0
            gpt_correct = 1 if t["gpt_direction"] == actual_result else 0
            
            db.execute("""
                UPDATE trades SET 
                    outcome = ?, pnl = ?, resolved_at = ?, actual_result = ?,
                    grok_correct = ?, gpt_correct = ?
                WHERE id = ?
            """, (outcome, pnl, datetime.now(timezone.utc).isoformat(), actual_result,
                  grok_correct, gpt_correct, t["id"]))
            db.commit()
            
            # Alert
            emoji = "🎉" if outcome == "WIN" else "❌"
            grok_emoji = "✅" if grok_correct else "❌"
            gpt_emoji = "✅" if gpt_correct else "❌"
            
            msg = f"""{emoji} <b>RESOLVED: {outcome}</b>

{t['question'][:50]}...

Result: <b>{actual_result}</b>
Our bet: {t['direction']}
P&L: <b>${pnl:+.2f}</b>

LLM Accuracy:
├ Grok: {grok_emoji} ({t['grok_direction']})
└ GPT: {gpt_emoji} ({t['gpt_direction']})"""
            
            await telegram(msg)
            resolved.append({**t, "outcome": outcome, "pnl": pnl})
            print(f"  {emoji} {outcome}: {t['question'][:40]}... (${pnl:+.2f})")
    
    return resolved

def save_trade(db, t: dict) -> bool:
    """Save trade to database."""
    try:
        db.execute("""
            INSERT INTO trades (
                tier, source, market_id, question, direction, entry_price,
                bet_size, potential_payout, potential_profit, risk_reward,
                grok_fair_value, grok_direction, grok_confidence, grok_reason,
                gpt_fair_value, gpt_direction, gpt_confidence, gpt_reason,
                consensus, consensus_confidence, combined_edge,
                hours_left, volume, created_at, closes_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            t["tier"], t["source"], t["market_id"], t["question"], t["direction"],
            t["entry_price"], t["bet_size"], t["potential_payout"], t["potential_profit"],
            t["risk_reward"], t["grok_fair_value"], t["grok_direction"], t["grok_confidence"],
            t["grok_reason"], t["gpt_fair_value"], t["gpt_direction"], t["gpt_confidence"],
            t["gpt_reason"], t["consensus"], t["consensus_confidence"], t["combined_edge"],
            t["hours_left"], t["volume"], datetime.now(timezone.utc).isoformat(), t["closes_at"]
        ))
        db.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# ============================================================================
# SMART SCANNER
# ============================================================================

async def run_dual_scan(db, modes: list = None):
    """Run dual-LLM scan."""
    if modes is None:
        modes = ["conservative", "standard", "longshot"]
    
    print("📡 Fetching markets...")
    all_markets = await fetch_markets()
    print(f"  Found {len(all_markets)} markets")
    
    all_trades = []
    
    for mode in modes:
        print(f"\n🔍 Scanning mode: {mode.upper()}")
        
        # Filter candidates by mode
        if mode == "conservative":
            candidates = [m for m in all_markets 
                         if (m["price"] >= 0.65 or m["price"] <= 0.35) and m["vol"] > 20000]
        elif mode == "longshot":
            candidates = [m for m in all_markets 
                         if 0.10 <= m["price"] <= 0.40 and m["vol"] > 15000]
        else:  # standard
            candidates = [m for m in all_markets if m["vol"] > 30000]
        
        candidates.sort(key=lambda x: -x["vol"])
        candidates = candidates[:10]
        
        if not candidates:
            print(f"  No candidates for {mode}")
            continue
        
        # Dual LLM analysis
        print(f"  🧠 Analyzing with Grok...")
        grok_results = await analyze_with_grok(candidates, mode)
        
        print(f"  🤖 Analyzing with GPT...")
        gpt_results = await analyze_with_gpt(candidates, mode)
        
        # Combine
        trades = combine_analyses(candidates, grok_results, gpt_results, mode)
        
        if trades:
            print(f"  ✅ Found {len(trades)} trades")
            for t in trades:
                emoji = "✅✅" if t["consensus"] == "AGREE" else "⚠️"
                print(f"    {emoji} {t['direction']} {t['question'][:40]}... (+{t['combined_edge']:.1%})")
            all_trades.extend(trades)
        else:
            print(f"  ❌ No trades found")
        
        await asyncio.sleep(1)
    
    # Save and alert
    new_count = 0
    for t in all_trades:
        if save_trade(db, t):
            new_count += 1
            await alert_trade(t)
            await asyncio.sleep(0.5)
    
    print(f"\n📊 Saved {new_count} new trades")
    return all_trades

async def show_status(db):
    """Show portfolio status."""
    cursor = db.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN outcome = 'WIN' THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN outcome = 'LOSS' THEN 1 ELSE 0 END) as losses,
            SUM(CASE WHEN outcome IS NOT NULL THEN pnl ELSE 0 END) as pnl,
            SUM(CASE WHEN outcome IS NULL THEN bet_size ELSE 0 END) as at_risk,
            COUNT(CASE WHEN outcome IS NULL THEN 1 END) as open_count
        FROM trades
    """)
    r = cursor.fetchone()
    
    print("\n📊 PORTFOLIO STATUS")
    print(f"Total: {r[0]} | Wins: {r[1] or 0} | Losses: {r[2] or 0}")
    print(f"P&L: ${r[3] or 0:.2f} | At Risk: ${r[4] or 0:.2f} | Open: {r[5] or 0}")
    
    # Consensus stats
    print("\n📈 CONSENSUS PERFORMANCE:")
    cons_stats = get_consensus_accuracy(db)
    for cons, stats in cons_stats.items():
        print(f"  {cons}: {stats['win_rate']:.0f}% ({stats['total']} trades) ${stats['pnl']:+.2f}")
    
    # LLM accuracy
    print("\n🧠 LLM ACCURACY:")
    grok = get_llm_accuracy(db, "grok")
    gpt = get_llm_accuracy(db, "gpt")
    print(f"  Grok: {grok['accuracy']:.1f}% ({grok['total']} predictions)")
    print(f"  GPT: {gpt['accuracy']:.1f}% ({gpt['total']} predictions)")

# ============================================================================
# MAIN
# ============================================================================

async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", action="store_true", help="Full dual-LLM scan")
    parser.add_argument("--conservative", action="store_true", help="Conservative only")
    parser.add_argument("--aggressive", action="store_true", help="Aggressive only")
    parser.add_argument("--longshot", action="store_true", help="Longshots only")
    parser.add_argument("--resolve", action="store_true", help="Check resolutions")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--accuracy", action="store_true", help="Send accuracy report")
    parser.add_argument("--daemon", action="store_true", help="Run daemon")
    args = parser.parse_args()
    
    db = init_db()
    
    if args.status:
        await show_status(db)
    elif args.accuracy:
        await send_accuracy_report(db)
    elif args.resolve:
        await check_resolutions(db)
    elif args.conservative:
        await run_dual_scan(db, ["conservative"])
    elif args.aggressive:
        await run_dual_scan(db, ["standard", "aggressive"])
    elif args.longshot:
        await run_dual_scan(db, ["longshot"])
    elif args.daemon:
        print("🤖 Starting V3 Daemon...")
        await telegram("🤖 <b>Oracle V3 Daemon Started</b>\nDual-LLM Consensus System Active")
        
        while True:
            try:
                print(f"\n⏰ {datetime.now().strftime('%H:%M')}")
                await check_resolutions(db)
                await run_dual_scan(db)
                await show_status(db)
                await asyncio.sleep(1800)  # 30 min
            except KeyboardInterrupt:
                await telegram("🛑 Oracle V3 Stopped")
                break
            except Exception as e:
                print(f"Error: {e}")
                await asyncio.sleep(300)
    else:
        await run_dual_scan(db)
        await show_status(db)

if __name__ == "__main__":
    asyncio.run(main())
