#!/usr/bin/env python3
"""
🧠 PREDICTION ORACLE V4 - FULL LOGGING + DEEP ANALYSIS
========================================================
- Comprehensive structured logging
- LLM request/response logging for debugging
- Performance metrics and timing
- Rotating log files
- Full audit trail for analysis
"""
import asyncio
import sqlite3
import httpx
import json
import os
import re
import sys
import logging
import traceback
from datetime import datetime, timezone, timedelta
from typing import Optional, Any
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv

load_dotenv("/root/prediction_oracle/.env")

# ============================================================================
# LOGGING SETUP
# ============================================================================

LOG_DIR = Path("/root/prediction_oracle/logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logging():
    """Setup comprehensive logging with rotation."""
    
    # Main logger
    logger = logging.getLogger("oracle")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    # Format with microseconds for timing analysis
    detailed_format = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-12s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    # Console handler - INFO level
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(simple_format)
    logger.addHandler(console)
    
    # Main log file - DEBUG level (10MB, keep 5)
    main_file = RotatingFileHandler(
        LOG_DIR / "oracle.log", maxBytes=10*1024*1024, backupCount=5
    )
    main_file.setLevel(logging.DEBUG)
    main_file.setFormatter(detailed_format)
    logger.addHandler(main_file)
    
    # Error log - WARNING and above
    error_file = RotatingFileHandler(
        LOG_DIR / "errors.log", maxBytes=5*1024*1024, backupCount=3
    )
    error_file.setLevel(logging.WARNING)
    error_file.setFormatter(detailed_format)
    logger.addHandler(error_file)
    
    # LLM-specific logger for deep analysis
    llm_logger = logging.getLogger("oracle.llm")
    llm_file = RotatingFileHandler(
        LOG_DIR / "llm_analysis.log", maxBytes=20*1024*1024, backupCount=10
    )
    llm_file.setLevel(logging.DEBUG)
    llm_file.setFormatter(detailed_format)
    llm_logger.addHandler(llm_file)
    
    # Trade logger
    trade_logger = logging.getLogger("oracle.trades")
    trade_file = RotatingFileHandler(
        LOG_DIR / "trades.log", maxBytes=10*1024*1024, backupCount=5
    )
    trade_file.setLevel(logging.DEBUG)
    trade_file.setFormatter(detailed_format)
    trade_logger.addHandler(trade_file)
    
    # Market data logger
    market_logger = logging.getLogger("oracle.markets")
    market_file = RotatingFileHandler(
        LOG_DIR / "markets.log", maxBytes=10*1024*1024, backupCount=5
    )
    market_file.setLevel(logging.DEBUG)
    market_file.setFormatter(detailed_format)
    market_logger.addHandler(market_file)
    
    return logger

log = setup_logging()
llm_log = logging.getLogger("oracle.llm")
trade_log = logging.getLogger("oracle.trades")
market_log = logging.getLogger("oracle.markets")

# ============================================================================
# CONFIG
# ============================================================================

DB_PATH = "/root/prediction_oracle/oracle_v4.db"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
XAI_KEY = os.getenv("XAI_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

BASE_BET = 5.0
BANKROLL = 1000.0

log.info(f"Oracle V4 initialized | DB: {DB_PATH}")
log.info(f"Telegram: {'Enabled' if TELEGRAM_TOKEN else 'Disabled'}")
log.info(f"Grok: {'Enabled' if XAI_KEY else 'Disabled'} | GPT: {'Enabled' if OPENAI_KEY else 'Disabled'}")

# ============================================================================
# DATABASE WITH LLM TRACKING
# ============================================================================

def init_db():
    log.debug("Initializing database...")
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
            grok_response_time REAL,
            
            gpt_fair_value REAL,
            gpt_direction TEXT,
            gpt_confidence TEXT,
            gpt_reason TEXT,
            gpt_response_time REAL,
            
            consensus TEXT,
            consensus_confidence TEXT,
            combined_edge REAL,
            
            hours_left REAL,
            volume REAL,
            created_at TEXT,
            closes_at TEXT,
            resolved_at TEXT,
            outcome TEXT,
            actual_result TEXT,
            pnl REAL,
            
            grok_correct INTEGER,
            gpt_correct INTEGER,
            
            -- Metadata
            scan_mode TEXT,
            scan_batch_id TEXT
        );
        
        CREATE TABLE IF NOT EXISTS llm_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            llm TEXT,
            mode TEXT,
            markets_count INTEGER,
            prompt_tokens INTEGER,
            response_tokens INTEGER,
            response_time_ms REAL,
            status TEXT,
            error_message TEXT,
            raw_prompt TEXT,
            raw_response TEXT,
            parsed_results TEXT,
            batch_id TEXT
        );
        
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT,
            timestamp TEXT,
            modes TEXT,
            markets_fetched INTEGER,
            kalshi_count INTEGER,
            poly_count INTEGER,
            trades_found INTEGER,
            trades_saved INTEGER,
            grok_calls INTEGER,
            gpt_calls INTEGER,
            total_time_sec REAL,
            errors TEXT
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
        
        CREATE INDEX IF NOT EXISTS idx_trades_consensus ON trades(consensus);
        CREATE INDEX IF NOT EXISTS idx_trades_outcome ON trades(outcome);
        CREATE INDEX IF NOT EXISTS idx_trades_batch ON trades(scan_batch_id);
        CREATE UNIQUE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id, direction);
        CREATE INDEX IF NOT EXISTS idx_llm_requests_batch ON llm_requests(batch_id);
    """)
    conn.commit()
    log.info("Database initialized successfully")
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

def log_llm_request(db, batch_id: str, llm: str, mode: str, markets_count: int,
                    prompt: str, response: str, parsed: list, status: str,
                    response_time_ms: float, error_msg: str = None):
    """Log LLM request for deep analysis."""
    try:
        db.execute("""
            INSERT INTO llm_requests (
                timestamp, llm, mode, markets_count, prompt_tokens, response_tokens,
                response_time_ms, status, error_message, raw_prompt, raw_response,
                parsed_results, batch_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now(timezone.utc).isoformat(),
            llm, mode, markets_count,
            len(prompt.split()), len(response.split()) if response else 0,
            response_time_ms, status, error_msg,
            prompt[:5000],  # Truncate for storage
            response[:10000] if response else None,
            json.dumps(parsed) if parsed else None,
            batch_id
        ))
        db.commit()
    except Exception as e:
        log.error(f"Failed to log LLM request: {e}")

def generate_batch_id() -> str:
    """Generate unique batch ID for tracking."""
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

# ============================================================================
# TELEGRAM
# ============================================================================

async def telegram(msg: str, urgent: bool = False):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT:
        log.debug("Telegram disabled, skipping message")
        return
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            resp = await c.post(f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={"chat_id": TELEGRAM_CHAT, "text": msg, "parse_mode": "HTML"})
            if resp.status_code == 200:
                log.debug(f"Telegram sent: {msg[:50]}...")
            else:
                log.warning(f"Telegram failed: {resp.status_code}")
    except Exception as e:
        log.error(f"Telegram error: {e}")

async def alert_trade(t: dict):
    """Send rich trade alert with dual-LLM analysis."""
    tier_emoji = {"CONSERVATIVE": "🛡️", "AGGRESSIVE": "🔥", "LONGSHOT": "🎰", "WHALE": "🐋"}.get(t["tier"], "📊")
    consensus_emoji = {"AGREE": "✅✅", "PARTIAL": "⚠️", "DISAGREE": "❌"}.get(t["consensus"], "")
    
    hrs = t["hours_left"]
    time_str = f"{hrs:.0f}h" if hrs < 24 else f"{hrs/24:.1f}d"
    
    urgent = t["tier"] == "WHALE" or (t["consensus"] == "AGREE" and t["consensus_confidence"] == "ULTRA_HIGH")
    
    msg = f"""{tier_emoji} <b>{'🚨 HIGH CONVICTION!' if urgent else 'NEW TRADE'}</b>

<b>{t['direction']}</b> @ {t['entry_price']:.1%}
📋 {t['question'][:65]}

{'─'*30}
🧠 <b>DUAL LLM ANALYSIS</b>

<b>Grok:</b> {t['grok_direction']} @ {t['grok_fair_value']:.0%} ({t['grok_confidence']})
  "{t['grok_reason']}"

<b>GPT:</b> {t['gpt_direction']} @ {t['gpt_fair_value']:.0%} ({t['gpt_confidence']})
  "{t['gpt_reason']}"

{consensus_emoji} <b>{t['consensus']}</b> ({t['consensus_confidence']})

{'─'*30}
💰 Bet: <b>${t['bet_size']:.2f}</b> → Win: ${t['potential_payout']:.2f}
📈 Edge: +{t['combined_edge']:.1%} | R:R: 1:{t['risk_reward']:.1f}
⏰ Closes: {time_str} | 🏷️ {t['tier']}"""
    
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
├ Edge (correct): +{grok['edge_when_correct']:.1%}
└ Edge (wrong): +{grok['edge_when_wrong']:.1%}

🤖 <b>GPT</b>
├ Predictions: {gpt['total']}
├ Accuracy: {gpt['accuracy']:.1f}%
├ Edge (correct): +{gpt['edge_when_correct']:.1%}
└ Edge (wrong): +{gpt['edge_when_wrong']:.1%}

<b>CONSENSUS</b>"""
    
    for cons, stats in consensus.items():
        emoji = "✅" if cons == "AGREE" else "⚠️"
        msg += f"\n{emoji} {cons}: {stats['win_rate']:.0f}% ({stats['total']}) ${stats['pnl']:+.2f}"
    
    await telegram(msg)

# ============================================================================
# DUAL LLM ANALYSIS WITH FULL LOGGING
# ============================================================================

async def analyze_with_grok(db, markets: list, mode: str, batch_id: str) -> tuple[list, float]:
    """Analyze with Grok 4.1 - returns (results, response_time_ms)."""
    if not XAI_KEY or not markets:
        llm_log.warning("Grok analysis skipped - no API key or empty markets")
        return [], 0
    
    # Mode-specific prompts
    contexts = {
        "conservative": "Be VERY conservative. Only recommend bets you're 85%+ confident in.",
        "aggressive": "Look for HIGH EDGE opportunities. Willing to take more risk.",
        "longshot": "Find UNDERVALUED underdogs (10-40% prices). Look for mispriced long shots.",
        "standard": "Standard analysis. Find mispriced markets with 5%+ edge."
    }
    context = contexts.get(mode, contexts["standard"])
    
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

    llm_log.info(f"GROK REQUEST | mode={mode} | markets={len(markets)} | batch={batch_id}")
    llm_log.debug(f"GROK PROMPT:\n{prompt}")
    
    start = datetime.now()
    
    try:
        async with httpx.AsyncClient(timeout=90) as c:
            resp = await c.post("https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_KEY}"},
                json={"model": "grok-4-1-fast-reasoning", 
                      "messages": [{"role": "user", "content": prompt}], 
                      "max_tokens": 800})
            
            elapsed_ms = (datetime.now() - start).total_seconds() * 1000
            
            if resp.status_code != 200:
                llm_log.error(f"GROK ERROR | status={resp.status_code} | response={resp.text[:500]}")
                log_llm_request(db, batch_id, "grok", mode, len(markets), prompt, 
                               resp.text, [], "ERROR", elapsed_ms, f"HTTP {resp.status_code}")
                return [], elapsed_ms
            
            content = resp.json()["choices"][0]["message"]["content"]
            llm_log.debug(f"GROK RAW RESPONSE:\n{content}")
            
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                results = json.loads(match.group())
                llm_log.info(f"GROK SUCCESS | results={len(results)} | time={elapsed_ms:.0f}ms")
                
                # Log each prediction
                for r in results:
                    llm_log.debug(f"GROK PREDICTION | m={r.get('m')} | fv={r.get('fv')} | "
                                 f"d={r.get('d')} | c={r.get('c')} | r={r.get('r')}")
                
                log_llm_request(db, batch_id, "grok", mode, len(markets), prompt, 
                               content, results, "SUCCESS", elapsed_ms)
                return results, elapsed_ms
            else:
                llm_log.warning(f"GROK PARSE FAIL | no JSON found in response")
                log_llm_request(db, batch_id, "grok", mode, len(markets), prompt, 
                               content, [], "PARSE_ERROR", elapsed_ms, "No JSON array found")
                return [], elapsed_ms
                
    except Exception as e:
        elapsed_ms = (datetime.now() - start).total_seconds() * 1000
        llm_log.error(f"GROK EXCEPTION | error={str(e)}\n{traceback.format_exc()}")
        log_llm_request(db, batch_id, "grok", mode, len(markets), prompt, 
                       "", [], "EXCEPTION", elapsed_ms, str(e))
        return [], elapsed_ms

async def analyze_with_gpt(db, markets: list, mode: str, batch_id: str) -> tuple[list, float]:
    """Analyze with GPT - returns (results, response_time_ms)."""
    if not OPENAI_KEY or not markets:
        llm_log.warning("GPT analysis skipped - no API key or empty markets")
        return [], 0
    
    contexts = {
        "conservative": "Be conservative. Only recommend very safe bets with high confidence.",
        "aggressive": "Find high edge opportunities, willing to take calculated risks.",
        "longshot": "Find undervalued underdogs with big potential.",
        "standard": "Standard value analysis."
    }
    context = contexts.get(mode, contexts["standard"])
    
    markets_text = "\n".join([
        f"{i}. \"{m['q']}\" - {m['price']:.0%} YES, {m['hours']:.0f}h"
        for i, m in enumerate(markets[:10], 1)
    ])
    
    prompt = f"""Prediction market analyst. {context}

Markets:
{markets_text}

For each give TRUE probability.
JSON array only: [{{"m":1,"fv":0.XX,"d":"YES/NO/SKIP","c":"HIGH/MEDIUM/LOW","r":"reason"}}]"""

    llm_log.info(f"GPT REQUEST | mode={mode} | markets={len(markets)} | batch={batch_id}")
    llm_log.debug(f"GPT PROMPT:\n{prompt}")
    
    start = datetime.now()
    
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            resp = await c.post("https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                json={"model": "gpt-4o-mini",
                      "messages": [{"role": "user", "content": prompt}],
                      "max_tokens": 600})
            
            elapsed_ms = (datetime.now() - start).total_seconds() * 1000
            
            if resp.status_code != 200:
                llm_log.error(f"GPT ERROR | status={resp.status_code} | response={resp.text[:500]}")
                log_llm_request(db, batch_id, "gpt", mode, len(markets), prompt, 
                               resp.text, [], "ERROR", elapsed_ms, f"HTTP {resp.status_code}")
                return [], elapsed_ms
            
            content = resp.json()["choices"][0]["message"]["content"]
            llm_log.debug(f"GPT RAW RESPONSE:\n{content}")
            
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                results = json.loads(match.group())
                llm_log.info(f"GPT SUCCESS | results={len(results)} | time={elapsed_ms:.0f}ms")
                
                for r in results:
                    llm_log.debug(f"GPT PREDICTION | m={r.get('m')} | fv={r.get('fv')} | "
                                 f"d={r.get('d')} | c={r.get('c')} | r={r.get('r')}")
                
                log_llm_request(db, batch_id, "gpt", mode, len(markets), prompt, 
                               content, results, "SUCCESS", elapsed_ms)
                return results, elapsed_ms
            else:
                llm_log.warning(f"GPT PARSE FAIL | no JSON found")
                log_llm_request(db, batch_id, "gpt", mode, len(markets), prompt, 
                               content, [], "PARSE_ERROR", elapsed_ms, "No JSON array found")
                return [], elapsed_ms
                
    except Exception as e:
        elapsed_ms = (datetime.now() - start).total_seconds() * 1000
        llm_log.error(f"GPT EXCEPTION | error={str(e)}\n{traceback.format_exc()}")
        log_llm_request(db, batch_id, "gpt", mode, len(markets), prompt, 
                       "", [], "EXCEPTION", elapsed_ms, str(e))
        return [], elapsed_ms

def combine_analyses(markets: list, grok_results: list, gpt_results: list, 
                     mode: str, grok_time: float, gpt_time: float, batch_id: str) -> list:
    """Combine both LLM analyses and determine consensus."""
    
    grok_by_m = {a.get("m", 0): a for a in grok_results}
    gpt_by_m = {a.get("m", 0): a for a in gpt_results}
    
    combined = []
    
    for i, m in enumerate(markets[:10], 1):
        grok = grok_by_m.get(i, {})
        gpt = gpt_by_m.get(i, {})
        
        grok_dir = grok.get("d", "SKIP").upper()
        gpt_dir = gpt.get("d", "SKIP").upper()
        
        llm_log.debug(f"COMBINE | market={i} | grok={grok_dir} | gpt={gpt_dir}")
        
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
            llm_log.info(f"DISAGREE | market={i} | grok={grok_dir} | gpt={gpt_dir} | skipping")
            continue
        
        # Calculate entry and edge
        if direction == "YES":
            entry = m["price"]
            edge = fair_value - entry
        else:
            entry = 1 - m["price"]
            edge = m["price"] - fair_value
        
        if entry <= 0 or entry >= 1 or edge < 0.03:
            llm_log.debug(f"SKIP | market={i} | invalid entry/edge | entry={entry:.2f} | edge={edge:.2%}")
            continue
        
        # Consensus confidence
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
        
        # Tier logic
        if mode == "conservative":
            if consensus != "AGREE" or cons_conf not in ["ULTRA_HIGH", "HIGH"]:
                continue
            tier = "CONSERVATIVE"
            bet_mult = 2.0 if cons_conf == "ULTRA_HIGH" else 1.5
        elif mode in ["aggressive", "longshot"]:
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
        
        trade = {
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
            "grok_response_time": grok_time,
            
            "gpt_fair_value": gpt_fv,
            "gpt_direction": gpt_dir,
            "gpt_confidence": gpt_conf,
            "gpt_reason": gpt.get("r", ""),
            "gpt_response_time": gpt_time,
            
            "consensus": consensus,
            "consensus_confidence": cons_conf,
            "combined_edge": edge,
            
            "hours_left": m["hours"],
            "volume": m["vol"],
            "closes_at": m.get("close_time", ""),
            "scan_mode": mode,
            "scan_batch_id": batch_id
        }
        
        trade_log.info(f"TRADE FOUND | {tier} | {consensus} | {direction} | "
                      f"edge={edge:.1%} | {m['q'][:50]}")
        combined.append(trade)
    
    return combined

# ============================================================================
# MARKET FETCHING
# ============================================================================

async def fetch_markets() -> tuple[list, dict]:
    """Fetch all markets - returns (markets, stats)."""
    markets = []
    stats = {"kalshi": 0, "poly": 0, "errors": []}
    now = datetime.now(timezone.utc)
    
    market_log.info("Starting market fetch...")
    
    # Kalshi
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            cursor = None
            for page in range(5):
                params = {"limit": 100, "status": "open"}
                if cursor:
                    params["cursor"] = cursor
                
                resp = await c.get("https://api.elections.kalshi.com/trade-api/v2/markets", params=params)
                
                if resp.status_code != 200:
                    market_log.warning(f"Kalshi page {page} error: {resp.status_code}")
                    stats["errors"].append(f"Kalshi HTTP {resp.status_code}")
                    break
                    
                data = resp.json()
                page_count = 0
                
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
                            page_count += 1
                    except Exception as e:
                        market_log.debug(f"Kalshi market parse error: {e}")
                
                stats["kalshi"] += page_count
                market_log.debug(f"Kalshi page {page}: {page_count} markets")
                
                cursor = data.get("cursor")
                if not cursor:
                    break
    except Exception as e:
        market_log.error(f"Kalshi fetch error: {e}")
        stats["errors"].append(f"Kalshi: {str(e)}")
    
    # Polymarket
    try:
        async with httpx.AsyncClient(timeout=30) as c:
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
                            stats["poly"] += 1
                    except Exception as e:
                        market_log.debug(f"Poly market parse error: {e}")
            else:
                market_log.warning(f"Polymarket error: {resp.status_code}")
                stats["errors"].append(f"Poly HTTP {resp.status_code}")
    except Exception as e:
        market_log.error(f"Polymarket fetch error: {e}")
        stats["errors"].append(f"Poly: {str(e)}")
    
    market_log.info(f"Fetch complete | Kalshi={stats['kalshi']} | Poly={stats['poly']} | Total={len(markets)}")
    return markets, stats

# ============================================================================
# RESOLUTION & TRACKING
# ============================================================================

async def check_resolutions(db):
    """Check resolutions and update LLM accuracy."""
    log.info("Checking resolutions...")
    
    cursor = db.execute("SELECT * FROM trades WHERE outcome IS NULL")
    cols = [d[0] for d in cursor.description]
    trades = [dict(zip(cols, row)) for row in cursor.fetchall()]
    
    log.info(f"Checking {len(trades)} open trades")
    resolved = []
    
    for t in trades:
        if t.get("closes_at"):
            try:
                close = datetime.fromisoformat(t["closes_at"].replace("Z", "+00:00"))
                if datetime.now(timezone.utc) < close - timedelta(minutes=30):
                    continue
            except:
                pass
        
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
            except Exception as e:
                log.debug(f"Poly resolution check error: {e}")
        else:
            try:
                async with httpx.AsyncClient(timeout=15) as c:
                    resp = await c.get(f"https://api.elections.kalshi.com/trade-api/v2/markets/{t['market_id']}")
                    if resp.status_code == 200:
                        data = resp.json().get("market", {})
                        if data.get("status") == "finalized":
                            actual_result = "YES" if data.get("result") == "yes" else "NO"
            except Exception as e:
                log.debug(f"Kalshi resolution check error: {e}")
        
        if actual_result:
            outcome = "WIN" if actual_result == t["direction"] else "LOSS"
            pnl = t["potential_profit"] if outcome == "WIN" else -t["bet_size"]
            
            grok_correct = 1 if t["grok_direction"] == actual_result else 0
            gpt_correct = 1 if t["gpt_direction"] == actual_result else 0
            
            trade_log.info(f"RESOLVED | {outcome} | ${pnl:+.2f} | "
                          f"grok={'✓' if grok_correct else '✗'} | gpt={'✓' if gpt_correct else '✗'} | "
                          f"{t['question'][:40]}")
            
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
Result: <b>{actual_result}</b> | P&L: <b>${pnl:+.2f}</b>

LLM: Grok {grok_emoji} | GPT {gpt_emoji}"""
            
            await telegram(msg)
            resolved.append({**t, "outcome": outcome, "pnl": pnl})
    
    log.info(f"Resolved {len(resolved)} trades")
    return resolved

def save_trade(db, t: dict) -> bool:
    """Save trade to database."""
    try:
        db.execute("""
            INSERT INTO trades (
                tier, source, market_id, question, direction, entry_price,
                bet_size, potential_payout, potential_profit, risk_reward,
                grok_fair_value, grok_direction, grok_confidence, grok_reason, grok_response_time,
                gpt_fair_value, gpt_direction, gpt_confidence, gpt_reason, gpt_response_time,
                consensus, consensus_confidence, combined_edge,
                hours_left, volume, created_at, closes_at, scan_mode, scan_batch_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            t["tier"], t["source"], t["market_id"], t["question"], t["direction"],
            t["entry_price"], t["bet_size"], t["potential_payout"], t["potential_profit"],
            t["risk_reward"], t["grok_fair_value"], t["grok_direction"], t["grok_confidence"],
            t["grok_reason"], t.get("grok_response_time", 0),
            t["gpt_fair_value"], t["gpt_direction"], t["gpt_confidence"],
            t["gpt_reason"], t.get("gpt_response_time", 0),
            t["consensus"], t["consensus_confidence"], t["combined_edge"],
            t["hours_left"], t["volume"], datetime.now(timezone.utc).isoformat(), 
            t["closes_at"], t.get("scan_mode"), t.get("scan_batch_id")
        ))
        db.commit()
        trade_log.info(f"SAVED | {t['tier']} | {t['direction']} | ${t['bet_size']:.2f} | {t['market_id']}")
        return True
    except sqlite3.IntegrityError:
        trade_log.debug(f"DUPLICATE | {t['market_id']}")
        return False

def log_scan_history(db, batch_id: str, modes: list, markets_fetched: int, 
                     stats: dict, trades_found: int, trades_saved: int,
                     grok_calls: int, gpt_calls: int, total_time: float, errors: list):
    """Log scan for historical analysis."""
    try:
        db.execute("""
            INSERT INTO scan_history (
                batch_id, timestamp, modes, markets_fetched, kalshi_count, poly_count,
                trades_found, trades_saved, grok_calls, gpt_calls, total_time_sec, errors
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            batch_id, datetime.now(timezone.utc).isoformat(),
            ",".join(modes), markets_fetched, stats.get("kalshi", 0), stats.get("poly", 0),
            trades_found, trades_saved, grok_calls, gpt_calls, total_time,
            json.dumps(errors) if errors else None
        ))
        db.commit()
    except Exception as e:
        log.error(f"Failed to log scan history: {e}")

# ============================================================================
# SMART SCANNER
# ============================================================================

async def run_dual_scan(db, modes: list = None):
    """Run dual-LLM scan with full logging."""
    batch_id = generate_batch_id()
    scan_start = datetime.now()
    
    if modes is None:
        modes = ["conservative", "standard", "longshot"]
    
    log.info(f"═══════════════════════════════════════════════════")
    log.info(f"SCAN START | batch={batch_id} | modes={modes}")
    log.info(f"═══════════════════════════════════════════════════")
    
    print(f"\n📡 Fetching markets... [batch: {batch_id}]")
    all_markets, fetch_stats = await fetch_markets()
    print(f"  Found {len(all_markets)} markets (Kalshi: {fetch_stats['kalshi']}, Poly: {fetch_stats['poly']})")
    
    all_trades = []
    all_errors = fetch_stats.get("errors", [])
    grok_calls = 0
    gpt_calls = 0
    
    for mode in modes:
        log.info(f"─── Scanning mode: {mode.upper()} ───")
        print(f"\n🔍 {mode.upper()} mode...")
        
        # Filter candidates
        if mode == "conservative":
            candidates = [m for m in all_markets 
                         if (m["price"] >= 0.65 or m["price"] <= 0.35) and m["vol"] > 20000]
        elif mode == "longshot":
            candidates = [m for m in all_markets 
                         if 0.10 <= m["price"] <= 0.40 and m["vol"] > 15000]
        else:
            candidates = [m for m in all_markets if m["vol"] > 30000]
        
        candidates.sort(key=lambda x: -x["vol"])
        candidates = candidates[:10]
        
        if not candidates:
            log.info(f"No candidates for {mode}")
            print(f"  No candidates")
            continue
        
        log.info(f"Candidates: {len(candidates)}")
        for c in candidates:
            market_log.debug(f"CANDIDATE | {mode} | {c['src']} | {c['price']:.0%} | ${c['vol']:,.0f} | {c['q'][:50]}")
        
        # Dual LLM analysis
        print(f"  🧠 Grok analyzing {len(candidates)} markets...")
        grok_results, grok_time = await analyze_with_grok(db, candidates, mode, batch_id)
        grok_calls += 1
        
        await asyncio.sleep(0.5)
        
        print(f"  🤖 GPT analyzing {len(candidates)} markets...")
        gpt_results, gpt_time = await analyze_with_gpt(db, candidates, mode, batch_id)
        gpt_calls += 1
        
        # Combine
        trades = combine_analyses(candidates, grok_results, gpt_results, mode, grok_time, gpt_time, batch_id)
        
        if trades:
            print(f"  ✅ Found {len(trades)} trades")
            log.info(f"Found {len(trades)} trades for {mode}")
            for t in trades:
                emoji = "✅✅" if t["consensus"] == "AGREE" else "⚠️"
                print(f"    {emoji} {t['direction']} {t['question'][:40]}... (+{t['combined_edge']:.1%})")
            all_trades.extend(trades)
        else:
            print(f"  ❌ No trades found")
            log.info(f"No trades for {mode}")
        
        await asyncio.sleep(1)
    
    # Save and alert
    new_count = 0
    for t in all_trades:
        if save_trade(db, t):
            new_count += 1
            await alert_trade(t)
            await asyncio.sleep(0.5)
    
    total_time = (datetime.now() - scan_start).total_seconds()
    
    # Log scan history
    log_scan_history(db, batch_id, modes, len(all_markets), fetch_stats,
                    len(all_trades), new_count, grok_calls, gpt_calls, total_time, all_errors)
    
    log.info(f"═══════════════════════════════════════════════════")
    log.info(f"SCAN COMPLETE | batch={batch_id} | trades={new_count} | time={total_time:.1f}s")
    log.info(f"═══════════════════════════════════════════════════")
    
    print(f"\n📊 Saved {new_count} new trades in {total_time:.1f}s")
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
    
    log.info(f"STATUS | total={r[0]} | wins={r[1] or 0} | losses={r[2] or 0} | pnl=${r[3] or 0:.2f}")
    
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
    parser = argparse.ArgumentParser(description="Oracle V4 - Full Logging")
    parser.add_argument("--scan", action="store_true", help="Full dual-LLM scan")
    parser.add_argument("--conservative", action="store_true", help="Conservative only")
    parser.add_argument("--aggressive", action="store_true", help="Aggressive only")
    parser.add_argument("--longshot", action="store_true", help="Longshots only")
    parser.add_argument("--resolve", action="store_true", help="Check resolutions")
    parser.add_argument("--status", action="store_true", help="Show status")
    parser.add_argument("--accuracy", action="store_true", help="Send accuracy report")
    parser.add_argument("--daemon", action="store_true", help="Run daemon")
    parser.add_argument("--interval", type=int, default=30, help="Daemon interval (minutes)")
    args = parser.parse_args()
    
    log.info(f"Oracle V4 starting | args={vars(args)}")
    
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
        log.info(f"═══════════════════════════════════════════════════")
        log.info(f"DAEMON MODE | interval={args.interval}min")
        log.info(f"═══════════════════════════════════════════════════")
        
        print(f"🤖 Starting V4 Daemon (interval: {args.interval}min)...")
        print(f"📁 Logs: {LOG_DIR}")
        
        await telegram(f"""🤖 <b>Oracle V4 Daemon Started</b>

📊 Dual-LLM Consensus System
⏱️ Scan Interval: {args.interval} min
📁 Logs: {LOG_DIR}

<i>Full logging enabled for deep analysis</i>""")
        
        cycle = 0
        while True:
            cycle += 1
            try:
                log.info(f"═══ DAEMON CYCLE {cycle} ═══")
                print(f"\n{'═'*50}")
                print(f"⏰ Cycle {cycle} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'═'*50}")
                
                await check_resolutions(db)
                await run_dual_scan(db)
                await show_status(db)
                
                # Send hourly reports
                if cycle % 2 == 0:  # Every 2 cycles
                    await send_accuracy_report(db)
                
                log.info(f"Sleeping {args.interval} minutes...")
                print(f"\n💤 Sleeping {args.interval} minutes... (Ctrl+C to stop)")
                await asyncio.sleep(args.interval * 60)
                
            except KeyboardInterrupt:
                log.info("Daemon stopped by user")
                await telegram("🛑 Oracle V4 Stopped")
                break
            except Exception as e:
                log.error(f"Daemon error: {e}\n{traceback.format_exc()}")
                print(f"❌ Error: {e}")
                await asyncio.sleep(300)
    else:
        await run_dual_scan(db)
        await show_status(db)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Interrupted by user")
    except Exception as e:
        log.error(f"Fatal error: {e}\n{traceback.format_exc()}")
        raise
