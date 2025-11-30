#!/usr/bin/env python3
"""
🧠 PREDICTION ORACLE V5 - SMART CACHING + REAL-TIME DATA
==========================================================
- Smart LLM response caching (avoid duplicate API calls)
- Market data caching with TTL
- Real-time news/context injection for better predictions
- Token-efficient batch processing
- Full audit trail
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
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Optional, Any
from logging.handlers import RotatingFileHandler
from pathlib import Path
from dotenv import load_dotenv
from functools import lru_cache
import time

load_dotenv("/root/prediction_oracle/.env")

# ============================================================================
# LOGGING SETUP
# ============================================================================

LOG_DIR = Path("/root/prediction_oracle/logs")
LOG_DIR.mkdir(exist_ok=True)

def setup_logging():
    """Setup comprehensive logging."""
    logger = logging.getLogger("oracle")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    
    detailed_format = logging.Formatter(
        '%(asctime)s.%(msecs)03d | %(levelname)-8s | %(name)-12s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    simple_format = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
    
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(simple_format)
    logger.addHandler(console)
    
    main_file = RotatingFileHandler(LOG_DIR / "oracle.log", maxBytes=10*1024*1024, backupCount=5)
    main_file.setLevel(logging.DEBUG)
    main_file.setFormatter(detailed_format)
    logger.addHandler(main_file)
    
    llm_logger = logging.getLogger("oracle.llm")
    llm_file = RotatingFileHandler(LOG_DIR / "llm_analysis.log", maxBytes=20*1024*1024, backupCount=10)
    llm_file.setLevel(logging.DEBUG)
    llm_file.setFormatter(detailed_format)
    llm_logger.addHandler(llm_file)
    
    cache_logger = logging.getLogger("oracle.cache")
    cache_file = RotatingFileHandler(LOG_DIR / "cache.log", maxBytes=5*1024*1024, backupCount=3)
    cache_file.setLevel(logging.DEBUG)
    cache_file.setFormatter(detailed_format)
    cache_logger.addHandler(cache_file)
    
    return logger

log = setup_logging()
llm_log = logging.getLogger("oracle.llm")
cache_log = logging.getLogger("oracle.cache")

# ============================================================================
# CONFIG
# ============================================================================

DB_PATH = "/root/prediction_oracle/oracle_v5.db"
TELEGRAM_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT = os.getenv("TELEGRAM_CHAT_ID", "")
XAI_KEY = os.getenv("XAI_API_KEY", "")
OPENAI_KEY = os.getenv("OPENAI_API_KEY", "")

BASE_BET = 5.0
BANKROLL = 1000.0

# Cache settings
LLM_CACHE_TTL_HOURS = 4  # Cache LLM responses for 4 hours
MARKET_CACHE_TTL_MIN = 15  # Cache market data for 15 min
NEWS_CACHE_TTL_MIN = 30  # Cache news for 30 min

log.info(f"Oracle V5 initialized | Smart Caching Enabled")
log.info(f"Cache TTL: LLM={LLM_CACHE_TTL_HOURS}h, Markets={MARKET_CACHE_TTL_MIN}m")

# ============================================================================
# DATABASE WITH CACHING
# ============================================================================

def init_db():
    log.debug("Initializing database with caching tables...")
    conn = sqlite3.connect(DB_PATH)
    conn.executescript("""
        -- Main trades table
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
            
            grok_fair_value REAL,
            grok_direction TEXT,
            grok_confidence TEXT,
            grok_reason TEXT,
            grok_response_time REAL,
            grok_cached INTEGER DEFAULT 0,
            
            gpt_fair_value REAL,
            gpt_direction TEXT,
            gpt_confidence TEXT,
            gpt_reason TEXT,
            gpt_response_time REAL,
            gpt_cached INTEGER DEFAULT 0,
            
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
            
            scan_mode TEXT,
            scan_batch_id TEXT,
            news_context TEXT
        );
        
        -- LLM Response Cache (avoid duplicate API calls)
        CREATE TABLE IF NOT EXISTS llm_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cache_key TEXT UNIQUE,
            llm TEXT,
            mode TEXT,
            market_hash TEXT,
            response TEXT,
            parsed_results TEXT,
            created_at TEXT,
            expires_at TEXT,
            hit_count INTEGER DEFAULT 0,
            tokens_saved INTEGER DEFAULT 0
        );
        
        -- Market Data Cache
        CREATE TABLE IF NOT EXISTS market_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT UNIQUE,
            source TEXT,
            data TEXT,
            fetched_at TEXT,
            expires_at TEXT
        );
        
        -- News/Context Cache
        CREATE TABLE IF NOT EXISTS news_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            category TEXT,
            headlines TEXT,
            context TEXT,
            fetched_at TEXT,
            expires_at TEXT
        );
        
        -- LLM Request Log (for analysis)
        CREATE TABLE IF NOT EXISTS llm_requests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            llm TEXT,
            mode TEXT,
            markets_count INTEGER,
            cache_hit INTEGER DEFAULT 0,
            response_time_ms REAL,
            tokens_used INTEGER,
            status TEXT,
            batch_id TEXT
        );
        
        -- Scan History
        CREATE TABLE IF NOT EXISTS scan_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            batch_id TEXT,
            timestamp TEXT,
            modes TEXT,
            markets_fetched INTEGER,
            trades_found INTEGER,
            trades_saved INTEGER,
            cache_hits INTEGER,
            tokens_saved INTEGER,
            total_time_sec REAL
        );
        
        -- Cache Statistics
        CREATE TABLE IF NOT EXISTS cache_stats (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT,
            llm_hits INTEGER DEFAULT 0,
            llm_misses INTEGER DEFAULT 0,
            tokens_saved INTEGER DEFAULT 0,
            api_calls_saved INTEGER DEFAULT 0
        );
        
        CREATE INDEX IF NOT EXISTS idx_llm_cache_key ON llm_cache(cache_key);
        CREATE INDEX IF NOT EXISTS idx_llm_cache_expires ON llm_cache(expires_at);
        CREATE INDEX IF NOT EXISTS idx_trades_market ON trades(market_id, direction);
        CREATE INDEX IF NOT EXISTS idx_market_cache_id ON market_cache(market_id);
    """)
    conn.commit()
    log.info("Database initialized with caching tables")
    return conn

# ============================================================================
# SMART CACHING SYSTEM
# ============================================================================

class SmartCache:
    """Intelligent caching for LLM responses and market data."""
    
    def __init__(self, db):
        self.db = db
        self.stats = {"hits": 0, "misses": 0, "tokens_saved": 0}
    
    def _generate_cache_key(self, llm: str, mode: str, markets: list) -> str:
        """Generate unique cache key based on markets and mode."""
        # Create hash of market IDs and prices (price changes invalidate cache)
        market_data = "|".join([
            f"{m['id']}:{m['price']:.2f}:{m['hours']:.0f}"
            for m in sorted(markets, key=lambda x: x['id'])
        ])
        content = f"{llm}|{mode}|{market_data}"
        return hashlib.sha256(content.encode()).hexdigest()[:32]
    
    def get_llm_response(self, llm: str, mode: str, markets: list) -> Optional[dict]:
        """Check cache for existing LLM response."""
        cache_key = self._generate_cache_key(llm, mode, markets)
        now = datetime.now(timezone.utc).isoformat()
        
        cursor = self.db.execute("""
            SELECT parsed_results, response, tokens_saved 
            FROM llm_cache 
            WHERE cache_key = ? AND expires_at > ?
        """, (cache_key, now))
        
        row = cursor.fetchone()
        if row:
            self.stats["hits"] += 1
            self.stats["tokens_saved"] += row[2] or 500  # Estimate
            
            # Update hit count
            self.db.execute("""
                UPDATE llm_cache SET hit_count = hit_count + 1 WHERE cache_key = ?
            """, (cache_key,))
            self.db.commit()
            
            cache_log.info(f"CACHE HIT | {llm} | {mode} | key={cache_key[:8]}... | saved ~{row[2]} tokens")
            
            return {
                "results": json.loads(row[0]) if row[0] else [],
                "raw": row[1],
                "cached": True
            }
        
        self.stats["misses"] += 1
        cache_log.debug(f"CACHE MISS | {llm} | {mode} | key={cache_key[:8]}...")
        return None
    
    def store_llm_response(self, llm: str, mode: str, markets: list, 
                           results: list, raw_response: str, tokens_used: int = 500):
        """Store LLM response in cache."""
        cache_key = self._generate_cache_key(llm, mode, markets)
        now = datetime.now(timezone.utc)
        expires = now + timedelta(hours=LLM_CACHE_TTL_HOURS)
        
        market_hash = hashlib.md5(
            json.dumps([m['id'] for m in markets]).encode()
        ).hexdigest()[:16]
        
        try:
            self.db.execute("""
                INSERT OR REPLACE INTO llm_cache 
                (cache_key, llm, mode, market_hash, response, parsed_results, 
                 created_at, expires_at, tokens_saved)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache_key, llm, mode, market_hash, 
                raw_response[:10000], json.dumps(results),
                now.isoformat(), expires.isoformat(), tokens_used
            ))
            self.db.commit()
            cache_log.info(f"CACHE STORE | {llm} | {mode} | key={cache_key[:8]}... | TTL={LLM_CACHE_TTL_HOURS}h")
        except Exception as e:
            cache_log.error(f"Cache store error: {e}")
    
    def cleanup_expired(self):
        """Remove expired cache entries."""
        now = datetime.now(timezone.utc).isoformat()
        cursor = self.db.execute("DELETE FROM llm_cache WHERE expires_at < ?", (now,))
        deleted = cursor.rowcount
        if deleted:
            cache_log.info(f"Cleaned up {deleted} expired cache entries")
        
        cursor = self.db.execute("DELETE FROM market_cache WHERE expires_at < ?", (now,))
        cursor = self.db.execute("DELETE FROM news_cache WHERE expires_at < ?", (now,))
        self.db.commit()
    
    def get_stats(self) -> dict:
        """Get cache statistics."""
        cursor = self.db.execute("""
            SELECT COUNT(*), SUM(hit_count), SUM(tokens_saved) FROM llm_cache
        """)
        row = cursor.fetchone()
        return {
            "entries": row[0] or 0,
            "total_hits": row[1] or 0,
            "tokens_saved": row[2] or 0,
            "session_hits": self.stats["hits"],
            "session_misses": self.stats["misses"],
            "hit_rate": (self.stats["hits"] / max(1, self.stats["hits"] + self.stats["misses"])) * 100
        }

# ============================================================================
# REAL-TIME NEWS/CONTEXT FETCHING
# ============================================================================

async def fetch_real_time_context(topic: str) -> str:
    """Fetch real-time context for better predictions."""
    # Using DuckDuckGo instant answers (free, no API key)
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            # Get current date context
            now = datetime.now(timezone.utc)
            date_context = f"Current date: {now.strftime('%B %d, %Y')}. "
            
            # Try to get relevant news
            resp = await c.get(
                "https://api.duckduckgo.com/",
                params={"q": topic, "format": "json", "no_html": 1}
            )
            if resp.status_code == 200:
                data = resp.json()
                abstract = data.get("Abstract", "")
                if abstract:
                    return date_context + abstract[:200]
            
            return date_context
    except Exception as e:
        log.debug(f"Context fetch error: {e}")
        return f"Current date: {datetime.now(timezone.utc).strftime('%B %d, %Y')}."

def extract_topic_from_question(question: str) -> str:
    """Extract main topic from market question for context lookup."""
    # Extract key entities
    keywords = []
    
    # Political figures
    politicians = ["Trump", "Biden", "Harris", "DeSantis", "Newsom", "Putin", "Zelensky", 
                   "Netanyahu", "Maduro", "Xi", "Modi", "Macron", "Starmer"]
    for p in politicians:
        if p.lower() in question.lower():
            keywords.append(p)
    
    # Topics
    topics = ["Bitcoin", "Ethereum", "crypto", "S&P", "Fed", "interest rate", 
              "Ukraine", "Russia", "Israel", "Gaza", "Taiwan", "China",
              "election", "Congress", "Supreme Court", "AI", "OpenAI"]
    for t in topics:
        if t.lower() in question.lower():
            keywords.append(t)
    
    if keywords:
        return " ".join(keywords[:3]) + " news 2025"
    return ""

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
    except Exception as e:
        log.error(f"Telegram error: {e}")

async def alert_trade(t: dict):
    """Send rich trade alert."""
    tier_emoji = {"CONSERVATIVE": "🛡️", "AGGRESSIVE": "🔥", "LONGSHOT": "🎰", "WHALE": "🐋"}.get(t["tier"], "📊")
    consensus_emoji = {"AGREE": "✅✅", "PARTIAL": "⚠️"}.get(t["consensus"], "")
    cache_emoji = "💾" if t.get("grok_cached") or t.get("gpt_cached") else ""
    
    hrs = t["hours_left"]
    time_str = f"{hrs:.0f}h" if hrs < 24 else f"{hrs/24:.1f}d"
    
    urgent = t["tier"] == "WHALE" or (t["consensus"] == "AGREE" and t["consensus_confidence"] == "ULTRA_HIGH")
    
    msg = f"""{tier_emoji} <b>{'🚨 HIGH CONVICTION!' if urgent else 'NEW TRADE'}</b> {cache_emoji}

<b>{t['direction']}</b> @ {t['entry_price']:.1%}
📋 {t['question'][:65]}

🧠 <b>Grok:</b> {t['grok_direction']} @ {t['grok_fair_value']:.0%} ({t['grok_confidence']})
🤖 <b>GPT:</b> {t['gpt_direction']} @ {t['gpt_fair_value']:.0%} ({t['gpt_confidence']})
{consensus_emoji} <b>{t['consensus']}</b> ({t['consensus_confidence']})

💰 Bet: <b>${t['bet_size']:.2f}</b> → Win: ${t['potential_payout']:.2f}
📈 Edge: +{t['combined_edge']:.1%} | R:R 1:{t['risk_reward']:.1f}
⏰ {time_str} | 🏷️ {t['tier']}"""
    
    await telegram(msg, urgent)

# ============================================================================
# OPTIMIZED LLM PROMPTS WITH REAL-TIME CONTEXT
# ============================================================================

def build_grok_prompt(markets: list, mode: str, date_context: str) -> str:
    """Build optimized Grok prompt with real-time context."""
    
    mode_instructions = {
        "conservative": """CONSERVATIVE MODE: Only recommend HIGH confidence bets (85%+).
Focus on near-term markets with clear catalysts. Safety over edge.""",
        
        "aggressive": """AGGRESSIVE MODE: Find HIGH EDGE opportunities (10%+).
Look for market inefficiencies and mispriced events.""",
        
        "longshot": """LONGSHOT MODE: Find UNDERVALUED underdogs (10-40% range).
Look for scenarios the market is underweighting.""",
        
        "standard": """STANDARD MODE: Find solid value bets with 5%+ edge.
Balance risk and reward."""
    }
    
    markets_text = "\n".join([
        f"{i}. [{m['src']}] \"{m['q']}\" | Price: {m['price']:.0%} YES | Closes: {m['hours']:.0f}h | Vol: ${m['vol']:,.0f}"
        for i, m in enumerate(markets[:10], 1)
    ])
    
    return f"""You are Grok, an elite prediction market analyst with access to real-time information.

📅 {date_context}

{mode_instructions.get(mode, mode_instructions['standard'])}

MARKETS TO ANALYZE:
{markets_text}

ANALYSIS FRAMEWORK:
1. Current probability vs TRUE probability based on latest events
2. Recent news, developments, or catalysts affecting each market
3. Historical patterns and base rates
4. Market sentiment vs reality

YOUR TASK:
For each market, provide your TRUE probability estimate. Use your real-time knowledge of current events.

RESPOND WITH JSON ARRAY ONLY:
[{{"m": 1, "fv": 0.XX, "d": "YES/NO/SKIP", "c": "HIGH/MEDIUM/LOW", "r": "reason based on current events (max 20 words)"}}]

IMPORTANT:
- fv = your fair value (true probability)
- d = direction to bet (YES if fv > price, NO if fv < price, SKIP if no edge)
- c = confidence in your assessment
- r = brief reason citing current events/data
- Your predictions are being tracked for accuracy

Be precise. Use current information."""

def build_gpt_prompt(markets: list, mode: str, date_context: str) -> str:
    """Build optimized GPT prompt with real-time context."""
    
    mode_context = {
        "conservative": "Conservative mode - only high confidence picks.",
        "aggressive": "Aggressive mode - find high edge opportunities.", 
        "longshot": "Longshot mode - find undervalued underdogs.",
        "standard": "Standard mode - solid value bets."
    }
    
    markets_text = "\n".join([
        f"{i}. \"{m['q']}\" - {m['price']:.0%} YES, {m['hours']:.0f}h"
        for i, m in enumerate(markets[:10], 1)
    ])
    
    return f"""Expert prediction market analyst. {date_context}

{mode_context.get(mode, mode_context['standard'])}

Markets:
{markets_text}

For each market, estimate TRUE probability based on current events and data.

JSON ONLY:
[{{"m":1,"fv":0.XX,"d":"YES/NO/SKIP","c":"HIGH/MEDIUM/LOW","r":"reason citing current events"}}]

fv=fair value, d=direction, c=confidence, r=reason"""

# ============================================================================
# DUAL LLM ANALYSIS WITH SMART CACHING
# ============================================================================

async def analyze_with_grok(db, cache: SmartCache, markets: list, mode: str, 
                            batch_id: str, date_context: str) -> tuple[list, float, bool]:
    """Analyze with Grok - returns (results, response_time_ms, was_cached)."""
    if not XAI_KEY or not markets:
        return [], 0, False
    
    # Check cache first
    cached = cache.get_llm_response("grok", mode, markets)
    if cached:
        llm_log.info(f"GROK CACHE HIT | mode={mode} | markets={len(markets)}")
        return cached["results"], 0, True
    
    prompt = build_grok_prompt(markets, mode, date_context)
    llm_log.info(f"GROK API CALL | mode={mode} | markets={len(markets)} | batch={batch_id}")
    llm_log.debug(f"GROK PROMPT:\n{prompt}")
    
    start = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=120) as c:
            resp = await c.post("https://api.x.ai/v1/chat/completions",
                headers={"Authorization": f"Bearer {XAI_KEY}"},
                json={
                    "model": "grok-4-1-fast-reasoning",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 1000,
                    "temperature": 0.3  # Lower temp for more consistent predictions
                })
            
            elapsed_ms = (time.time() - start) * 1000
            
            if resp.status_code != 200:
                llm_log.error(f"GROK ERROR | status={resp.status_code}")
                return [], elapsed_ms, False
            
            content = resp.json()["choices"][0]["message"]["content"]
            llm_log.debug(f"GROK RESPONSE:\n{content}")
            
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                results = json.loads(match.group())
                llm_log.info(f"GROK SUCCESS | results={len(results)} | time={elapsed_ms:.0f}ms")
                
                # Store in cache
                tokens_est = len(prompt.split()) + len(content.split())
                cache.store_llm_response("grok", mode, markets, results, content, tokens_est)
                
                return results, elapsed_ms, False
            
            llm_log.warning("GROK PARSE FAIL | no JSON found")
            return [], elapsed_ms, False
            
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        llm_log.error(f"GROK EXCEPTION | {e}")
        return [], elapsed_ms, False

async def analyze_with_gpt(db, cache: SmartCache, markets: list, mode: str,
                           batch_id: str, date_context: str) -> tuple[list, float, bool]:
    """Analyze with GPT - returns (results, response_time_ms, was_cached)."""
    if not OPENAI_KEY or not markets:
        return [], 0, False
    
    # Check cache first
    cached = cache.get_llm_response("gpt", mode, markets)
    if cached:
        llm_log.info(f"GPT CACHE HIT | mode={mode} | markets={len(markets)}")
        return cached["results"], 0, True
    
    prompt = build_gpt_prompt(markets, mode, date_context)
    llm_log.info(f"GPT API CALL | mode={mode} | markets={len(markets)} | batch={batch_id}")
    
    start = time.time()
    
    try:
        async with httpx.AsyncClient(timeout=60) as c:
            resp = await c.post("https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_KEY}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": 800,
                    "temperature": 0.3
                })
            
            elapsed_ms = (time.time() - start) * 1000
            
            if resp.status_code != 200:
                llm_log.error(f"GPT ERROR | status={resp.status_code}")
                return [], elapsed_ms, False
            
            content = resp.json()["choices"][0]["message"]["content"]
            
            match = re.search(r'\[.*\]', content, re.DOTALL)
            if match:
                results = json.loads(match.group())
                llm_log.info(f"GPT SUCCESS | results={len(results)} | time={elapsed_ms:.0f}ms")
                
                tokens_est = len(prompt.split()) + len(content.split())
                cache.store_llm_response("gpt", mode, markets, results, content, tokens_est)
                
                return results, elapsed_ms, False
            
            return [], elapsed_ms, False
            
    except Exception as e:
        elapsed_ms = (time.time() - start) * 1000
        llm_log.error(f"GPT EXCEPTION | {e}")
        return [], elapsed_ms, False

# ============================================================================
# COMBINE ANALYSES
# ============================================================================

def combine_analyses(markets: list, grok_results: list, gpt_results: list,
                     mode: str, grok_time: float, gpt_time: float,
                     grok_cached: bool, gpt_cached: bool, batch_id: str) -> list:
    """Combine LLM analyses with consensus logic."""
    
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
        
        # Consensus logic
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
            continue
        
        # Calculate edge
        if direction == "YES":
            entry = m["price"]
            edge = fair_value - entry
        else:
            entry = 1 - m["price"]
            edge = m["price"] - fair_value
        
        if entry <= 0 or entry >= 1 or edge < 0.03:
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
        
        # Tier and bet sizing
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
            "grok_response_time": grok_time,
            "grok_cached": 1 if grok_cached else 0,
            
            "gpt_fair_value": gpt_fv,
            "gpt_direction": gpt_dir,
            "gpt_confidence": gpt_conf,
            "gpt_reason": gpt.get("r", ""),
            "gpt_response_time": gpt_time,
            "gpt_cached": 1 if gpt_cached else 0,
            
            "consensus": consensus,
            "consensus_confidence": cons_conf,
            "combined_edge": edge,
            
            "hours_left": m["hours"],
            "volume": m["vol"],
            "closes_at": m.get("close_time", ""),
            "scan_mode": mode,
            "scan_batch_id": batch_id
        })
    
    return combined

# ============================================================================
# MARKET FETCHING
# ============================================================================

async def fetch_markets() -> tuple[list, dict]:
    """Fetch all markets from Kalshi + Polymarket."""
    markets = []
    stats = {"kalshi": 0, "poly": 0}
    now = datetime.now(timezone.utc)
    
    log.info("Fetching markets...")
    
    # Kalshi
    try:
        async with httpx.AsyncClient(timeout=30) as c:
            cursor = None
            for _ in range(5):
                params = {"limit": 100, "status": "open"}
                if cursor:
                    params["cursor"] = cursor
                
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
                            stats["kalshi"] += 1
                    except:
                        pass
                
                cursor = data.get("cursor")
                if not cursor:
                    break
    except Exception as e:
        log.error(f"Kalshi fetch error: {e}")
    
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
                    except:
                        pass
    except Exception as e:
        log.error(f"Polymarket fetch error: {e}")
    
    log.info(f"Fetched {len(markets)} markets (Kalshi: {stats['kalshi']}, Poly: {stats['poly']})")
    return markets, stats

# ============================================================================
# RESOLUTION TRACKING
# ============================================================================

async def check_resolutions(db):
    """Check and update resolved trades."""
    cursor = db.execute("SELECT * FROM trades WHERE outcome IS NULL")
    cols = [d[0] for d in cursor.description]
    trades = [dict(zip(cols, row)) for row in cursor.fetchall()]
    
    log.info(f"Checking {len(trades)} open trades for resolution")
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
            except:
                pass
        else:
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
            grok_correct = 1 if t["grok_direction"] == actual_result else 0
            gpt_correct = 1 if t["gpt_direction"] == actual_result else 0
            
            db.execute("""
                UPDATE trades SET outcome=?, pnl=?, resolved_at=?, actual_result=?,
                grok_correct=?, gpt_correct=? WHERE id=?
            """, (outcome, pnl, datetime.now(timezone.utc).isoformat(), actual_result,
                  grok_correct, gpt_correct, t["id"]))
            db.commit()
            
            emoji = "🎉" if outcome == "WIN" else "❌"
            msg = f"""{emoji} <b>RESOLVED: {outcome}</b>
{t['question'][:50]}...
Result: <b>{actual_result}</b> | P&L: <b>${pnl:+.2f}</b>
Grok: {'✅' if grok_correct else '❌'} | GPT: {'✅' if gpt_correct else '❌'}"""
            await telegram(msg)
            resolved.append(t)
            log.info(f"RESOLVED | {outcome} | ${pnl:+.2f} | {t['question'][:40]}")
    
    return resolved

def save_trade(db, t: dict) -> bool:
    """Save trade to database."""
    try:
        db.execute("""
            INSERT INTO trades (
                tier, source, market_id, question, direction, entry_price,
                bet_size, potential_payout, potential_profit, risk_reward,
                grok_fair_value, grok_direction, grok_confidence, grok_reason, 
                grok_response_time, grok_cached,
                gpt_fair_value, gpt_direction, gpt_confidence, gpt_reason,
                gpt_response_time, gpt_cached,
                consensus, consensus_confidence, combined_edge,
                hours_left, volume, created_at, closes_at, scan_mode, scan_batch_id
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
        """, (
            t["tier"], t["source"], t["market_id"], t["question"], t["direction"],
            t["entry_price"], t["bet_size"], t["potential_payout"], t["potential_profit"],
            t["risk_reward"], t["grok_fair_value"], t["grok_direction"], t["grok_confidence"],
            t["grok_reason"], t.get("grok_response_time", 0), t.get("grok_cached", 0),
            t["gpt_fair_value"], t["gpt_direction"], t["gpt_confidence"],
            t["gpt_reason"], t.get("gpt_response_time", 0), t.get("gpt_cached", 0),
            t["consensus"], t["consensus_confidence"], t["combined_edge"],
            t["hours_left"], t["volume"], datetime.now(timezone.utc).isoformat(),
            t["closes_at"], t.get("scan_mode"), t.get("scan_batch_id")
        ))
        db.commit()
        return True
    except sqlite3.IntegrityError:
        return False

# ============================================================================
# MAIN SCANNER
# ============================================================================

async def run_dual_scan(db, cache: SmartCache, modes: list = None):
    """Run dual-LLM scan with smart caching."""
    batch_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    
    if modes is None:
        modes = ["conservative", "standard", "longshot"]
    
    log.info(f"{'='*60}")
    log.info(f"SCAN START | batch={batch_id} | modes={modes}")
    log.info(f"{'='*60}")
    
    # Get current date context
    date_context = f"Current date: {datetime.now(timezone.utc).strftime('%B %d, %Y, %H:%M UTC')}."
    
    print(f"\n📡 Fetching markets... [batch: {batch_id}]")
    all_markets, fetch_stats = await fetch_markets()
    print(f"  Found {len(all_markets)} markets (Kalshi: {fetch_stats['kalshi']}, Poly: {fetch_stats['poly']})")
    
    # Cleanup expired cache
    cache.cleanup_expired()
    
    all_trades = []
    cache_hits = 0
    
    for mode in modes:
        log.info(f"--- Scanning: {mode.upper()} ---")
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
            print(f"  No candidates")
            continue
        
        # Dual LLM analysis with caching
        print(f"  🧠 Grok analyzing {len(candidates)} markets...")
        grok_results, grok_time, grok_cached = await analyze_with_grok(
            db, cache, candidates, mode, batch_id, date_context)
        if grok_cached:
            cache_hits += 1
            print(f"    💾 Cache hit!")
        
        await asyncio.sleep(0.3)
        
        print(f"  🤖 GPT analyzing {len(candidates)} markets...")
        gpt_results, gpt_time, gpt_cached = await analyze_with_gpt(
            db, cache, candidates, mode, batch_id, date_context)
        if gpt_cached:
            cache_hits += 1
            print(f"    💾 Cache hit!")
        
        # Combine
        trades = combine_analyses(candidates, grok_results, gpt_results, mode,
                                 grok_time, gpt_time, grok_cached, gpt_cached, batch_id)
        
        if trades:
            print(f"  ✅ Found {len(trades)} trades")
            for t in trades:
                emoji = "✅✅" if t["consensus"] == "AGREE" else "⚠️"
                cached = "💾" if t["grok_cached"] or t["gpt_cached"] else ""
                print(f"    {emoji} {t['direction']} {t['question'][:40]}... +{t['combined_edge']:.1%} {cached}")
            all_trades.extend(trades)
        else:
            print(f"  ❌ No trades")
        
        await asyncio.sleep(0.5)
    
    # Save and alert
    new_count = 0
    for t in all_trades:
        if save_trade(db, t):
            new_count += 1
            await alert_trade(t)
            await asyncio.sleep(0.3)
    
    # Cache stats
    stats = cache.get_stats()
    log.info(f"CACHE STATS | hits={stats['session_hits']} | misses={stats['session_misses']} | "
             f"tokens_saved={stats['tokens_saved']}")
    
    print(f"\n📊 Saved {new_count} trades | Cache hits: {cache_hits} | Tokens saved: ~{stats['tokens_saved']}")
    return all_trades

async def show_status(db, cache: SmartCache):
    """Show portfolio and cache status."""
    cursor = db.execute("""
        SELECT COUNT(*), 
               SUM(CASE WHEN outcome='WIN' THEN 1 ELSE 0 END),
               SUM(CASE WHEN outcome='LOSS' THEN 1 ELSE 0 END),
               SUM(CASE WHEN outcome IS NOT NULL THEN pnl ELSE 0 END),
               SUM(CASE WHEN outcome IS NULL THEN bet_size ELSE 0 END),
               COUNT(CASE WHEN outcome IS NULL THEN 1 END)
        FROM trades
    """)
    r = cursor.fetchone()
    
    print("\n📊 PORTFOLIO STATUS")
    print(f"Total: {r[0]} | Wins: {r[1] or 0} | Losses: {r[2] or 0}")
    print(f"P&L: ${r[3] or 0:.2f} | At Risk: ${r[4] or 0:.2f} | Open: {r[5] or 0}")
    
    # LLM accuracy
    for llm in ["grok", "gpt"]:
        cursor = db.execute(f"""
            SELECT COUNT(*), SUM({llm}_correct)
            FROM trades WHERE outcome IS NOT NULL AND {llm}_correct IS NOT NULL
        """)
        r = cursor.fetchone()
        acc = (r[1]/r[0]*100) if r[0] and r[1] else 0
        print(f"  {llm.upper()}: {acc:.1f}% ({r[0] or 0} predictions)")
    
    # Cache stats
    stats = cache.get_stats()
    print(f"\n💾 CACHE: {stats['entries']} entries | {stats['total_hits']} total hits | ~{stats['tokens_saved']} tokens saved")

# ============================================================================
# MAIN
# ============================================================================

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Oracle V5 - Smart Caching")
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--conservative", action="store_true")
    parser.add_argument("--longshot", action="store_true")
    parser.add_argument("--resolve", action="store_true")
    parser.add_argument("--status", action="store_true")
    parser.add_argument("--daemon", action="store_true")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--clear-cache", action="store_true")
    args = parser.parse_args()
    
    db = init_db()
    cache = SmartCache(db)
    
    if args.clear_cache:
        db.execute("DELETE FROM llm_cache")
        db.commit()
        print("✅ Cache cleared")
        return
    
    if args.status:
        await show_status(db, cache)
    elif args.resolve:
        await check_resolutions(db)
    elif args.conservative:
        await run_dual_scan(db, cache, ["conservative"])
    elif args.longshot:
        await run_dual_scan(db, cache, ["longshot"])
    elif args.daemon:
        log.info(f"DAEMON MODE | interval={args.interval}min")
        print(f"🤖 Starting V5 Daemon (interval: {args.interval}min)")
        print(f"💾 Smart caching enabled (TTL: {LLM_CACHE_TTL_HOURS}h)")
        
        await telegram(f"""🤖 <b>Oracle V5 Daemon Started</b>

💾 Smart Caching: {LLM_CACHE_TTL_HOURS}h TTL
⏱️ Interval: {args.interval} min
📊 Dual-LLM Consensus Active""")
        
        cycle = 0
        while True:
            cycle += 1
            try:
                print(f"\n{'='*50}")
                print(f"⏰ Cycle {cycle} | {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"{'='*50}")
                
                await check_resolutions(db)
                await run_dual_scan(db, cache)
                await show_status(db, cache)
                
                print(f"\n💤 Sleeping {args.interval} minutes...")
                await asyncio.sleep(args.interval * 60)
                
            except KeyboardInterrupt:
                await telegram("🛑 Oracle V5 Stopped")
                break
            except Exception as e:
                log.error(f"Daemon error: {e}\n{traceback.format_exc()}")
                await asyncio.sleep(300)
    else:
        await run_dual_scan(db, cache)
        await show_status(db, cache)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
