#!/usr/bin/env python3
"""
Paper Trader v2 - Automated paper trading with continuous monitoring
Places bets on high-edge opportunities and tracks LLM performance by category
"""
import asyncio
import sqlite3
import httpx
import json
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict
import os
import sys
import signal

sys.path.insert(0, '/root/prediction_oracle')
os.chdir('/root/prediction_oracle')

# Import the multi-scanner functions
from multi_scanner import (
    fetch_polymarket, fetch_polymarket_events, 
    fetch_manifold, fetch_manifold_search,
    fetch_crypto_predictions, fetch_espn_sports, 
    fetch_stock_predictions, fetch_weather_predictions,
    analyze_with_llm, categorize, save_to_db
)

# Configuration - DYNAMIC BET SIZING
MIN_EDGE = 0.05  # Minimum 5% edge to place bet
MAX_BETS_PER_SCAN = 5  # Max new bets per scan
SCAN_INTERVAL = 1800  # 30 minutes between scans
DB_PATH = 'paper_trades.db'

# Bet sizing tiers based on confidence and edge
def calculate_bet_size(edge: float, confidence: str, hours_left: float) -> float:
    """
    Dynamic bet sizing:
    - $5 base for longshots (low confidence, high edge but risky)
    - $25 for medium confidence with good edge
    - $50 for high confidence with solid edge
    - $100 for high confidence + high edge + shorter timeframe
    """
    edge_pct = edge * 100  # Convert to percentage
    
    # High confidence bets
    if confidence == 'HIGH':
        if edge_pct >= 15 and hours_left <= 72:  # 15%+ edge, closes in 3 days
            return 100.0  # MAX BET - very confident, high edge, soon
        elif edge_pct >= 10:
            return 50.0   # Strong bet
        elif edge_pct >= 5:
            return 25.0   # Solid bet
        else:
            return 10.0   # Still worth it
    
    # Medium confidence bets
    elif confidence == 'MEDIUM':
        if edge_pct >= 20:
            return 25.0   # High edge compensates for medium confidence
        elif edge_pct >= 10:
            return 15.0   # Decent opportunity
        else:
            return 10.0   # Small position
    
    # Low confidence - longshots only
    else:
        if edge_pct >= 30:
            return 10.0   # Big edge, worth a small bet
        else:
            return 5.0    # Minimum bet for longshots

# Global flag for graceful shutdown
running = True

def signal_handler(sig, frame):
    global running
    print("\n\n🛑 Shutdown signal received. Finishing current scan...")
    running = False

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


def init_database():
    """Initialize the paper trading database with all tracking tables."""
    db = sqlite3.connect(DB_PATH)
    c = db.cursor()
    
    # Main trades table
    c.execute('''CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY,
        trade_id TEXT UNIQUE,
        source TEXT,
        market_id TEXT,
        category TEXT,
        question TEXT,
        direction TEXT,
        entry_price REAL,
        bet_size REAL,
        potential_payout REAL,
        potential_profit REAL,
        edge REAL,
        llm_fair REAL,
        llm_confidence TEXT,
        llm_reason TEXT,
        grok_fair REAL,
        grok_dir TEXT,
        gpt_fair REAL,
        gpt_dir TEXT,
        hours_left REAL,
        closes_at TEXT,
        created_at TEXT,
        resolved_at TEXT,
        outcome TEXT,
        actual_price REAL,
        pnl REAL,
        grok_correct INTEGER,
        gpt_correct INTEGER
    )''')
    
    # Category performance tracking
    c.execute('''CREATE TABLE IF NOT EXISTS category_stats (
        category TEXT PRIMARY KEY,
        total_trades INTEGER DEFAULT 0,
        wins INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        total_pnl REAL DEFAULT 0,
        avg_edge REAL DEFAULT 0,
        grok_accuracy REAL DEFAULT 0,
        gpt_accuracy REAL DEFAULT 0,
        last_updated TEXT
    )''')
    
    # Source performance tracking  
    c.execute('''CREATE TABLE IF NOT EXISTS source_stats (
        source TEXT PRIMARY KEY,
        total_trades INTEGER DEFAULT 0,
        wins INTEGER DEFAULT 0,
        losses INTEGER DEFAULT 0,
        total_pnl REAL DEFAULT 0,
        avg_edge REAL DEFAULT 0,
        last_updated TEXT
    )''')
    
    # Scan history
    c.execute('''CREATE TABLE IF NOT EXISTS scan_history (
        id INTEGER PRIMARY KEY,
        scan_time TEXT,
        markets_found INTEGER,
        opportunities INTEGER,
        trades_placed INTEGER,
        sources TEXT
    )''')
    
    db.commit()
    db.close()
    print("✓ Database initialized")


def get_existing_trade_ids(db) -> set:
    """Get all existing trade IDs to avoid duplicates."""
    c = db.cursor()
    c.execute("SELECT trade_id FROM trades WHERE resolved_at IS NULL")
    return {row[0] for row in c.fetchall()}


def place_paper_trade(db, market: Dict) -> tuple:
    """Place a paper trade on a market. Returns (success, bet_size)."""
    c = db.cursor()
    
    # Generate unique trade ID
    trade_id = f"{market['source']}-{market['id']}-{market['llm_direction']}"
    
    # Check if already traded
    c.execute("SELECT id FROM trades WHERE trade_id = ?", (trade_id,))
    if c.fetchone():
        return False, 0
    
    # Calculate trade details
    if market['llm_direction'] == 'YES':
        entry_price = market['price']
    else:
        entry_price = 1 - market['price']
    
    # DYNAMIC BET SIZING based on confidence and edge
    bet_size = calculate_bet_size(
        edge=market.get('edge', 0),
        confidence=market.get('llm_confidence', 'LOW'),
        hours_left=market.get('hours_left', 999)
    )
    
    potential_payout = bet_size / entry_price if entry_price > 0 else 0
    potential_profit = potential_payout - bet_size
    
    grok = market.get('grok', {}) or {}
    gpt = market.get('gpt', {}) or {}
    
    now = datetime.now(timezone.utc).isoformat()
    
    c.execute('''INSERT INTO trades 
        (trade_id, source, market_id, category, question, direction,
         entry_price, bet_size, potential_payout, potential_profit, edge,
         llm_fair, llm_confidence, llm_reason,
         grok_fair, grok_dir, gpt_fair, gpt_dir,
         hours_left, closes_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (trade_id, market['source'], market['id'], market['category'],
         market['question'], market['llm_direction'],
         entry_price, bet_size, potential_payout, potential_profit, market.get('edge', 0),
         market.get('llm_fair'), market.get('llm_confidence'), market.get('llm_reason'),
         grok.get('fair'), grok.get('direction'), gpt.get('fair'), gpt.get('direction'),
         market.get('hours_left'), market.get('closes_at'), now))
    
    db.commit()
    return True, bet_size


async def check_resolutions(db):
    """Check for resolved markets and update trades."""
    c = db.cursor()
    
    # Get unresolved trades that should have closed
    now = datetime.now(timezone.utc)
    c.execute("""
        SELECT id, trade_id, source, market_id, direction, entry_price, 
               bet_size, closes_at, category, grok_dir, gpt_dir
        FROM trades 
        WHERE resolved_at IS NULL AND closes_at < ?
    """, (now.isoformat(),))
    
    pending = c.fetchall()
    
    if not pending:
        return 0
    
    resolved_count = 0
    
    async with httpx.AsyncClient(timeout=30) as client:
        for row in pending:
            trade_id, source, market_id, direction = row[1], row[2], row[3], row[4]
            entry_price, bet_size, closes_at = row[5], row[6], row[7]
            category, grok_dir, gpt_dir = row[8], row[9], row[10]
            
            outcome = None
            actual_price = None
            
            # Try to get resolution from source
            if source == 'POLYMARKET':
                try:
                    resp = await client.get(
                        f"https://gamma-api.polymarket.com/markets/{market_id}"
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get('closed'):
                            # Check outcome
                            prices_str = data.get('outcomePrices', '')
                            if prices_str:
                                prices = [float(p.strip('" ')) for p in prices_str.strip('[]').split(',') if p.strip()]
                                if prices:
                                    actual_price = prices[0]
                                    # If YES price is > 0.95, YES won; if < 0.05, NO won
                                    if actual_price > 0.95:
                                        outcome = 'YES'
                                    elif actual_price < 0.05:
                                        outcome = 'NO'
                except:
                    pass
            
            elif source == 'MANIFOLD':
                try:
                    resp = await client.get(
                        f"https://api.manifold.markets/v0/market/{market_id}"
                    )
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get('isResolved'):
                            resolution = data.get('resolution')
                            if resolution == 'YES':
                                outcome = 'YES'
                                actual_price = 1.0
                            elif resolution == 'NO':
                                outcome = 'NO'
                                actual_price = 0.0
                except:
                    pass
            
            elif source == 'CRYPTO_PRED':
                # Crypto predictions - check current price vs threshold
                try:
                    # Parse the market_id to get crypto and threshold
                    # Format: CRYPTO-BTC-24h-91000
                    parts = market_id.split('-')
                    if len(parts) >= 4:
                        symbol = parts[1].lower()
                        threshold = float(parts[3])
                        
                        crypto_map = {
                            'btc': 'bitcoin', 'eth': 'ethereum', 
                            'sol': 'solana', 'doge': 'dogecoin', 'xrp': 'ripple'
                        }
                        
                        if symbol in crypto_map:
                            resp = await client.get(
                                "https://api.coingecko.com/api/v3/simple/price",
                                params={"ids": crypto_map[symbol], "vs_currencies": "usd"}
                            )
                            if resp.status_code == 200:
                                data = resp.json()
                                current = data.get(crypto_map[symbol], {}).get('usd', 0)
                                if current > 0:
                                    actual_price = 1.0 if current > threshold else 0.0
                                    outcome = 'YES' if current > threshold else 'NO'
                except:
                    pass
            
            elif source.startswith('ESPN'):
                # Sports games - check if game ended
                # For now, mark as pending if not resolved
                pass
            
            # If we have an outcome, update the trade
            if outcome:
                # Calculate P&L
                if direction == outcome:
                    pnl = (bet_size / entry_price) - bet_size  # Win
                else:
                    pnl = -bet_size  # Loss
                
                # Check LLM correctness
                grok_correct = 1 if grok_dir == outcome else 0 if grok_dir in ['YES', 'NO'] else None
                gpt_correct = 1 if gpt_dir == outcome else 0 if gpt_dir in ['YES', 'NO'] else None
                
                c.execute("""
                    UPDATE trades SET 
                        resolved_at = ?, outcome = ?, actual_price = ?, 
                        pnl = ?, grok_correct = ?, gpt_correct = ?
                    WHERE id = ?
                """, (now.isoformat(), outcome, actual_price, pnl, grok_correct, gpt_correct, row[0]))
                
                # Update category stats
                update_category_stats(db, category, pnl > 0, pnl, grok_correct, gpt_correct)
                
                # Update source stats
                update_source_stats(db, source, pnl > 0, pnl)
                
                resolved_count += 1
                
                win_loss = "✅ WIN" if pnl > 0 else "❌ LOSS"
                print(f"   {win_loss}: {direction} on '{row[1][:40]}' → {outcome} (${pnl:+.2f})")
    
    db.commit()
    return resolved_count


def update_category_stats(db, category: str, won: bool, pnl: float, 
                          grok_correct: Optional[int], gpt_correct: Optional[int]):
    """Update category performance statistics."""
    c = db.cursor()
    now = datetime.now(timezone.utc).isoformat()
    
    c.execute("SELECT * FROM category_stats WHERE category = ?", (category,))
    existing = c.fetchone()
    
    if existing:
        c.execute("""
            UPDATE category_stats SET
                total_trades = total_trades + 1,
                wins = wins + ?,
                losses = losses + ?,
                total_pnl = total_pnl + ?,
                last_updated = ?
            WHERE category = ?
        """, (1 if won else 0, 0 if won else 1, pnl, now, category))
    else:
        c.execute("""
            INSERT INTO category_stats (category, total_trades, wins, losses, total_pnl, last_updated)
            VALUES (?, 1, ?, ?, ?, ?)
        """, (category, 1 if won else 0, 0 if won else 1, pnl, now))
    
    db.commit()


def update_source_stats(db, source: str, won: bool, pnl: float):
    """Update source performance statistics."""
    c = db.cursor()
    now = datetime.now(timezone.utc).isoformat()
    
    c.execute("SELECT * FROM source_stats WHERE source = ?", (source,))
    existing = c.fetchone()
    
    if existing:
        c.execute("""
            UPDATE source_stats SET
                total_trades = total_trades + 1,
                wins = wins + ?,
                losses = losses + ?,
                total_pnl = total_pnl + ?,
                last_updated = ?
            WHERE source = ?
        """, (1 if won else 0, 0 if won else 1, pnl, now, source))
    else:
        c.execute("""
            INSERT INTO source_stats (source, total_trades, wins, losses, total_pnl, last_updated)
            VALUES (?, 1, ?, ?, ?, ?)
        """, (source, 1 if won else 0, 0 if won else 1, pnl, now))
    
    db.commit()


def display_portfolio(db):
    """Display current portfolio status."""
    c = db.cursor()
    
    print("\n" + "="*70)
    print("📊 PORTFOLIO STATUS")
    print("="*70)
    
    # Open positions
    c.execute("""
        SELECT direction, question, entry_price, bet_size, potential_profit, 
               edge, hours_left, category, source
        FROM trades WHERE resolved_at IS NULL
        ORDER BY hours_left
    """)
    open_trades = c.fetchall()
    
    if open_trades:
        print(f"\n🔓 OPEN POSITIONS ({len(open_trades)}):")
        print("-"*70)
        total_risk = 0
        total_potential = 0
        for t in open_trades[:10]:  # Show first 10
            hrs = t[6]
            time_str = f"{hrs:.0f}h" if hrs and hrs < 48 else f"{hrs/24:.0f}d" if hrs else "?"
            print(f"  [{t[7][:6]}] {t[0]} @ {t[2]:.0%} | ${t[3]:.0f} risk → ${t[4]:+.2f} | {time_str}")
            print(f"           {t[1][:50]}...")
            total_risk += t[3]
            total_potential += t[4]
        
        if len(open_trades) > 10:
            print(f"  ... and {len(open_trades) - 10} more positions")
        
        print(f"\n  Total at risk: ${total_risk:.2f}")
        print(f"  Total potential profit: ${total_potential:.2f}")
    
    # Resolved trades summary
    c.execute("""
        SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), 
               SUM(CASE WHEN pnl <= 0 THEN 1 ELSE 0 END), SUM(pnl)
        FROM trades WHERE resolved_at IS NOT NULL
    """)
    resolved = c.fetchone()
    
    if resolved[0]:
        win_rate = resolved[1] / resolved[0] * 100 if resolved[0] > 0 else 0
        print(f"\n📈 RESOLVED TRADES:")
        print("-"*70)
        print(f"  Total: {resolved[0]} | Wins: {resolved[1]} | Losses: {resolved[2]}")
        print(f"  Win Rate: {win_rate:.1f}%")
        print(f"  Total P&L: ${resolved[3]:+.2f}")
    
    # Category performance
    c.execute("""
        SELECT category, total_trades, wins, losses, total_pnl
        FROM category_stats ORDER BY total_pnl DESC
    """)
    cat_stats = c.fetchall()
    
    if cat_stats:
        print(f"\n🏷️ PERFORMANCE BY CATEGORY:")
        print("-"*70)
        for cat in cat_stats:
            win_rate = cat[2] / cat[1] * 100 if cat[1] > 0 else 0
            print(f"  {cat[0]:12} | {cat[1]:3} trades | {win_rate:5.1f}% win | ${cat[4]:+8.2f}")
    
    # Source performance
    c.execute("""
        SELECT source, total_trades, wins, losses, total_pnl
        FROM source_stats ORDER BY total_pnl DESC
    """)
    src_stats = c.fetchall()
    
    if src_stats:
        print(f"\n📍 PERFORMANCE BY SOURCE:")
        print("-"*70)
        for src in src_stats:
            win_rate = src[2] / src[1] * 100 if src[1] > 0 else 0
            print(f"  {src[0]:12} | {src[1]:3} trades | {win_rate:5.1f}% win | ${src[4]:+8.2f}")
    
    print("\n" + "="*70)


async def run_scan_cycle(db):
    """Run one complete scan and trading cycle."""
    print(f"\n{'='*70}")
    print(f"🔄 SCAN CYCLE - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*70}")
    
    # First, check for resolved trades
    print("\n📋 Checking resolutions...")
    resolved = await check_resolutions(db)
    print(f"   Resolved {resolved} trades")
    
    # Fetch markets from all sources
    print("\n📡 Fetching markets...")
    
    max_hours = 720  # 30 days
    
    poly_task = fetch_polymarket(max_hours)
    poly_events_task = fetch_polymarket_events(max_hours)
    manifold_task = fetch_manifold(max_hours)
    manifold_hot_task = fetch_manifold_search(max_hours)
    crypto_task = fetch_crypto_predictions(168)
    espn_task = fetch_espn_sports(72)
    stock_task = fetch_stock_predictions(168)
    weather_task = fetch_weather_predictions(168)
    
    poly, poly_events, manifold, manifold_hot, crypto, espn, stocks, weather = await asyncio.gather(
        poly_task, poly_events_task, manifold_task, manifold_hot_task,
        crypto_task, espn_task, stock_task, weather_task
    )
    
    # Combine and dedupe
    all_markets = poly + poly_events + manifold + manifold_hot + crypto + espn + stocks + weather
    
    seen_questions = set()
    unique_markets = []
    for m in all_markets:
        q_key = m['question'][:50].lower()
        if q_key not in seen_questions:
            seen_questions.add(q_key)
            unique_markets.append(m)
    
    all_markets = sorted(unique_markets, key=lambda x: x['hours_left'])
    
    print(f"   Found {len(all_markets)} total markets (after dedup)")
    print(f"   - Polymarket: {len(poly)} + {len(poly_events)} events")
    print(f"   - Manifold: {len(manifold)} + {len(manifold_hot)} hot")
    print(f"   - ESPN Sports: {len(espn)}")
    print(f"   - Crypto: {len(crypto)}")
    print(f"   - Stocks: {len(stocks)}")
    print(f"   - Weather: {len(weather)}")
    
    if not all_markets:
        return
    
    # Get existing trade IDs to avoid duplicates
    existing_ids = get_existing_trade_ids(db)
    
    # Filter out already traded markets
    new_markets = []
    for m in all_markets:
        trade_id = f"{m['source']}-{m['id']}-YES"
        trade_id_no = f"{m['source']}-{m['id']}-NO"
        if trade_id not in existing_ids and trade_id_no not in existing_ids:
            new_markets.append(m)
    
    print(f"   {len(new_markets)} new markets (not already traded)")
    
    if not new_markets:
        print("   No new markets to analyze")
        return
    
    # Analyze with LLMs
    print("\n🤖 Running LLM analysis...")
    analyzed = await analyze_with_llm(new_markets, max_calls=20)
    
    # Find high-edge opportunities
    opportunities = [
        m for m in analyzed 
        if m.get('llm_direction') in ['YES', 'NO'] 
        and m.get('edge', 0) >= MIN_EDGE
    ]
    
    opportunities.sort(key=lambda x: x.get('edge', 0), reverse=True)
    
    print(f"\n🎯 Found {len(opportunities)} opportunities with ≥{MIN_EDGE:.0%} edge")
    
    # Place trades on best opportunities
    trades_placed = 0
    total_risked = 0
    for m in opportunities[:MAX_BETS_PER_SCAN]:
        success, bet_size = place_paper_trade(db, m)
        if success:
            trades_placed += 1
            total_risked += bet_size
            
            if m['llm_direction'] == 'YES':
                entry = m['price']
            else:
                entry = 1 - m['price']
            
            payout = bet_size / entry if entry > 0 else 0
            profit = payout - bet_size
            
            hrs = m['hours_left']
            time_str = f"{hrs:.0f}h" if hrs < 48 else f"{hrs/24:.0f}d"
            
            # Show bet tier
            if bet_size >= 100:
                tier = "💰 MAX BET"
            elif bet_size >= 50:
                tier = "🔥 STRONG"
            elif bet_size >= 25:
                tier = "✅ SOLID"
            else:
                tier = "🎲 SMALL"
            
            print(f"\n   📝 {tier}: {m['llm_direction']} @ {entry:.0%}")
            print(f"      [{m['category']}] {m['question'][:45]}...")
            print(f"      Edge: +{m['edge']:.1%} | Conf: {m.get('llm_confidence', 'N/A')}")
            print(f"      ${bet_size:.0f} → ${payout:.2f} (+${profit:.2f})")
            print(f"      Closes: {time_str} | Source: {m['source']}")
    
    # Log scan
    c = db.cursor()
    c.execute("""
        INSERT INTO scan_history (scan_time, markets_found, opportunities, trades_placed, sources)
        VALUES (?, ?, ?, ?, ?)
    """, (datetime.now(timezone.utc).isoformat(), len(all_markets), len(opportunities), 
          trades_placed, f"POLY:{len(poly)},MANI:{len(manifold)},ESPN:{len(espn)},CRYPTO:{len(crypto)}"))
    db.commit()
    
    print(f"\n✅ Cycle complete: {trades_placed} trades, ${total_risked:.0f} risked this scan")
    
    # Show portfolio status
    display_portfolio(db)


async def main():
    # Check for --once flag
    once_mode = '--once' in sys.argv
    
    print("="*70)
    print("🤖 PAPER TRADER v2 - Dynamic Bet Sizing")
    print("="*70)
    print("  Bet Tiers:")
    print("    💰 $100 - HIGH confidence + 15%+ edge + <3 days")
    print("    🔥 $50  - HIGH confidence + 10%+ edge")
    print("    ✅ $25  - HIGH/MED confidence + good edge")
    print("    🎲 $5-15 - Lower confidence longshots")
    print(f"  Min Edge: {MIN_EDGE:.0%}")
    print(f"  Max Bets/Scan: {MAX_BETS_PER_SCAN}")
    print(f"  Scan Interval: {SCAN_INTERVAL}s ({SCAN_INTERVAL//60} min)")
    print(f"  Database: {DB_PATH}")
    print(f"  Mode: {'Single scan' if once_mode else 'Continuous'}")
    print("="*70)
    
    if not once_mode:
        print("\nPress Ctrl+C to stop gracefully\n")
    
    # Initialize database
    init_database()
    
    db = sqlite3.connect(DB_PATH)
    
    scan_count = 0
    
    while running:
        try:
            scan_count += 1
            print(f"\n{'#'*70}")
            print(f"# SCAN #{scan_count}")
            print(f"{'#'*70}")
            
            await run_scan_cycle(db)
            
            if once_mode or not running:
                break
            
            # Wait for next scan
            print(f"\n⏰ Next scan in {SCAN_INTERVAL//60} minutes...")
            print("   (Press Ctrl+C to stop)")
            
            # Sleep in small increments to allow graceful shutdown
            for _ in range(SCAN_INTERVAL // 10):
                if not running:
                    break
                await asyncio.sleep(10)
                
        except Exception as e:
            print(f"\n❌ Error during scan: {e}")
            import traceback
            traceback.print_exc()
            
            if once_mode or not running:
                break
            
            print("   Waiting 60s before retry...")
            await asyncio.sleep(60)
    
    # Final portfolio display
    print("\n" + "="*70)
    print("� FINAL PORTFOLIO STATUS")
    print("="*70)
    display_portfolio(db)
    
    db.close()
    print("\n✅ Paper trader stopped. Database saved.")


if __name__ == "__main__":
    asyncio.run(main())
