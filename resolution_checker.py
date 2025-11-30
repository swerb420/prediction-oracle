#!/usr/bin/env python3
"""
Resolution Checker - Monitors and resolves paper trades
Tracks LLM performance by category
"""

import asyncio
import sqlite3
import httpx
from datetime import datetime, timezone
from typing import Optional
import json

DB_PATH = "/root/prediction_oracle/paper_trades.db"

def get_db():
    return sqlite3.connect(DB_PATH)

async def fetch_manifold_resolution(market_id: str) -> Optional[dict]:
    """Check if a Manifold market has resolved"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"https://api.manifold.markets/v0/market/{market_id}")
            if resp.status_code == 200:
                data = resp.json()
                if data.get("isResolved"):
                    return {
                        "resolved": True,
                        "outcome": "YES" if data.get("resolution") == "YES" else "NO",
                        "final_prob": data.get("probability", 0.5)
                    }
                return {"resolved": False, "current_prob": data.get("probability")}
    except Exception as e:
        print(f"  Error checking Manifold {market_id}: {e}")
    return None

async def fetch_polymarket_resolution(market_id: str) -> Optional[dict]:
    """Check if a Polymarket market has resolved"""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            resp = await client.get(f"https://gamma-api.polymarket.com/markets/{market_id}")
            if resp.status_code == 200:
                data = resp.json()
                # Check if resolved
                if data.get("resolved") or data.get("closed"):
                    outcome_price = data.get("outcomePrices", "")
                    if outcome_price:
                        try:
                            # Handle string like '"0.89"' or '[0.89, 0.11]'
                            if isinstance(outcome_price, str):
                                outcome_price = outcome_price.replace('"', '')
                                if outcome_price.startswith('['):
                                    prices = json.loads(outcome_price)
                                else:
                                    prices = [float(outcome_price)]
                            else:
                                prices = outcome_price
                            
                            if len(prices) >= 2:
                                if float(prices[0]) > 0.9:
                                    return {"resolved": True, "outcome": "YES", "final_prob": float(prices[0])}
                                elif float(prices[1]) > 0.9:
                                    return {"resolved": True, "outcome": "NO", "final_prob": 1 - float(prices[1])}
                        except Exception as e:
                            pass
                
                # Get current probability for unresolved markets
                try:
                    prob_str = data.get("outcomePrices", "0.5")
                    if isinstance(prob_str, str):
                        prob_str = prob_str.replace('"', '').strip('[]').split(',')[0]
                    current_prob = float(prob_str)
                except:
                    current_prob = 0.5
                    
                return {"resolved": False, "current_prob": current_prob}
    except Exception as e:
        print(f"  Error checking Polymarket {market_id}: {e}")
    return None

async def check_sports_resolution(question: str, source: str) -> Optional[dict]:
    """Check sports game results"""
    # For sports, we check if the game time has passed
    # In real implementation, would check ESPN API for final scores
    return None  # Will implement with actual game checking

async def check_and_resolve_trade(trade: tuple) -> Optional[dict]:
    """Check if a single trade has resolved"""
    trade_id, source, market_id, question, direction, entry_price, bet_size, edge, \
    grok_fair, grok_dir, gpt_fair, gpt_dir, hours_left, closes_at, category = trade
    
    result = None
    
    if "MANIFOLD" in source.upper():
        result = await fetch_manifold_resolution(market_id)
    elif "POLYMARKET" in source.upper():
        result = await fetch_polymarket_resolution(market_id)
    elif "ESPN" in source.upper():
        result = await check_sports_resolution(question, source)
    
    return result

def calculate_pnl(direction: str, outcome: str, bet_size: float, entry_price: float) -> float:
    """Calculate profit/loss for a trade"""
    if direction == outcome:
        # Won - get payout minus bet
        if direction == "YES":
            payout = bet_size / entry_price
        else:
            payout = bet_size / (1 - entry_price)
        return payout - bet_size
    else:
        # Lost - lose entire bet
        return -bet_size

def check_llm_correctness(llm_dir: str, outcome: str) -> int:
    """Check if LLM prediction was correct"""
    if not llm_dir:
        return 0
    return 1 if llm_dir.upper() == outcome.upper() else 0

async def resolve_trades():
    """Main resolution checking loop"""
    db = get_db()
    cursor = db.cursor()
    
    # Get all unresolved trades
    cursor.execute("""
        SELECT trade_id, source, market_id, question, direction, entry_price, bet_size, edge,
               grok_fair, grok_dir, gpt_fair, gpt_dir, hours_left, closes_at, category
        FROM trades 
        WHERE resolved_at IS NULL
        ORDER BY hours_left ASC
    """)
    trades = cursor.fetchall()
    
    print(f"\n{'='*60}")
    print(f"🔍 RESOLUTION CHECKER - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")
    print(f"Checking {len(trades)} open trades...\n")
    
    resolved_count = 0
    total_pnl = 0
    
    for trade in trades:
        trade_id = trade[0]
        source = trade[1]
        market_id = trade[2]
        question = trade[3][:50]
        direction = trade[4]
        entry_price = trade[5]
        bet_size = trade[6]
        grok_dir = trade[9]
        gpt_dir = trade[11]
        category = trade[14]
        
        result = await check_and_resolve_trade(trade)
        
        if result and result.get("resolved"):
            outcome = result["outcome"]
            pnl = calculate_pnl(direction, outcome, bet_size, entry_price)
            grok_correct = check_llm_correctness(grok_dir, outcome)
            gpt_correct = check_llm_correctness(gpt_dir, outcome)
            
            # Update trade
            cursor.execute("""
                UPDATE trades SET
                    resolved_at = ?,
                    outcome = ?,
                    actual_price = ?,
                    pnl = ?,
                    grok_correct = ?,
                    gpt_correct = ?
                WHERE trade_id = ?
            """, (
                datetime.now(timezone.utc).isoformat(),
                outcome,
                result.get("final_prob", 0.5),
                pnl,
                grok_correct,
                gpt_correct,
                trade_id
            ))
            
            resolved_count += 1
            total_pnl += pnl
            
            win_lose = "✅ WIN" if pnl > 0 else "❌ LOSS"
            print(f"{win_lose} | ${pnl:+.2f} | {category}")
            print(f"  Q: {question}...")
            print(f"  Bet: ${bet_size} {direction} | Outcome: {outcome}")
            print(f"  Grok: {grok_dir} {'✓' if grok_correct else '✗'} | GPT: {gpt_dir} {'✓' if gpt_correct else '✗'}")
            print()
    
    db.commit()
    
    if resolved_count > 0:
        print(f"\n{'='*60}")
        print(f"📊 SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Resolved: {resolved_count} trades")
        print(f"Session P&L: ${total_pnl:+.2f}")
    
    # Show overall stats
    await show_performance_stats(cursor)
    
    db.close()
    return resolved_count

async def show_performance_stats(cursor):
    """Display LLM performance by category"""
    
    # Overall stats
    cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(pnl) as total_pnl,
            SUM(grok_correct) as grok_wins,
            SUM(gpt_correct) as gpt_wins
        FROM trades WHERE resolved_at IS NOT NULL
    """)
    overall = cursor.fetchone()
    
    if overall[0] > 0:
        print(f"\n{'='*60}")
        print(f"📈 OVERALL PERFORMANCE")
        print(f"{'='*60}")
        print(f"Total Resolved: {overall[0]} trades")
        print(f"Win Rate: {overall[1]}/{overall[0]} ({100*overall[1]/overall[0]:.1f}%)")
        print(f"Total P&L: ${overall[2]:+.2f}")
        print(f"\n🤖 LLM ACCURACY:")
        print(f"  Grok: {overall[3]}/{overall[0]} ({100*overall[3]/overall[0]:.1f}%)")
        print(f"  GPT:  {overall[4]}/{overall[0]} ({100*overall[4]/overall[0]:.1f}%)")
        
        # By category
        cursor.execute("""
            SELECT 
                category,
                COUNT(*) as total,
                SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
                SUM(pnl) as total_pnl,
                SUM(grok_correct) as grok_wins,
                SUM(gpt_correct) as gpt_wins
            FROM trades 
            WHERE resolved_at IS NOT NULL
            GROUP BY category
            ORDER BY total DESC
        """)
        categories = cursor.fetchall()
        
        if categories:
            print(f"\n{'='*60}")
            print(f"📊 PERFORMANCE BY CATEGORY")
            print(f"{'='*60}")
            print(f"{'Category':<15} {'Trades':<8} {'Win%':<8} {'P&L':<10} {'Grok%':<8} {'GPT%':<8}")
            print("-" * 60)
            
            for cat in categories:
                cat_name, total, wins, pnl, grok, gpt = cat
                win_pct = 100 * wins / total if total > 0 else 0
                grok_pct = 100 * grok / total if total > 0 else 0
                gpt_pct = 100 * gpt / total if total > 0 else 0
                
                print(f"{cat_name:<15} {total:<8} {win_pct:<7.1f}% ${pnl:<9.2f} {grok_pct:<7.1f}% {gpt_pct:<7.1f}%")
            
            # Find which LLM is better per category
            print(f"\n🏆 LLM CATEGORY STRENGTHS:")
            for cat in categories:
                cat_name, total, wins, pnl, grok, gpt = cat
                if total >= 3:  # Only show if enough data
                    if grok > gpt:
                        print(f"  {cat_name}: Grok leads ({grok}/{total} vs {gpt}/{total})")
                    elif gpt > grok:
                        print(f"  {cat_name}: GPT leads ({gpt}/{total} vs {grok}/{total})")
                    else:
                        print(f"  {cat_name}: Tied ({grok}/{total})")
    else:
        print("\n⏳ No resolved trades yet. Waiting for markets to close...")

async def continuous_check(interval_minutes: int = 5):
    """Run resolution checker continuously"""
    print(f"🚀 Starting continuous resolution checker (every {interval_minutes} min)")
    print(f"Press Ctrl+C to stop\n")
    
    while True:
        try:
            resolved = await resolve_trades()
            print(f"\n⏰ Next check in {interval_minutes} minutes...")
            await asyncio.sleep(interval_minutes * 60)
        except KeyboardInterrupt:
            print("\n\n👋 Resolution checker stopped.")
            break
        except Exception as e:
            print(f"Error: {e}")
            await asyncio.sleep(60)

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--continuous":
        interval = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        asyncio.run(continuous_check(interval))
    else:
        # Single check
        asyncio.run(resolve_trades())
