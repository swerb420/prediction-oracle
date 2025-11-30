#!/usr/bin/env python3
"""
Add New Categories - Place paper bets on Kalshi + more Polymarket categories
Focus: GEOPOLITICS, CRYPTO price targets, ECONOMICS (Fed), STOCKS
"""
import asyncio
import sqlite3
import httpx
import json
from datetime import datetime, timezone, timedelta
import os
import sys

sys.path.insert(0, '/root/prediction_oracle')
os.chdir('/root/prediction_oracle')

from multi_scanner import categorize, analyze_with_llm

DB_PATH = 'paper_trades.db'

# BET SIZING
def calculate_bet_size(edge: float, confidence: str, hours_left: float) -> float:
    """Dynamic bet sizing based on edge and confidence"""
    edge_pct = edge * 100
    
    if confidence == 'HIGH':
        if edge_pct >= 15 and hours_left <= 72:
            return 100.0
        elif edge_pct >= 10:
            return 50.0
        elif edge_pct >= 5:
            return 25.0
        else:
            return 10.0
    elif confidence == 'MEDIUM':
        if edge_pct >= 20:
            return 25.0
        elif edge_pct >= 10:
            return 15.0
        else:
            return 10.0
    else:
        if edge_pct >= 30:
            return 10.0
        else:
            return 5.0


async def fetch_kalshi_markets(max_hours=168):
    """Fetch Kalshi markets"""
    markets = []
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(hours=max_hours)
            
            cursor = None
            for _ in range(5):
                params = {"limit": 100, "status": "open"}
                if cursor:
                    params["cursor"] = cursor
                
                resp = await client.get(
                    "https://api.elections.kalshi.com/trade-api/v2/markets",
                    params=params
                )
                
                if resp.status_code != 200:
                    print(f"   Kalshi API error: {resp.status_code}")
                    break
                
                data = resp.json()
                
                for m in data.get("markets", []):
                    close_str = m.get("close_time") or m.get("expected_expiration_time")
                    if not close_str:
                        continue
                    
                    try:
                        close = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                    except:
                        continue
                    
                    if close > cutoff or close < now:
                        continue
                    
                    price = m.get("last_price", 50) / 100
                    if price < 0.05 or price > 0.95:
                        continue
                    
                    hours_left = (close - now).total_seconds() / 3600
                    
                    markets.append({
                        'id': m.get("ticker"),
                        'source': 'KALSHI',
                        'question': m.get("title", "Unknown"),
                        'price': price,
                        'volume': float(m.get("volume_24h", 0) or 0),
                        'hours_left': hours_left,
                        'closes_at': close.isoformat(),
                        'category': categorize(m.get("title", "")),
                    })
                
                cursor = data.get("cursor")
                if not cursor:
                    break
            
        except Exception as e:
            print(f"   Kalshi error: {e}")
    
    return sorted(markets, key=lambda x: x['hours_left'])


async def fetch_polymarket_crypto():
    """Fetch crypto-focused markets from Polymarket"""
    markets = []
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={"closed": "false", "active": "true", "limit": 500}
            )
            if resp.status_code != 200:
                return []
            
            data = resp.json()
            now = datetime.now(timezone.utc)
            
            for m in data:
                question = m.get('question', '')
                category = categorize(question)
                
                # Only interested in specific categories
                if category not in ['CRYPTO', 'GEOPOLITICS', 'ECONOMICS', 'STOCKS']:
                    continue
                
                end_str = m.get('endDate') or m.get('end_date_iso')
                if not end_str:
                    continue
                
                try:
                    end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                except:
                    continue
                
                hours_left = (end - now).total_seconds() / 3600
                if hours_left < 0 or hours_left > 720:  # Max 30 days
                    continue
                
                prices_str = m.get('outcomePrices', '')
                if not prices_str:
                    continue
                
                try:
                    prices = [float(p.strip('" ')) for p in prices_str.strip('[]').split(',') if p.strip()]
                    price = prices[0] if prices else 0.5
                except:
                    continue
                
                if price < 0.05 or price > 0.95:
                    continue
                
                markets.append({
                    'id': m.get('id'),
                    'source': 'POLYMARKET',
                    'question': question,
                    'price': price,
                    'volume': float(m.get('volume24hr', 0) or 0),
                    'hours_left': hours_left,
                    'closes_at': end.isoformat(),
                    'category': category,
                })
            
        except Exception as e:
            print(f"   Polymarket error: {e}")
    
    return sorted(markets, key=lambda x: x['hours_left'])


def place_paper_trade(db, market, direction, confidence, fair_value, reason, grok_data, gpt_data):
    """Place a paper trade"""
    cursor = db.cursor()
    
    trade_id = f"{market['source']}-{market['id']}-{direction}"
    
    # Check if already traded
    cursor.execute("SELECT id FROM trades WHERE trade_id = ?", (trade_id,))
    if cursor.fetchone():
        return False, 0, "Already traded"
    
    if direction == 'YES':
        entry_price = market['price']
    else:
        entry_price = 1 - market['price']
    
    edge = abs(fair_value - market['price'])
    bet_size = calculate_bet_size(edge, confidence, market['hours_left'])
    potential_payout = bet_size / entry_price if entry_price > 0 else 0
    potential_profit = potential_payout - bet_size
    
    now = datetime.now(timezone.utc).isoformat()
    
    cursor.execute('''INSERT INTO trades 
        (trade_id, source, market_id, category, question, direction,
         entry_price, bet_size, potential_payout, potential_profit, edge,
         llm_fair, llm_confidence, llm_reason,
         grok_fair, grok_dir, gpt_fair, gpt_dir,
         hours_left, closes_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (trade_id, market['source'], market['id'], market['category'],
         market['question'], direction,
         entry_price, bet_size, potential_payout, potential_profit, edge,
         fair_value, confidence, reason,
         grok_data.get('fair'), grok_data.get('direction'),
         gpt_data.get('fair'), gpt_data.get('direction'),
         market['hours_left'], market['closes_at'], now))
    
    db.commit()
    return True, bet_size, f"Trade placed: ${bet_size:.0f} on {direction}"


async def main():
    print("🆕 ADD NEW CATEGORIES SCANNER")
    print("=" * 70)
    print("Targets: KALSHI, POLYMARKET (Crypto/Geo/Economics/Stocks)")
    print("=" * 70)
    
    # Fetch markets
    print("\n📡 Fetching markets...")
    
    kalshi, poly_special = await asyncio.gather(
        fetch_kalshi_markets(336),  # 2 weeks
        fetch_polymarket_crypto()
    )
    
    print(f"   Kalshi:          {len(kalshi)} markets")
    print(f"   Poly Special:    {len(poly_special)} markets (Crypto/Geo/Econ/Stocks)")
    
    # Filter by interesting categories
    target_categories = ['GEOPOLITICS', 'CRYPTO', 'ECONOMICS', 'STOCKS']
    
    kalshi_filtered = [m for m in kalshi if m['category'] in target_categories]
    all_markets = kalshi_filtered + poly_special
    
    print(f"\n📊 Interesting markets by category:")
    for cat in target_categories:
        count = len([m for m in all_markets if m['category'] == cat])
        print(f"   {cat}: {count} markets")
    
    # Show some examples
    print(f"\n🔥 TOP OPPORTUNITIES:")
    print("-" * 70)
    
    # Sort by hours_left (most urgent first) and show top 20
    sorted_markets = sorted(all_markets, key=lambda x: x['hours_left'])[:30]
    
    for m in sorted_markets:
        hrs = m['hours_left']
        time_str = f"{hrs:.0f}h" if hrs < 48 else f"{hrs/24:.0f}d"
        src = m['source'][:5]
        
        print(f"\n[{src}][{m['category'][:6]}] {m['question'][:55]}")
        print(f"  → Price: {m['price']:.0%} YES | Closes: {time_str}")
    
    # Analyze with LLMs and place bets
    print("\n" + "=" * 70)
    print("🧠 ANALYZING WITH LLMs...")
    print("=" * 70)
    
    # Just analyze top 10 for now
    to_analyze = sorted_markets[:10]
    
    if not to_analyze:
        print("No markets to analyze!")
        return
    
    analyzed = await analyze_with_llm(to_analyze)
    
    # Open database
    db = sqlite3.connect(DB_PATH)
    
    trades_placed = 0
    total_bet = 0
    
    print("\n🎯 TRADE DECISIONS:")
    print("-" * 70)
    
    for m in analyzed:
        if m.get('llm_direction') not in ['YES', 'NO']:
            continue
        
        edge = m.get('edge', 0)
        if edge < 0.05:  # Skip low edge
            continue
        
        direction = m['llm_direction']
        confidence = m.get('llm_confidence', 'MEDIUM')
        fair_value = m.get('llm_fair', m['price'])
        reason = m.get('llm_reason', 'LLM analysis')
        
        grok = m.get('grok', {}) or {}
        gpt = m.get('gpt', {}) or {}
        
        success, bet_size, msg = place_paper_trade(
            db, m, direction, confidence, fair_value, reason, grok, gpt
        )
        
        if success:
            trades_placed += 1
            total_bet += bet_size
            
            win_emoji = "🟢" if confidence == 'HIGH' else "🟡"
            print(f"\n{win_emoji} [{m['source'][:5]}][{m['category'][:6]}]")
            print(f"   {m['question'][:50]}")
            print(f"   → {direction} @ {m['price']:.0%} | Edge: +{edge:.1%} | ${bet_size:.0f}")
            print(f"   → Grok: {grok.get('direction', 'N/A')} | GPT: {gpt.get('direction', 'N/A')}")
            print(f"   → {reason[:60]}")
    
    db.close()
    
    print("\n" + "=" * 70)
    print(f"✅ PLACED {trades_placed} NEW TRADES | Total: ${total_bet:.0f}")
    print("=" * 70)
    
    # Show current portfolio summary
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    
    cursor.execute("""
        SELECT category, COUNT(*), SUM(bet_size) 
        FROM trades 
        WHERE outcome IS NULL 
        GROUP BY category
    """)
    
    print("\n📂 OPEN POSITIONS BY CATEGORY:")
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]} trades, ${row[2]:.0f} at risk")
    
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
