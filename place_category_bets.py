#!/usr/bin/env python3
"""
Place targeted paper bets on Polymarket categories:
- GEOPOLITICS (Ukraine, Venezuela, Iran)
- CRYPTO (Bitcoin price targets)
- ECONOMICS (Fed rates)
- STOCKS (Market cap leaders)
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

DB_PATH = 'paper_trades.db'

# Target markets with our LLM analysis
TARGET_MARKETS = [
    # GEOPOLITICS
    {
        'id': '516719',
        'source': 'POLYMARKET',
        'question': 'Russia x Ukraine ceasefire in 2025?',
        'category': 'GEOPOLITICS',
        'end_date': '2025-12-31T12:00:00Z',
        'our_analysis': {
            'direction': 'NO',
            'fair_value': 0.05,
            'confidence': 'HIGH',
            'reason': 'Both sides entrenched; neither has incentive to stop fighting. Russia wants more territory, Ukraine wont cede without NATO guarantees.'
        }
    },
    {
        'id': '516947',
        'source': 'POLYMARKET', 
        'question': 'Maduro out in 2025?',
        'category': 'GEOPOLITICS',
        'end_date': '2025-12-31T12:00:00Z',
        'our_analysis': {
            'direction': 'NO',
            'fair_value': 0.10,
            'confidence': 'MEDIUM',
            'reason': 'Maduro controls military and has survived multiple challenges. Trump admin may push but regime is resilient.'
        }
    },
    {
        'id': '516720',
        'source': 'POLYMARKET',
        'question': 'Putin out as President of Russia in 2025?',
        'category': 'GEOPOLITICS', 
        'end_date': '2025-12-31T12:00:00Z',
        'our_analysis': {
            'direction': 'NO',
            'fair_value': 0.02,
            'confidence': 'HIGH',
            'reason': 'Putin has consolidated power for 25 years. No credible challenger and security apparatus loyal to him.'
        }
    },
    # ECONOMICS - Fed Rates
    {
        'id': '516725',
        'source': 'POLYMARKET',
        'question': 'Will 2 Fed rate cuts happen in 2025?',
        'category': 'ECONOMICS',
        'end_date': '2025-12-10T12:00:00Z',
        'our_analysis': {
            'direction': 'YES',
            'fair_value': 0.20,
            'confidence': 'MEDIUM',
            'reason': 'Fed has already cut twice this year (Sept + Nov). Dec meeting may be 3rd cut. 2 is floor not ceiling.'
        }
    },
    {
        'id': '516726',
        'source': 'POLYMARKET',
        'question': 'Will 3 Fed rate cuts happen in 2025?',
        'category': 'ECONOMICS',
        'end_date': '2025-12-10T12:00:00Z',
        'our_analysis': {
            'direction': 'YES',
            'fair_value': 0.90,
            'confidence': 'HIGH',
            'reason': 'Fed dot plot shows 3+ cuts expected. Already 2 in, Dec highly likely. Market pricing at 87% for Dec cut.'
        }
    },
    # CRYPTO - Bitcoin
    {
        'id': '516871',
        'source': 'POLYMARKET',
        'question': 'Will Bitcoin dip to $70,000 by December 31, 2025?',
        'category': 'CRYPTO',
        'end_date': '2025-12-31T12:00:00Z',
        'our_analysis': {
            'direction': 'NO',
            'fair_value': 0.08,
            'confidence': 'MEDIUM',
            'reason': 'BTC at ~$95k now. Strong institutional buying (ETFs). Unlikely 25% dip unless major black swan.'
        }
    },
    {
        'id': '516865',
        'source': 'POLYMARKET',
        'question': 'Will Bitcoin reach $130,000 by December 31, 2025?',
        'category': 'CRYPTO',
        'end_date': '2025-12-31T12:00:00Z',
        'our_analysis': {
            'direction': 'NO',
            'fair_value': 0.03,
            'confidence': 'MEDIUM',
            'reason': 'BTC at ~$95k needs 35%+ rally in 33 days. Very aggressive. Market says 4.5% is too low but 10% would be fair.'
        }
    },
    # STOCKS - Market Cap
    {
        'id': '516818',
        'source': 'POLYMARKET',
        'question': 'Will NVIDIA be the largest company in the world by market cap on December 31?',
        'category': 'STOCKS',
        'end_date': '2025-12-31T12:00:00Z',
        'our_analysis': {
            'direction': 'YES',
            'fair_value': 0.70,
            'confidence': 'MEDIUM',
            'reason': 'NVDA currently largest (~$3.4T). Apple close behind. AI demand still growing. Likely holds top spot.'
        }
    },
    {
        'id': '516821',
        'source': 'POLYMARKET',
        'question': 'Will Apple be the largest company in the world by market cap on December 31?',
        'category': 'STOCKS',
        'end_date': '2025-12-31T12:00:00Z',
        'our_analysis': {
            'direction': 'NO',
            'fair_value': 0.15,
            'confidence': 'MEDIUM',
            'reason': 'Apple at ~$3.3T, slightly behind NVDA. iPhone cycle slowing. Would need NVDA crash or Apple AI breakthrough.'
        }
    },
    {
        'id': '516960',
        'source': 'POLYMARKET',
        'question': 'Will Google have the top AI model on December 31?',
        'category': 'AI_TECH',
        'end_date': '2025-12-31T12:00:00Z',
        'our_analysis': {
            'direction': 'NO',
            'fair_value': 0.75,
            'confidence': 'LOW',
            'reason': 'Google Gemini 2.0 strong but OpenAI still leads in perception. Market at 88.5% seems high.'
        }
    },
]


async def fetch_current_prices():
    """Fetch current prices for our target markets"""
    async with httpx.AsyncClient(timeout=30) as client:
        for market in TARGET_MARKETS:
            try:
                resp = await client.get(
                    f"https://gamma-api.polymarket.com/markets/{market['id']}"
                )
                if resp.status_code == 200:
                    data = resp.json()
                    prices_str = data.get('outcomePrices', '')
                    if prices_str:
                        prices = [float(p.strip('" ')) for p in prices_str.strip('[]').split(',') if p.strip()]
                        market['current_price'] = prices[0] if prices else 0.5
                        market['volume'] = float(data.get('volume24hr', 0) or 0)
                        
                        # Calculate hours left
                        end = datetime.fromisoformat(market['end_date'].replace('Z', '+00:00'))
                        now = datetime.now(timezone.utc)
                        market['hours_left'] = (end - now).total_seconds() / 3600
                        market['closes_at'] = end.isoformat()
            except Exception as e:
                print(f"   Error fetching {market['id']}: {e}")
                market['current_price'] = None


def calculate_edge(direction, current_price, fair_value):
    """Calculate betting edge"""
    if direction == 'YES':
        return fair_value - current_price
    else:
        return current_price - fair_value


def calculate_bet_size(edge, confidence, hours_left):
    """Calculate bet size based on edge and confidence"""
    edge_pct = edge * 100
    
    if confidence == 'HIGH':
        if edge_pct >= 15 and hours_left <= 336:  # 2 weeks
            return 50.0
        elif edge_pct >= 10:
            return 25.0
        else:
            return 15.0
    elif confidence == 'MEDIUM':
        if edge_pct >= 15:
            return 20.0
        elif edge_pct >= 10:
            return 15.0
        else:
            return 10.0
    else:
        return 5.0


def place_trade(db, market):
    """Place a paper trade"""
    cursor = db.cursor()
    
    analysis = market['our_analysis']
    direction = analysis['direction']
    
    trade_id = f"{market['source']}-{market['id']}-{direction}"
    
    # Check if already traded
    cursor.execute("SELECT id FROM trades WHERE trade_id = ?", (trade_id,))
    if cursor.fetchone():
        return False, 0, "Already traded"
    
    current_price = market.get('current_price')
    if current_price is None:
        return False, 0, "No price data"
    
    fair_value = analysis['fair_value']
    edge = calculate_edge(direction, current_price, fair_value)
    
    if edge < 0.02:  # Minimum 2% edge
        return False, 0, f"Edge too low ({edge:.1%})"
    
    if direction == 'YES':
        entry_price = current_price
    else:
        entry_price = 1 - current_price
    
    bet_size = calculate_bet_size(edge, analysis['confidence'], market.get('hours_left', 720))
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
         fair_value, analysis['confidence'], analysis['reason'],
         fair_value, direction, fair_value, direction,  # Using same for grok/gpt
         market.get('hours_left', 720), market.get('closes_at', ''), now))
    
    db.commit()
    return True, bet_size, f"Trade placed"


async def main():
    print("🎯 TARGETED CATEGORY BETS")
    print("=" * 70)
    print("Categories: GEOPOLITICS, CRYPTO, ECONOMICS, STOCKS, AI_TECH")
    print("=" * 70)
    
    # Fetch current prices
    print("\n📡 Fetching current market prices...")
    await fetch_current_prices()
    
    # Open database
    db = sqlite3.connect(DB_PATH)
    
    trades_placed = 0
    total_bet = 0
    
    print("\n🎯 PLACING PAPER BETS:")
    print("-" * 70)
    
    for market in TARGET_MARKETS:
        current_price = market.get('current_price')
        if current_price is None:
            print(f"\n❌ [{market['category'][:6]}] {market['question'][:45]}")
            print(f"   → No price data available")
            continue
        
        analysis = market['our_analysis']
        direction = analysis['direction']
        fair_value = analysis['fair_value']
        
        edge = calculate_edge(direction, current_price, fair_value)
        
        success, bet_size, msg = place_trade(db, market)
        
        if success:
            trades_placed += 1
            total_bet += bet_size
            
            hours = market.get('hours_left', 0)
            time_str = f"{hours:.0f}h" if hours < 48 else f"{hours/24:.0f}d"
            
            emoji = "🟢" if analysis['confidence'] == 'HIGH' else "🟡" if analysis['confidence'] == 'MEDIUM' else "⚪"
            
            print(f"\n{emoji} [{market['category'][:6]}] {market['question'][:50]}")
            print(f"   Market: {current_price:.1%} YES | Our Fair: {fair_value:.1%}")
            print(f"   → {direction} @ {current_price:.1%} | Edge: +{edge:.1%} | ${bet_size:.0f}")
            print(f"   → Closes: {time_str} | {analysis['reason'][:55]}")
        else:
            print(f"\n⏭️  [{market['category'][:6]}] {market['question'][:45]}")
            print(f"   → Skipped: {msg}")
    
    db.close()
    
    print("\n" + "=" * 70)
    print(f"✅ PLACED {trades_placed} NEW TRADES | Total: ${total_bet:.0f}")
    print("=" * 70)
    
    # Show portfolio by category
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    
    print("\n📊 PORTFOLIO BY CATEGORY:")
    print("-" * 70)
    
    cursor.execute("""
        SELECT category, 
               COUNT(*) as trades,
               SUM(bet_size) as total_bet,
               SUM(CASE WHEN outcome IS NOT NULL THEN 1 ELSE 0 END) as resolved,
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
               COALESCE(SUM(pnl), 0) as pnl
        FROM trades 
        GROUP BY category
        ORDER BY total_bet DESC
    """)
    
    for row in cursor.fetchall():
        cat, trades, total_bet, resolved, wins, pnl = row
        open_trades = trades - resolved
        win_rate = wins/resolved*100 if resolved else 0
        
        status = f"✅ {wins}W/{resolved-wins}L ({win_rate:.0f}%)" if resolved else f"📂 {open_trades} open"
        pnl_str = f"${pnl:+.0f}" if resolved else ""
        
        print(f"   {cat:<12}: {trades:2} trades, ${total_bet:>5.0f} bet | {status} {pnl_str}")
    
    cursor.execute("SELECT SUM(bet_size), SUM(pnl) FROM trades WHERE outcome IS NULL")
    at_risk, _ = cursor.fetchone()
    
    cursor.execute("SELECT SUM(pnl) FROM trades WHERE outcome IS NOT NULL")
    total_pnl = cursor.fetchone()[0] or 0
    
    print("-" * 70)
    print(f"   TOTAL: ${at_risk:.0f} at risk | Realized P&L: ${total_pnl:+.0f}")
    
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
