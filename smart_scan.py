#!/usr/bin/env python3
"""
Smart Scanner v2 - Efficient short-term trade finder with category tracking
Scans BOTH Polymarket AND Kalshi for opportunities
Optimized for low token usage while finding high-value opportunities
"""
import asyncio
import sqlite3
import httpx
from datetime import datetime, timezone, timedelta
from typing import Optional
import os
import sys

sys.path.insert(0, '/root/prediction_oracle')
os.chdir('/root/prediction_oracle')

# Category detection
def categorize(question: str) -> str:
    q = question.lower()
    if any(x in q for x in ['vs', 'nba', 'nfl', 'nhl', 'mlb', 'game', 'match', 'team', 'win', 'score']):
        return 'SPORTS'
    if any(x in q for x in ['bitcoin', 'btc', 'eth', 'crypto', 'token', 'solana', 'doge']):
        return 'CRYPTO'
    if any(x in q for x in ['fed', 'rate', 'gdp', 'inflation', 'unemployment', 'fomc', 'cpi']):
        return 'ECONOMICS'
    if any(x in q for x in ['movie', 'film', 'oscar', 'box office', 'grammy', 'emmy']):
        return 'ENTERTAINMENT'
    if any(x in q for x in ['trump', 'biden', 'election', 'president', 'congress', 'senate', 'governor']):
        return 'POLITICS'
    if any(x in q for x in ['stock', 's&p', 'nasdaq', 'dow', 'market cap', 'nvidia', 'apple', 'tesla', 'earnings']):
        return 'STOCKS'
    if any(x in q for x in ['weather', 'hurricane', 'temperature', 'rain', 'snow']):
        return 'WEATHER'
    return 'OTHER'


async def fetch_polymarket_short_term(max_days=90):
    """Fetch markets closing within max_days days."""
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            # Get active markets - fetch more to find good ones
            resp = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={"closed": "false", "active": "true", "limit": 500}
            )
            if resp.status_code != 200:
                print(f"API error: {resp.status_code}")
                return []
            
            markets = resp.json()
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(days=max_days)
            
            short_term = []
            for m in markets:
                # Parse end date
                end_str = m.get('endDate') or m.get('end_date_iso')
                if not end_str:
                    continue
                
                try:
                    end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                except:
                    continue
                
                # Filter: closes within range AND in the future
                if end > cutoff:
                    continue
                if end < now:  # Skip expired markets
                    continue
                
                # Get price
                prices_str = m.get('outcomePrices', '')
                if not prices_str:
                    continue
                
                try:
                    prices = [float(p.strip('" ')) for p in prices_str.strip('[]').split(',') if p.strip()]
                    price = prices[0] if prices else 0.5
                except:
                    continue
                
                # Skip extreme prices (< 3% or > 97%)
                if price < 0.03 or price > 0.97:
                    continue
                
                hours_left = (end - now).total_seconds() / 3600
                volume = float(m.get('volume24hr', 0) or 0)
                
                short_term.append({
                    'id': m.get('id'),
                    'source': 'POLYMARKET',
                    'question': m.get('question', 'Unknown'),
                    'price': price,
                    'volume': volume,
                    'hours_left': hours_left,
                    'closes_at': end.isoformat(),
                    'category': categorize(m.get('question', '')),
                })
            
            # Sort by hours left (soonest first), but keep top 30 for analysis
            short_term.sort(key=lambda x: x['hours_left'])
            return short_term[:30]  # Top 30 shortest term
            
        except Exception as e:
            print(f"Polymarket fetch error: {e}")
            return []


async def fetch_kalshi_markets(max_days=30):
    """Fetch real Kalshi markets closing within max_days.
    Note: Kalshi primarily has long-term markets (elections, politics).
    Most markets close years out, so short-term may be empty.
    """
    
    async with httpx.AsyncClient(timeout=60) as client:
        try:
            # Kalshi elections API - get markets endpoint
            resp = await client.get(
                "https://api.elections.kalshi.com/trade-api/v2/markets",
                params={"limit": 500, "status": "open"}
            )
            
            if resp.status_code != 200:
                print(f"   Kalshi markets API error: {resp.status_code}")
                return []
            
            data = resp.json()
            markets = data.get('markets', [])
            
            if not markets:
                print(f"   Kalshi: no markets in response")
                return []
            
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(days=max_days)
            
            short_term = []
            soonest_days = 99999
            
            for m in markets:
                # Parse close time
                close_str = m.get('close_time') or m.get('expiration_time') or m.get('end_date')
                if not close_str:
                    continue
                
                try:
                    close = datetime.fromisoformat(close_str.replace('Z', '+00:00'))
                except:
                    continue
                
                if close < now:
                    continue
                    
                days_left = (close - now).days
                soonest_days = min(soonest_days, days_left)
                
                # Only include if within our timeframe
                if close > cutoff:
                    continue
                
                # Get price (yes_bid is the YES price in cents 0-100)
                yes_bid = m.get('yes_bid', 50)
                if isinstance(yes_bid, str):
                    try:
                        yes_bid = float(yes_bid)
                    except:
                        yes_bid = 50
                
                # Kalshi prices are in cents (0-100), convert to decimal
                price = yes_bid / 100 if yes_bid > 1 else yes_bid
                
                # Skip extreme prices
                if price < 0.03 or price > 0.97:
                    continue
                
                hours_left = (close - now).total_seconds() / 3600
                volume = float(m.get('volume', 0) or m.get('volume_24h', 0) or 0)
                
                # Get title/question
                title = m.get('title') or m.get('ticker', 'Unknown')
                
                short_term.append({
                    'id': m.get('ticker') or m.get('id'),
                    'source': 'KALSHI',
                    'question': title,
                    'price': price,
                    'volume': volume,
                    'hours_left': hours_left,
                    'closes_at': close.isoformat(),
                    'category': categorize(title),
                })
            
            if not short_term and soonest_days < 99999:
                print(f"   Kalshi: no markets within {max_days}d (soonest: {soonest_days}d)")
            
            short_term.sort(key=lambda x: x['hours_left'])
            return short_term[:30]
            
        except Exception as e:
            print(f"   Kalshi fetch error: {e}")
            return []
            return []


async def quick_llm_screen(markets: list, max_calls: int = 5):
    """Quick LLM screening - only analyze top candidates."""
    from prediction_oracle.config import settings
    
    # Check for API keys
    has_grok = settings.xai_api_key and settings.xai_api_key != 'your_xai_key'
    has_gpt = settings.openai_api_key and settings.openai_api_key != 'your_openai_key'
    
    if not has_grok and not has_gpt:
        print("No LLM API keys - using price-based screening only")
        return markets
    
    # Only analyze markets with good volume and mid-range prices
    candidates = [m for m in markets if m['volume'] > 10000 and 0.15 < m['price'] < 0.85][:max_calls]
    
    if not candidates:
        candidates = markets[:max_calls]
    
    analyzed = []
    
    async with httpx.AsyncClient(timeout=60) as client:
        for m in candidates:
            try:
                # Compact prompt to save tokens
                prompt = f"""Market: {m['question']}
Current YES price: {m['price']:.0%}
Closes in: {m['hours_left']:.0f} hours

What's the fair probability? Reply in exactly this format:
FAIR: XX%
DIRECTION: YES/NO/SKIP
CONFIDENCE: HIGH/MEDIUM/LOW
REASON: (max 15 words)"""

                if has_grok:
                    resp = await client.post(
                        "https://api.x.ai/v1/chat/completions",
                        headers={"Authorization": f"Bearer {settings.xai_api_key}"},
                        json={
                            "model": "grok-2-1212",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 100,
                            "temperature": 0.3
                        }
                    )
                else:
                    resp = await client.post(
                        "https://api.openai.com/v1/chat/completions",
                        headers={"Authorization": f"Bearer {settings.openai_api_key}"},
                        json={
                            "model": "gpt-4o-mini",
                            "messages": [{"role": "user", "content": prompt}],
                            "max_tokens": 100,
                            "temperature": 0.3
                        }
                    )
                
                if resp.status_code == 200:
                    text = resp.json()['choices'][0]['message']['content']
                    
                    # Parse response
                    fair = None
                    direction = 'SKIP'
                    confidence = 'LOW'
                    reason = ''
                    
                    for line in text.split('\n'):
                        line = line.strip()
                        if line.startswith('FAIR:'):
                            try:
                                fair = float(line.split(':')[1].strip().replace('%', '')) / 100
                            except:
                                pass
                        elif line.startswith('DIRECTION:'):
                            direction = line.split(':')[1].strip().upper()
                        elif line.startswith('CONFIDENCE:'):
                            confidence = line.split(':')[1].strip().upper()
                        elif line.startswith('REASON:'):
                            reason = line.split(':', 1)[1].strip()
                    
                    m['llm_fair'] = fair
                    m['llm_direction'] = direction
                    m['llm_confidence'] = confidence
                    m['llm_reason'] = reason
                    
                    # Calculate edge
                    if fair and direction != 'SKIP':
                        if direction == 'YES':
                            m['edge'] = fair - m['price']
                        else:
                            m['edge'] = m['price'] - fair
                    else:
                        m['edge'] = 0
                    
                    analyzed.append(m)
                    print(f"  ✓ {m['category']}: {m['question'][:40]}... → {direction}")
                
            except Exception as e:
                print(f"  ✗ LLM error: {e}")
                m['edge'] = 0
                analyzed.append(m)
    
    # Add non-analyzed markets
    for m in markets:
        if m not in analyzed:
            m['edge'] = 0
            m['llm_direction'] = 'SKIP'
            analyzed.append(m)
    
    return analyzed


def save_opportunities(markets: list):
    """Save opportunities to database."""
    db = sqlite3.connect('master_trades.db')
    c = db.cursor()
    
    # Ensure table exists
    c.execute('''CREATE TABLE IF NOT EXISTS opportunities (
        id INTEGER PRIMARY KEY,
        market_id TEXT,
        category TEXT,
        question TEXT,
        price REAL,
        volume REAL,
        hours_left REAL,
        closes_at TEXT,
        llm_fair REAL,
        llm_direction TEXT,
        llm_confidence TEXT,
        llm_reason TEXT,
        edge REAL,
        scanned_at TEXT,
        actioned INTEGER DEFAULT 0
    )''')
    
    now = datetime.now(timezone.utc).isoformat()
    
    for m in markets:
        if m.get('llm_direction') in ['YES', 'NO'] and m.get('edge', 0) > 0.03:
            c.execute('''INSERT INTO opportunities 
                (market_id, category, question, price, volume, hours_left, closes_at,
                 llm_fair, llm_direction, llm_confidence, llm_reason, edge, scanned_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
                (m['id'], m['category'], m['question'], m['price'], m['volume'],
                 m['hours_left'], m['closes_at'], m.get('llm_fair'), m.get('llm_direction'),
                 m.get('llm_confidence'), m.get('llm_reason'), m.get('edge'), now))
    
    db.commit()
    db.close()


def display_results(markets: list):
    """Display scan results."""
    print("\n" + "="*60)
    print("🔍 SHORT-TERM OPPORTUNITIES (Next 7 Days)")
    print("="*60)
    
    # Group by category
    by_cat = {}
    for m in markets:
        cat = m['category']
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(m)
    
    # Show actionable opportunities first
    actionable = [m for m in markets if m.get('llm_direction') in ['YES', 'NO'] and m.get('edge', 0) > 0.03]
    
    if actionable:
        print("\n🎯 ACTIONABLE TRADES:")
        print("-"*60)
        for m in sorted(actionable, key=lambda x: x.get('edge', 0), reverse=True)[:10]:
            hrs = m['hours_left']
            time_str = f"{hrs:.0f}h" if hrs < 48 else f"{hrs/24:.0f}d"
            source = m.get('source', 'UNKNOWN')[:4]  # POLY or KALS
            
            # Calculate $5 bet payouts
            if m['llm_direction'] == 'YES':
                entry = m['price']
            else:
                entry = 1 - m['price']
            
            payout = 5 / entry
            profit = payout - 5
            rr = profit / 5
            
            print(f"\n  [{source}][{m['category']}] {m['question'][:45]}")
            print(f"  → {m['llm_direction']} @ {entry:.0%} | Edge: +{m['edge']:.1%} | {m['llm_confidence']}")
            print(f"  → $5 bet → ${payout:.2f} payout (+${profit:.2f}) | R:R 1:{rr:.1f}")
            print(f"  → Closes: {time_str} | Vol: ${m['volume']:,.0f}")
            print(f"  → {m.get('llm_reason', 'No reason')}")
    
    print("\n📊 BY CATEGORY:")
    print("-"*60)
    for cat, items in sorted(by_cat.items()):
        actionable_count = len([m for m in items if m.get('llm_direction') in ['YES', 'NO']])
        print(f"  {cat}: {len(items)} found, {actionable_count} actionable")
    
    print("\n" + "="*60)


async def main():
    print("🔍 Smart Scanner v2 - Polymarket + Kalshi Trade Finder")
    print("="*60)
    
    max_days = 30  # Look 30 days out
    
    # Fetch from BOTH sources in parallel
    print(f"\n📡 Fetching markets (closing within {max_days} days)...")
    
    poly_task = fetch_polymarket_short_term(max_days=max_days)
    kalshi_task = fetch_kalshi_markets(max_days=max_days)
    
    poly_markets, kalshi_markets = await asyncio.gather(poly_task, kalshi_task)
    
    print(f"   Polymarket: {len(poly_markets)} markets")
    print(f"   Kalshi:     {len(kalshi_markets)} markets")
    
    # Combine all markets
    all_markets = poly_markets + kalshi_markets
    
    # Sort by hours left
    all_markets.sort(key=lambda x: x['hours_left'])
    
    if not all_markets:
        print("\nNo markets found from either source")
        return
    
    print(f"   TOTAL:      {len(all_markets)} markets")
    
    # Quick preview by source
    print("\n⏰ Soonest closing (by source):")
    
    print("\n  POLYMARKET:")
    for m in poly_markets[:3]:
        hrs = m['hours_left']
        time_str = f"{hrs:.0f}h" if hrs < 48 else f"{hrs/24:.0f}d"
        print(f"   [{m['category']}] {time_str}: {m['question'][:40]}...")
    
    print("\n  KALSHI:")
    for m in kalshi_markets[:3]:
        hrs = m['hours_left']
        time_str = f"{hrs:.0f}h" if hrs < 48 else f"{hrs/24:.0f}d"
        print(f"   [{m['category']}] {time_str}: {m['question'][:40]}...")
    
    # LLM screening (limited calls) - pick diverse set
    print("\n🤖 Quick LLM analysis (10 markets max - 5 per source)...")
    
    # Pick 5 from each source for analysis
    poly_candidates = [m for m in poly_markets if 0.15 < m['price'] < 0.85][:5]
    kalshi_candidates = [m for m in kalshi_markets if 0.15 < m['price'] < 0.85][:5]
    candidates = poly_candidates + kalshi_candidates
    
    if not candidates:
        candidates = all_markets[:10]
    
    analyzed = await quick_llm_screen(candidates, max_calls=10)
    
    # Add non-analyzed markets
    for m in all_markets:
        if m not in analyzed:
            m['edge'] = 0
            m['llm_direction'] = 'SKIP'
            analyzed.append(m)
    
    # Save and display
    save_opportunities(analyzed)
    display_results(analyzed)
    
    # Summary by source
    print("\n📈 BY SOURCE:")
    poly_actionable = len([m for m in analyzed if m.get('source') == 'POLYMARKET' and m.get('llm_direction') in ['YES', 'NO']])
    kalshi_actionable = len([m for m in analyzed if m.get('source') == 'KALSHI' and m.get('llm_direction') in ['YES', 'NO']])
    print(f"   Polymarket: {len(poly_markets)} scanned, {poly_actionable} actionable")
    print(f"   Kalshi:     {len(kalshi_markets)} scanned, {kalshi_actionable} actionable")
    
    print("\n✅ Scan complete. Results saved to master_trades.db")


if __name__ == "__main__":
    asyncio.run(main())
