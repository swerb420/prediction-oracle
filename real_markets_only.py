#!/usr/bin/env python3
"""
Real Markets Only - Paper trading on REAL prediction markets
Sources: Polymarket (crypto), ESPN Sports (real games)
NO: Manifold (play money), Weather (no market)
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

# Category detection
def categorize(question: str) -> str:
    q = question.lower()
    
    if any(x in q for x in ['vs', 'nba', 'nfl', 'nhl', 'mlb', 'ufc', 'boxing', 'game', 'beat', 'win against']):
        return 'SPORTS'
    if any(x in q for x in ['ukraine', 'russia', 'putin', 'zelensky', 'ceasefire', 'war', 'nato', 'israel', 'gaza', 
                            'venezuela', 'maduro', 'china', 'taiwan', 'military', 'invasion', 'peace deal']):
        return 'GEOPOLITICS'
    if any(x in q for x in ['bitcoin', 'btc', 'eth', 'ethereum', 'crypto', 'solana', 'doge', 'xrp']):
        return 'CRYPTO'
    if any(x in q for x in ['fed', 'interest rate', 'gdp', 'inflation', 'fomc', 'cpi', 'rate cut', 'powell']):
        return 'ECONOMICS'
    if any(x in q for x in ['trump', 'biden', 'election', 'president', 'congress', 'senate', 'epstein', 'pardon']):
        return 'POLITICS'
    if any(x in q for x in ['stock', 's&p', 'nasdaq', 'nvidia', 'apple', 'tesla', 'google', 'amazon', 'earnings']):
        return 'STOCKS'
    if any(x in q for x in ['ai', 'gpt', 'claude', 'openai', 'anthropic', 'gemini', 'artificial intelligence']):
        return 'AI_TECH'
    if any(x in q for x in ['elon', 'musk', 'tweet', 'celebrity', 'movie', 'oscar', 'grammy']):
        return 'ENTERTAINMENT'
    return 'OTHER'


async def fetch_polymarket_all(max_hours=720):
    """Fetch ALL Polymarket markets - these are REAL money markets"""
    markets = []
    
    async with httpx.AsyncClient(timeout=30) as client:
        try:
            resp = await client.get(
                "https://gamma-api.polymarket.com/markets",
                params={"closed": "false", "active": "true", "limit": 500}
            )
            if resp.status_code != 200:
                print(f"   Polymarket API error: {resp.status_code}")
                return []
            
            data = resp.json()
            now = datetime.now(timezone.utc)
            cutoff = now + timedelta(hours=max_hours)
            
            for m in data:
                end_str = m.get('endDate') or m.get('end_date_iso')
                if not end_str:
                    continue
                
                try:
                    end = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                except:
                    continue
                
                if end > cutoff or end < now:
                    continue
                
                prices_str = m.get('outcomePrices', '')
                if not prices_str:
                    continue
                
                try:
                    prices = [float(p.strip('" ')) for p in prices_str.strip('[]').split(',') if p.strip()]
                    price = prices[0] if prices else 0.5
                except:
                    continue
                
                # Skip extreme prices
                if price < 0.03 or price > 0.97:
                    continue
                
                hours_left = (end - now).total_seconds() / 3600
                volume = float(m.get('volume24hr', 0) or 0)
                
                markets.append({
                    'id': m.get('id'),
                    'source': 'POLYMARKET',
                    'question': m.get('question', 'Unknown'),
                    'price': price,
                    'volume': volume,
                    'hours_left': hours_left,
                    'closes_at': end.isoformat(),
                    'category': categorize(m.get('question', '')),
                    'slug': m.get('slug', ''),
                })
            
        except Exception as e:
            print(f"   Polymarket error: {e}")
    
    return sorted(markets, key=lambda x: x['hours_left'])


async def fetch_espn_sports(max_hours=72):
    """Fetch real sports games from ESPN - these resolve automatically!"""
    markets = []
    
    sports = [
        ("basketball", "nba", "NBA"),
        ("football", "nfl", "NFL"),
        ("hockey", "nhl", "NHL"),
    ]
    
    async with httpx.AsyncClient(timeout=15) as client:
        for sport, league, league_name in sports:
            try:
                resp = await client.get(
                    f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/scoreboard"
                )
                if resp.status_code != 200:
                    continue
                
                data = resp.json()
                events = data.get('events', [])
                now = datetime.now(timezone.utc)
                cutoff = now + timedelta(hours=max_hours)
                
                for event in events:
                    date_str = event.get('date')
                    if not date_str:
                        continue
                    
                    try:
                        game_time = datetime.fromisoformat(date_str.replace('Z', '+00:00'))
                    except:
                        continue
                    
                    # Only future games
                    if game_time > cutoff or game_time < now:
                        continue
                    
                    comps = event.get('competitions', [{}])[0].get('competitors', [])
                    if len(comps) < 2:
                        continue
                    
                    home = next((c for c in comps if c.get('homeAway') == 'home'), comps[0])
                    away = next((c for c in comps if c.get('homeAway') == 'away'), comps[1])
                    
                    home_name = home.get('team', {}).get('shortDisplayName', 'Home')
                    away_name = away.get('team', {}).get('shortDisplayName', 'Away')
                    
                    # Get betting odds
                    odds_data = event.get('competitions', [{}])[0].get('odds', [{}])
                    if not odds_data:
                        continue
                    
                    odds = odds_data[0] if odds_data else {}
                    home_ml = odds.get('homeTeamOdds', {}).get('moneyLine', 0)
                    
                    # Convert American odds to probability
                    def ml_to_prob(ml):
                        if not ml:
                            return 0.5
                        if ml > 0:
                            return 100 / (ml + 100)
                        else:
                            return abs(ml) / (abs(ml) + 100)
                    
                    home_prob = ml_to_prob(home_ml)
                    hours_left = (game_time - now).total_seconds() / 3600
                    
                    markets.append({
                        'id': f"ESPN-{league}-{event.get('id')}-HOME",
                        'source': f'ESPN_{league_name}',
                        'question': f"Will {home_name} beat {away_name}? ({league_name})",
                        'price': home_prob,
                        'volume': 0,
                        'hours_left': hours_left,
                        'closes_at': game_time.isoformat(),
                        'category': 'SPORTS',
                        'home_team': home_name,
                        'away_team': away_name,
                    })
                    
            except Exception as e:
                print(f"   ESPN {league_name} error: {e}")
    
    return sorted(markets, key=lambda x: x['hours_left'])


def remove_fake_markets():
    """Remove weather and manifold trades - keeping only real markets"""
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    
    # Count before
    cursor.execute("SELECT source, COUNT(*) FROM trades WHERE outcome IS NULL GROUP BY source")
    before = dict(cursor.fetchall())
    
    # Delete weather trades (no real market)
    cursor.execute("DELETE FROM trades WHERE source = 'WEATHER' AND outcome IS NULL")
    weather_deleted = cursor.rowcount
    
    # Keep Manifold for comparison but mark them
    # Actually let's keep them to compare LLM performance
    
    db.commit()
    
    print(f"🗑️  Removed {weather_deleted} WEATHER trades (no real market)")
    print(f"\n📊 Remaining by source:")
    
    cursor.execute("SELECT source, COUNT(*), SUM(bet_size) FROM trades WHERE outcome IS NULL GROUP BY source")
    for row in cursor.fetchall():
        real = "✅ REAL" if row[0] in ['POLYMARKET', 'ESPN_NFL', 'ESPN_NBA', 'ESPN_NHL'] else "📝 Paper"
        print(f"   {row[0]}: {row[1]} trades, ${row[2]:.0f} | {real}")
    
    db.close()
    return weather_deleted


async def main():
    print("🎯 REAL MARKETS SCANNER")
    print("=" * 70)
    print("Sources: Polymarket (REAL $), ESPN Sports (REAL games)")
    print("Removed: Weather, keeping Manifold for LLM comparison")
    print("=" * 70)
    
    # First, clean up fake markets
    print("\n🧹 CLEANING UP FAKE MARKETS...")
    removed = remove_fake_markets()
    
    # Fetch real markets
    print("\n📡 FETCHING REAL MARKETS...")
    
    poly, espn = await asyncio.gather(
        fetch_polymarket_all(720),  # 30 days
        fetch_espn_sports(72)  # 3 days for sports
    )
    
    print(f"   Polymarket: {len(poly)} markets (REAL MONEY)")
    print(f"   ESPN Sports: {len(espn)} games (REAL GAMES)")
    
    # Show by category
    all_markets = poly + espn
    
    by_cat = {}
    for m in all_markets:
        cat = m['category']
        if cat not in by_cat:
            by_cat[cat] = []
        by_cat[cat].append(m)
    
    print(f"\n📊 REAL MARKETS BY CATEGORY:")
    print("-" * 70)
    for cat, items in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        poly_count = len([m for m in items if m['source'] == 'POLYMARKET'])
        espn_count = len([m for m in items if 'ESPN' in m['source']])
        print(f"   {cat:15}: {len(items):3} total | Poly: {poly_count}, ESPN: {espn_count}")
    
    # Show interesting markets
    print(f"\n🔥 TOP REAL MARKETS (closing soon):")
    print("-" * 70)
    
    for m in sorted(all_markets, key=lambda x: x['hours_left'])[:20]:
        hrs = m['hours_left']
        time_str = f"{hrs:.0f}h" if hrs < 48 else f"{hrs/24:.0f}d"
        src = "POLY" if m['source'] == 'POLYMARKET' else m['source'].replace('ESPN_', '')
        
        print(f"\n[{src:4}][{m['category'][:6]}] {m['question'][:55]}")
        print(f"  → {m['price']:.0%} YES | Closes: {time_str}")
        if m['source'] == 'POLYMARKET':
            print(f"  → https://polymarket.com/event/{m.get('slug', m['id'])}")
    
    # Summary of what you can ACTUALLY bet on
    print("\n" + "=" * 70)
    print("💰 REAL BETTING OPTIONS:")
    print("=" * 70)
    print(f"""
    POLYMARKET ({len(poly)} markets):
    ├─ Real USDC bets on Polygon
    ├─ Min bet: ~$1
    ├─ Categories: Crypto, Politics, Geopolitics, Economics
    └─ Access: VPN needed for US
    
    ESPN SPORTS ({len(espn)} games):
    ├─ Real games, real odds
    ├─ Bet via: DraftKings, FanDuel, Bovada, etc
    ├─ Or DEX: Azuro frontends (bookmaker.xyz, etc)
    └─ Auto-resolves when game ends!
    
    MANIFOLD (paper only):
    ├─ Play money, but useful for LLM testing
    ├─ Good variety of questions
    └─ Kept for comparison
    """)
    
    # Show current portfolio breakdown
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    
    cursor.execute("""
        SELECT 
            CASE 
                WHEN source LIKE 'ESPN%' OR source = 'POLYMARKET' THEN 'REAL'
                ELSE 'PAPER'
            END as market_type,
            COUNT(*),
            SUM(bet_size)
        FROM trades 
        WHERE outcome IS NULL
        GROUP BY market_type
    """)
    
    print("\n📂 YOUR PORTFOLIO:")
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]} trades, ${row[2]:.0f}")
    
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
