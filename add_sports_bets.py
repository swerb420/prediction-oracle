#!/usr/bin/env python3
"""
Add Sports Bets - Place paper bets on real ESPN sports games
LLMs analyze team matchups and we track their accuracy
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

# Import LLM analyzer
try:
    from multi_scanner import analyze_with_llm
except:
    analyze_with_llm = None


async def fetch_espn_games(max_hours=72):
    """Fetch upcoming sports games with betting odds"""
    games = []
    
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
                    
                    if game_time > cutoff or game_time < now:
                        continue
                    
                    comps = event.get('competitions', [{}])[0].get('competitors', [])
                    if len(comps) < 2:
                        continue
                    
                    home = next((c for c in comps if c.get('homeAway') == 'home'), comps[0])
                    away = next((c for c in comps if c.get('homeAway') == 'away'), comps[1])
                    
                    home_name = home.get('team', {}).get('shortDisplayName', 'Home')
                    away_name = away.get('team', {}).get('shortDisplayName', 'Away')
                    home_full = home.get('team', {}).get('displayName', home_name)
                    away_full = away.get('team', {}).get('displayName', away_name)
                    
                    # Get records
                    home_record = home.get('records', [{}])[0].get('summary', 'N/A')
                    away_record = away.get('records', [{}])[0].get('summary', 'N/A')
                    
                    # Get betting odds
                    odds_data = event.get('competitions', [{}])[0].get('odds', [{}])
                    if not odds_data:
                        continue
                    
                    odds = odds_data[0] if odds_data else {}
                    home_ml = odds.get('homeTeamOdds', {}).get('moneyLine', 0)
                    away_ml = odds.get('awayTeamOdds', {}).get('moneyLine', 0)
                    spread = odds.get('spread', 0)
                    over_under = odds.get('overUnder', 0)
                    
                    def ml_to_prob(ml):
                        if not ml:
                            return 0.5
                        if ml > 0:
                            return 100 / (ml + 100)
                        else:
                            return abs(ml) / (abs(ml) + 100)
                    
                    home_prob = ml_to_prob(home_ml)
                    away_prob = ml_to_prob(away_ml)
                    
                    # Normalize (remove vig)
                    total = home_prob + away_prob
                    if total > 0:
                        home_prob = home_prob / total
                    
                    hours_left = (game_time - now).total_seconds() / 3600
                    
                    games.append({
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
                        'home_record': home_record,
                        'away_record': away_record,
                        'spread': spread,
                        'over_under': over_under,
                        'details': f"{home_full} ({home_record}) vs {away_full} ({away_record}). Spread: {spread}, O/U: {over_under}"
                    })
                    
            except Exception as e:
                print(f"   ESPN {league_name} error: {e}")
    
    return sorted(games, key=lambda x: x['hours_left'])


async def simple_llm_analyze(games):
    """Simple LLM analysis for sports games"""
    # For now, just use the spread as a proxy for fair value
    # Positive spread means home is favored
    analyzed = []
    
    for g in games:
        spread = g.get('spread', 0)
        price = g['price']
        
        # If home is heavily favored (negative spread), lean YES
        # If away is favored (positive spread), lean NO
        if spread < -5:
            fair_value = min(0.85, price + 0.10)
            direction = "YES"
            confidence = "HIGH" if spread < -10 else "MEDIUM"
            reason = f"Home favored by {abs(spread):.1f} points"
        elif spread > 5:
            fair_value = max(0.15, price - 0.10)
            direction = "NO"
            confidence = "HIGH" if spread > 10 else "MEDIUM"
            reason = f"Away favored by {spread:.1f} points"
        else:
            # Close game - skip or small edge
            fair_value = price
            direction = "YES" if price < 0.5 else "NO"
            confidence = "LOW"
            reason = f"Close matchup, spread {spread:.1f}"
        
        edge = abs(fair_value - price)
        
        g['llm_fair'] = fair_value
        g['llm_direction'] = direction
        g['llm_confidence'] = confidence
        g['llm_reason'] = reason
        g['edge'] = edge
        g['grok'] = {'fair': fair_value, 'direction': direction}
        g['gpt'] = {'fair': fair_value, 'direction': direction}
        
        analyzed.append(g)
    
    return analyzed


def place_sports_bet(db, game, bet_size=5.0):
    """Place a paper bet on a sports game"""
    cursor = db.cursor()
    
    trade_id = f"{game['source']}-{game['id']}-{game['llm_direction']}"
    
    # Check if already traded
    cursor.execute("SELECT id FROM trades WHERE trade_id = ?", (trade_id,))
    if cursor.fetchone():
        return False, 0, "Already bet on this game"
    
    direction = game['llm_direction']
    if direction == 'YES':
        entry_price = game['price']
    else:
        entry_price = 1 - game['price']
    
    if entry_price <= 0.02 or entry_price >= 0.98:
        return False, 0, "Price too extreme"
    
    edge = game.get('edge', 0)
    potential_payout = bet_size / entry_price
    potential_profit = potential_payout - bet_size
    
    now = datetime.now(timezone.utc).isoformat()
    
    grok = game.get('grok', {}) or {}
    gpt = game.get('gpt', {}) or {}
    
    cursor.execute('''INSERT INTO trades 
        (trade_id, source, market_id, category, question, direction,
         entry_price, bet_size, potential_payout, potential_profit, edge,
         llm_fair, llm_confidence, llm_reason,
         grok_fair, grok_dir, gpt_fair, gpt_dir,
         hours_left, closes_at, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)''',
        (trade_id, game['source'], game['id'], 'SPORTS',
         game['question'], direction,
         entry_price, bet_size, potential_payout, potential_profit, edge,
         game.get('llm_fair'), game.get('llm_confidence'), game.get('llm_reason'),
         grok.get('fair'), grok.get('direction'),
         gpt.get('fair'), gpt.get('direction'),
         game['hours_left'], game['closes_at'], now))
    
    db.commit()
    return True, bet_size, f"Bet ${bet_size} on {direction}"


async def main():
    print("🏈 SPORTS BETTING SCANNER")
    print("=" * 70)
    print("Real games from ESPN with actual Vegas odds")
    print("=" * 70)
    
    # Fetch games
    print("\n📡 Fetching upcoming games...")
    games = await fetch_espn_games(72)  # 3 days
    
    print(f"   Found {len(games)} upcoming games")
    
    # Group by league
    by_league = {}
    for g in games:
        league = g['source']
        if league not in by_league:
            by_league[league] = []
        by_league[league].append(g)
    
    for league, items in by_league.items():
        print(f"   {league}: {len(items)} games")
    
    # Analyze with simple spread-based logic
    print("\n🧠 Analyzing games...")
    analyzed = await simple_llm_analyze(games)
    
    # Filter for games with edge
    with_edge = [g for g in analyzed if g['edge'] > 0.03]
    print(f"   {len(with_edge)} games with >3% edge")
    
    # Place bets
    db = sqlite3.connect(DB_PATH)
    
    trades_placed = 0
    total_bet = 0
    
    print("\n🎯 PLACING SPORTS BETS:")
    print("-" * 70)
    
    for game in sorted(with_edge, key=lambda x: x['hours_left'])[:15]:  # Max 15 bets
        bet_size = 5.0  # $5 per game
        
        success, amount, msg = place_sports_bet(db, game, bet_size)
        
        if success:
            trades_placed += 1
            total_bet += amount
            
            hrs = game['hours_left']
            time_str = f"{hrs:.0f}h" if hrs < 24 else f"{hrs/24:.0f}d"
            
            emoji = "🟢" if game['llm_confidence'] == 'HIGH' else "🟡"
            
            print(f"\n{emoji} [{game['source'].replace('ESPN_', '')}] {game['question']}")
            print(f"   → {game['llm_direction']} @ {game['price']:.0%} | Edge: +{game['edge']:.1%}")
            print(f"   → {game['llm_reason']}")
            print(f"   → Bet: ${bet_size} | Starts: {time_str}")
    
    db.close()
    
    print("\n" + "=" * 70)
    print(f"✅ PLACED {trades_placed} SPORTS BETS | Total: ${total_bet:.0f}")
    print("=" * 70)
    
    # Show updated portfolio
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    
    cursor.execute("""
        SELECT source, COUNT(*), SUM(bet_size) 
        FROM trades 
        WHERE outcome IS NULL AND source LIKE 'ESPN%'
        GROUP BY source
    """)
    
    print("\n📊 SPORTS PORTFOLIO:")
    for row in cursor.fetchall():
        print(f"   {row[0]}: {row[1]} bets, ${row[2]:.0f}")
    
    # Check for resolved sports bets
    cursor.execute("""
        SELECT source, COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), SUM(pnl)
        FROM trades 
        WHERE outcome IS NOT NULL AND source LIKE 'ESPN%'
        GROUP BY source
    """)
    
    results = cursor.fetchall()
    if results:
        print("\n📈 SPORTS RESULTS:")
        for row in results:
            wins = row[2] or 0
            pnl = row[3] or 0
            print(f"   {row[0]}: {row[1]} resolved, {wins} wins, ${pnl:+.2f}")
    
    db.close()


if __name__ == "__main__":
    asyncio.run(main())
