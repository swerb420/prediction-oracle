#!/usr/bin/env python3
"""
Quick Resolution Checker - Check and resolve pending trades
Checks Manifold, Polymarket, and Weather sources
"""

import sqlite3
import subprocess
import json
from datetime import datetime, timezone
from ml_predictor import MLPredictor

DB_PATH = "/root/prediction_oracle/paper_trades.db"
ml_predictor = MLPredictor(model_path=None)  # Set model path if available

def fetch_url(url, timeout=10):
    """Fetch URL using wget"""
    try:
        result = subprocess.run(
            ['wget', '-q', '-O', '-', url],
            capture_output=True,
            text=True,
            timeout=timeout
        )
        if result.returncode == 0 and result.stdout:
            return json.loads(result.stdout)
    except Exception as e:
        pass
    return None

def get_manifold_status(market_id):
    """Get Manifold market status"""
    data = fetch_url(f'https://api.manifold.markets/v0/market/{market_id}')
    if data:
        return {
            'resolved': data.get('isResolved', False),
            'resolution': data.get('resolution'),
            'prob': data.get('probability'),
            'question': data.get('question', '')[:50]
        }
    return None

def get_polymarket_status(market_id):
    """Get Polymarket market status"""
    data = fetch_url(f'https://gamma-api.polymarket.com/markets/{market_id}')
    if data:
        resolved = data.get('resolved', False) or data.get('closed', False)
        outcome = None
        if resolved:
            # Check outcome prices
            prices = data.get('outcomePrices', '')
            if isinstance(prices, str):
                try:
                    prices = prices.replace('"', '').strip('[]').split(',')
                    if len(prices) >= 2:
                        if float(prices[0]) > 0.9:
                            outcome = "YES"
                        elif float(prices[1]) > 0.9:
                            outcome = "NO"
                except:
                    pass
        return {
            'resolved': resolved and outcome is not None,
            'resolution': outcome,
            'prob': data.get('probability'),
            'question': data.get('question', '')[:50]
        }
    return None

def get_kalshi_status(market_id):
    """Get Kalshi market status"""
    data = fetch_url(f'https://api.elections.kalshi.com/trade-api/v2/markets/{market_id}')
    if data and 'market' in data:
        market = data['market']
        status = market.get('status', '')
        result = market.get('result', '')
        
        resolved = status in ['closed', 'settled'] and result in ['yes', 'no']
        outcome = result.upper() if resolved else None
        
        return {
            'resolved': resolved,
            'resolution': outcome,
            'prob': market.get('last_price', 50) / 100,
            'question': market.get('title', '')[:50]
        }
    return None

def get_espn_game_result(market_id):
    """Check ESPN game result - market_id format: ESPN-{league}-{game_id}-HOME"""
    try:
        parts = market_id.split('-')
        if len(parts) < 4:
            return None
        
        league = parts[1].lower()
        game_id = parts[2]
        
        # Map league to sport
        sport_map = {
            'nfl': 'football',
            'nba': 'basketball',
            'nhl': 'hockey',
            'mlb': 'baseball'
        }
        
        sport = sport_map.get(league)
        if not sport:
            return None
        
        url = f"https://site.api.espn.com/apis/site/v2/sports/{sport}/{league}/summary?event={game_id}"
        data = fetch_url(url, timeout=15)
        
        if not data:
            return None
        
        # Check if game is complete
        header = data.get('header', {})
        competitions = header.get('competitions', [{}])
        if not competitions:
            return None
        
        comp = competitions[0]
        status = comp.get('status', {})
        status_type = status.get('type', {})
        
        # Check if game is final
        if status_type.get('completed') or status_type.get('name') == 'STATUS_FINAL':
            competitors = comp.get('competitors', [])
            if len(competitors) < 2:
                return None
            
            home = next((c for c in competitors if c.get('homeAway') == 'home'), competitors[0])
            away = next((c for c in competitors if c.get('homeAway') == 'away'), competitors[1])
            
            home_score = int(home.get('score', 0))
            away_score = int(away.get('score', 0))
            
            # HOME wins if home score > away score
            home_won = home_score > away_score
            
            return {
                'resolved': True,
                'resolution': 'YES' if home_won else 'NO',
                'prob': 1.0 if home_won else 0.0,
                'question': f"{home.get('team', {}).get('shortDisplayName', 'Home')} vs {away.get('team', {}).get('shortDisplayName', 'Away')}",
                'score': f"{home_score}-{away_score}"
            }
        
        return {
            'resolved': False,
            'resolution': None,
            'prob': None,
            'question': None
        }
        
    except Exception as e:
        return None

def calculate_pnl(direction, outcome, bet_size, entry_price):
    """Calculate profit/loss"""
    if direction == outcome:
        if direction == "YES":
            payout = bet_size / entry_price
        else:
            payout = bet_size / (1 - entry_price)
        return payout - bet_size
    return -bet_size

def ml_resolution_analysis(trade, status):
    # Feature engineering (example)
    features = [status.get('prob', 0.5), trade[5], trade[6]]  # entry_price, bet_size
    ml_probs = ml_predictor.predict(features)
    print(f"      ML Model Probabilities: {ml_probs}")

def main():
    db = sqlite3.connect(DB_PATH)
    cursor = db.cursor()
    
    print("\n🔄 QUICK RESOLUTION CHECK")
    print("=" * 70)
    
    # Get ALL pending trades
    cursor.execute("""
        SELECT trade_id, source, market_id, question, direction, entry_price, bet_size, 
               grok_dir, gpt_dir, category
        FROM trades 
        WHERE outcome IS NULL
        ORDER BY closes_at ASC
    """)
    trades = cursor.fetchall()
    
    # Group by source
    by_source = {}
    for t in trades:
        src = t[1] or 'UNKNOWN'
        if src not in by_source:
            by_source[src] = []
        by_source[src].append(t)
    
    print(f"Checking {len(trades)} trades across {len(by_source)} sources...\n")
    
    resolved_count = 0
    total_pnl = 0
    
    for source, source_trades in by_source.items():
        print(f"📡 {source} ({len(source_trades)} trades)")
        
        for trade in source_trades:
            trade_id, src, market_id, question, direction, entry_price, bet_size, grok_dir, gpt_dir, category = trade
            
            status = None
            if "MANIFOLD" in src.upper():
                status = get_manifold_status(market_id)
            elif "POLYMARKET" in src.upper():
                status = get_polymarket_status(market_id)
            elif "KALSHI" in src.upper():
                status = get_kalshi_status(market_id)
            elif "ESPN" in src.upper():
                status = get_espn_game_result(market_id)
            # Skip WEATHER - removed
            
            if status and status['resolved']:
                outcome = status['resolution']
                pnl = calculate_pnl(direction, outcome, bet_size, entry_price)
                
                grok_correct = 1 if grok_dir and grok_dir.upper() == outcome else 0
                gpt_correct = 1 if gpt_dir and gpt_dir.upper() == outcome else 0
                
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
                    status['prob'],
                    pnl,
                    grok_correct,
                    gpt_correct,
                    trade_id
                ))
                
                resolved_count += 1
                total_pnl += pnl
                
                result = "✅ WIN" if pnl > 0 else "❌ LOSS"
                print(f"   {result} | ${pnl:+.2f} | {question[:45]}...")
                print(f"      Bet: ${bet_size} {direction} → {outcome}")
                print(f"      LLM: Grok={grok_dir or 'N/A'}{'✓' if grok_correct else '✗'} | GPT={gpt_dir or 'N/A'}{'✓' if gpt_correct else '✗'}")
                ml_resolution_analysis(trade, status)
    
    db.commit()
    
    print("\n" + "=" * 70)
    if resolved_count > 0:
        print(f"✅ Resolved {resolved_count} trades | Net P&L: ${total_pnl:+.2f}")
    else:
        print("No new resolutions found")
    
    # Show detailed LLM performance
    print("\n" + "=" * 70)
    print("🤖 LLM PERFORMANCE ANALYSIS")
    print("=" * 70)
    
    cursor.execute("""
        SELECT 
            category,
            COUNT(*) as total,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN grok_correct = 1 THEN 1 ELSE 0 END) as grok_wins,
            SUM(CASE WHEN gpt_correct = 1 THEN 1 ELSE 0 END) as gpt_wins,
            SUM(pnl) as total_pnl,
            AVG(edge) as avg_edge
        FROM trades 
        WHERE outcome IS NOT NULL
        GROUP BY category
        ORDER BY total DESC
    """)
    cat_data = cursor.fetchall()
    
    if cat_data:
        print(f"\n{'Category':<15} {'Trades':<8} {'WinRate':<10} {'Grok':<10} {'GPT':<10} {'P&L':<12}")
        print("-" * 70)
        
        total_trades = 0
        total_grok = 0
        total_gpt = 0
        total_wins = 0
        
        for row in cat_data:
            cat, total, wins, grok, gpt, pnl, edge = row
            wins = wins or 0
            grok = grok or 0
            gpt = gpt or 0
            pnl = pnl or 0
            
            win_rate = wins/total*100 if total else 0
            grok_acc = grok/total*100 if total else 0
            gpt_acc = gpt/total*100 if total else 0
            
            total_trades += total
            total_grok += grok
            total_gpt += gpt
            total_wins += wins
            
            pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
            print(f"{cat:<15} {total:<8} {win_rate:>6.1f}%   {grok_acc:>6.1f}%   {gpt_acc:>6.1f}%   {pnl_str:<12}")
        
        print("-" * 70)
        overall_wr = total_wins/total_trades*100 if total_trades else 0
        overall_grok = total_grok/total_trades*100 if total_trades else 0
        overall_gpt = total_gpt/total_trades*100 if total_trades else 0
        print(f"{'OVERALL':<15} {total_trades:<8} {overall_wr:>6.1f}%   {overall_grok:>6.1f}%   {overall_gpt:>6.1f}%")
    
    # Final summary
    cursor.execute("""
        SELECT COUNT(*), SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END), SUM(pnl)
        FROM trades WHERE outcome IS NOT NULL
    """)
    total, wins, pnl = cursor.fetchone()
    wins = wins or 0
    pnl = pnl or 0
    
    print(f"\n📊 TOTAL: {total} resolved | {wins} wins ({wins/total*100:.0f}%) | P&L: ${pnl:+.2f}")
    
    # Open positions summary
    cursor.execute("SELECT COUNT(*), SUM(bet_size) FROM trades WHERE outcome IS NULL")
    open_count, at_risk = cursor.fetchone()
    print(f"📂 OPEN: {open_count} positions | ${at_risk:.0f} at risk")
    
    db.close()

if __name__ == "__main__":
    main()
