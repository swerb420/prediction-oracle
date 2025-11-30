#!/usr/bin/env python3
import sqlite3
import urllib.request
import json

DB_PATH = '/root/prediction_oracle/paper_trades.db'

def check_market(market_id):
    url = f'https://api.manifold.markets/v0/market/{market_id}'
    try:
        with urllib.request.urlopen(url, timeout=10) as response:
            return json.loads(response.read().decode())
    except Exception as e:
        print(f"Error checking {market_id}: {e}")
        return None

def main():
    db = sqlite3.connect(DB_PATH)
    c = db.cursor()
    
    # Get Manifold trades sorted by hours_left
    c.execute("""
        SELECT trade_id, market_id, question, direction, bet_size, 
               grok_dir, gpt_dir, category, hours_left
        FROM trades 
        WHERE source='MANIFOLD' AND resolved_at IS NULL
        ORDER BY hours_left LIMIT 10
    """)
    trades = c.fetchall()
    
    print(f"Checking {len(trades)} Manifold trades...")
    print("=" * 60)
    
    resolved_count = 0
    total_pnl = 0
    
    for trade in trades:
        trade_id, market_id, question, direction, bet_size, grok_dir, gpt_dir, category, hours_left = trade
        
        data = check_market(market_id)
        if not data:
            continue
            
        is_resolved = data.get('isResolved', False)
        resolution = data.get('resolution')
        prob = data.get('probability', 0.5)
        
        if is_resolved and resolution:
            # Calculate P&L
            won = (direction == resolution)
            pnl = bet_size if won else -bet_size
            
            grok_correct = 1 if grok_dir == resolution else 0
            gpt_correct = 1 if gpt_dir == resolution else 0
            
            emoji = "WIN" if won else "LOSS"
            print(f"\n{emoji}: {question[:50]}...")
            print(f"  Our bet: ${bet_size} {direction} | Outcome: {resolution}")
            print(f"  P&L: ${pnl:+.2f}")
            print(f"  Grok: {grok_dir} ({'correct' if grok_correct else 'wrong'})")
            print(f"  GPT: {gpt_dir} ({'correct' if gpt_correct else 'wrong'})")
            
            # Update database
            c.execute("""
                UPDATE trades SET 
                    resolved_at = datetime('now'),
                    outcome = ?,
                    pnl = ?,
                    grok_correct = ?,
                    gpt_correct = ?
                WHERE trade_id = ?
            """, (resolution, pnl, grok_correct, gpt_correct, trade_id))
            db.commit()
            
            resolved_count += 1
            total_pnl += pnl
        else:
            print(f"\nOPEN: {question[:45]}... ({hours_left:.1f}h when placed)")
            print(f"  Current prob: {prob:.1%} | Our bet: ${bet_size} {direction}")
    
    print("\n" + "=" * 60)
    if resolved_count > 0:
        print(f"Resolved {resolved_count} trades | Total P&L: ${total_pnl:+.2f}")
    else:
        print("No trades resolved yet")
    
    # Show stats
    c.execute("SELECT COUNT(*), SUM(pnl) FROM trades WHERE resolved_at IS NOT NULL")
    stats = c.fetchone()
    if stats[0] and stats[0] > 0:
        print(f"\nOverall: {stats[0]} resolved | Total P&L: ${stats[1]:+.2f}")
    
    db.close()

if __name__ == "__main__":
    main()
