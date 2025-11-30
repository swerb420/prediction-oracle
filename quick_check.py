#!/usr/bin/env python3
"""Quick resolution check script"""
import sqlite3
import subprocess
import json

db = sqlite3.connect('paper_trades.db')
c = db.cursor()

# Get Manifold trades
c.execute("""
    SELECT market_id, question, hours_left, direction, bet_size, grok_dir, gpt_dir, category
    FROM trades 
    WHERE source='MANIFOLD' AND resolved_at IS NULL
    ORDER BY hours_left
    LIMIT 15
""")
trades = c.fetchall()

print(f"Checking {len(trades)} Manifold trades...")
print("=" * 70)

for trade in trades:
    market_id, question, hours_left, direction, bet_size, grok_dir, gpt_dir, category = trade
    
    # Use curl to check market
    result = subprocess.run(
        ['curl', '-s', f'https://api.manifold.markets/v0/market/{market_id}'],
        capture_output=True, text=True
    )
    
    if result.returncode == 0:
        data = json.loads(result.stdout)
        is_resolved = data.get('isResolved', False)
        resolution = data.get('resolution')
        prob = data.get('probability', 0.5)
        
        if is_resolved:
            print(f"\n✅ RESOLVED: {resolution}")
            print(f"   Q: {question[:60]}...")
            print(f"   Bet: ${bet_size} {direction} | Category: {category}")
            print(f"   Grok said: {grok_dir} | GPT said: {gpt_dir}")
            
            # Check who was right
            grok_correct = "✓" if grok_dir == resolution else "✗"
            gpt_correct = "✓" if gpt_dir == resolution else "✗"
            our_correct = "WIN" if direction == resolution else "LOSS"
            
            print(f"   Result: {our_correct} | Grok {grok_correct} | GPT {gpt_correct}")
            
            # Update database
            pnl = bet_size if direction == resolution else -bet_size
            c.execute("""
                UPDATE trades SET 
                    resolved_at = datetime('now'),
                    outcome = ?,
                    pnl = ?,
                    grok_correct = ?,
                    gpt_correct = ?
                WHERE market_id = ?
            """, (resolution, pnl, 1 if grok_dir == resolution else 0, 1 if gpt_dir == resolution else 0, market_id))
            db.commit()
        else:
            print(f"\n⏳ {hours_left:.1f}h left | ${bet_size} {direction} | Prob: {prob:.1%}")
            print(f"   Q: {question[:55]}...")

db.close()
print("\n" + "=" * 70)
print("Check complete!")
