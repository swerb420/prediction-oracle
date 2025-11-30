#!/usr/bin/env python3
"""
Continuous Resolution Monitor
Checks trades every few minutes and logs results
"""
import sqlite3
import subprocess
import json
import time
from datetime import datetime

DB_PATH = '/root/prediction_oracle/paper_trades.db'
LOG_PATH = '/root/prediction_oracle/resolution_log.txt'

def log(msg):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{timestamp}] {msg}"
    print(line)
    with open(LOG_PATH, 'a') as f:
        f.write(line + '\n')

def check_manifold(market_id):
    """Check Manifold market resolution"""
    try:
        result = subprocess.run(
            ['curl', '-s', f'https://api.manifold.markets/v0/market/{market_id}'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            return {
                'resolved': data.get('isResolved', False),
                'resolution': data.get('resolution'),
                'prob': data.get('probability', 0.5)
            }
    except Exception as e:
        log(f"  Error checking Manifold {market_id}: {e}")
    return None

def check_all_trades():
    """Check all open trades for resolution"""
    db = sqlite3.connect(DB_PATH)
    c = db.cursor()
    
    # Get open Manifold trades
    c.execute("""
        SELECT trade_id, market_id, question, hours_left, direction, bet_size, 
               entry_price, grok_dir, gpt_dir, category, source
        FROM trades 
        WHERE resolved_at IS NULL AND source = 'MANIFOLD'
        ORDER BY hours_left
    """)
    trades = c.fetchall()
    
    resolved_count = 0
    total_pnl = 0
    
    for trade in trades:
        trade_id, market_id, question, hours_left, direction, bet_size, entry_price, grok_dir, gpt_dir, category, source = trade
        
        result = check_manifold(market_id)
        
        if result and result['resolved']:
            resolution = result['resolution']
            
            # Calculate P&L
            if direction == resolution:
                # Win
                if direction == 'YES':
                    pnl = (bet_size / entry_price) - bet_size
                else:
                    pnl = (bet_size / (1 - entry_price)) - bet_size
            else:
                # Loss
                pnl = -bet_size
            
            grok_correct = 1 if grok_dir == resolution else 0
            gpt_correct = 1 if gpt_dir == resolution else 0
            
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
            
            win_emoji = "✅" if pnl > 0 else "❌"
            log(f"{win_emoji} RESOLVED: {question[:50]}...")
            log(f"   Outcome: {resolution} | Our bet: {direction} | P&L: ${pnl:+.2f}")
            log(f"   Grok: {grok_dir} {'✓' if grok_correct else '✗'} | GPT: {gpt_dir} {'✓' if gpt_correct else '✗'}")
            log(f"   Category: {category}")
    
    db.close()
    return resolved_count, total_pnl

def show_stats():
    """Show current performance stats"""
    db = sqlite3.connect(DB_PATH)
    c = db.cursor()
    
    # Overall stats
    c.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(pnl) as total_pnl,
            SUM(grok_correct) as grok_wins,
            SUM(gpt_correct) as gpt_wins
        FROM trades WHERE resolved_at IS NOT NULL
    """)
    stats = c.fetchone()
    
    if stats[0] > 0:
        total, wins, pnl, grok, gpt = stats
        log("=" * 50)
        log(f"📊 PERFORMANCE SUMMARY")
        log(f"   Resolved: {total} | Wins: {wins} ({100*wins/total:.0f}%)")
        log(f"   Total P&L: ${pnl:+.2f}")
        log(f"   Grok: {grok}/{total} ({100*grok/total:.0f}%)")
        log(f"   GPT:  {gpt}/{total} ({100*gpt/total:.0f}%)")
        
        # By category
        c.execute("""
            SELECT category, COUNT(*), SUM(CASE WHEN pnl>0 THEN 1 ELSE 0 END), 
                   SUM(pnl), SUM(grok_correct), SUM(gpt_correct)
            FROM trades WHERE resolved_at IS NOT NULL
            GROUP BY category
        """)
        cats = c.fetchall()
        
        if cats:
            log("\n📈 BY CATEGORY:")
            for cat, tot, w, p, g, gp in cats:
                log(f"   {cat}: {w}/{tot} wins, ${p:+.2f}, Grok {g}/{tot}, GPT {gp}/{tot}")
        log("=" * 50)
    
    # Pending trades
    c.execute("SELECT COUNT(*), SUM(bet_size) FROM trades WHERE resolved_at IS NULL")
    pending = c.fetchone()
    log(f"⏳ Pending: {pending[0]} trades, ${pending[1]:.2f} at risk")
    
    db.close()

def main():
    log("\n🚀 Resolution Monitor Started")
    log("Checking every 5 minutes... Press Ctrl+C to stop\n")
    
    while True:
        try:
            resolved, pnl = check_all_trades()
            
            if resolved > 0:
                log(f"\n📢 {resolved} trades resolved! Session P&L: ${pnl:+.2f}")
                show_stats()
            else:
                # Just show a heartbeat
                db = sqlite3.connect(DB_PATH)
                c = db.cursor()
                c.execute("SELECT COUNT(*), MIN(hours_left) FROM trades WHERE resolved_at IS NULL")
                pending, next_close = c.fetchone()
                db.close()
                log(f"⏳ {pending} open trades. Next closes in {next_close:.1f}h")
            
            time.sleep(300)  # 5 minutes
            
        except KeyboardInterrupt:
            log("\n👋 Monitor stopped")
            show_stats()
            break
        except Exception as e:
            log(f"Error: {e}")
            time.sleep(60)

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--once":
        check_all_trades()
        show_stats()
    else:
        main()
