#!/usr/bin/env python3
"""
Trade Report Generator
Shows complete P&L history from all sources (DB + JSON logs)
"""
import sqlite3
import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict

def load_json_trades():
    """Load all trades from JSON log files."""
    trades = []
    log_dir = Path("logs")
    
    for log_file in sorted(log_dir.glob("trading_*.jsonl")):
        with open(log_file) as f:
            for line in f:
                try:
                    d = json.loads(line)
                    if d.get('category') == 'trade':
                        trades.append(d)
                except:
                    pass
    
    return trades


def load_db_trades():
    """Load all trades from SQLite database."""
    conn = sqlite3.connect('data/polymarket_real.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        SELECT id, symbol, timestamp, direction, entry_price, size_usd, 
               confidence, grok_used, grok_agreed, actual_outcome, pnl, 
               was_correct, closed_at
        FROM paper_trades 
        ORDER BY id
    ''')
    
    trades = [dict(row) for row in c.fetchall()]
    conn.close()
    return trades


def print_report():
    """Generate and print comprehensive trade report."""
    print("=" * 80)
    print("                    TRADE REPORT - ALL HISTORY")
    print("=" * 80)
    print()
    
    # Load from DB
    db_trades = load_db_trades()
    
    # Load from JSON logs
    json_trades = load_json_trades()
    
    # Group JSON by trade_id
    json_by_id = defaultdict(lambda: {'entry': None, 'exit': None})
    for t in json_trades:
        tid = t.get('data', {}).get('trade_id')
        if tid:
            action = t.get('data', {}).get('action', '')
            if 'entry' in action:
                json_by_id[tid]['entry'] = t
            elif 'exit' in action:
                json_by_id[tid]['exit'] = t
    
    # Combine data
    all_trades = {}
    
    for dt in db_trades:
        tid = dt['id']
        all_trades[tid] = {
            'id': tid,
            'symbol': dt['symbol'],
            'timestamp': dt['timestamp'],
            'direction': dt['direction'],
            'entry_price': dt['entry_price'],
            'size_usd': dt['size_usd'],
            'confidence': dt['confidence'] or 0,
            'grok_used': dt['grok_used'],
            'grok_agreed': dt['grok_agreed'],
            'outcome': dt['actual_outcome'],
            'pnl': dt['pnl'] or 0,
            'was_correct': dt['was_correct'],
            'closed_at': dt['closed_at'],
        }
    
    # Add any trades from JSON not in DB
    for tid, data in json_by_id.items():
        if tid not in all_trades and data['entry']:
            e = data['entry']
            ed = e.get('data', {})
            all_trades[tid] = {
                'id': tid,
                'symbol': e.get('symbol', '?'),
                'timestamp': e.get('timestamp', '')[:19],
                'direction': ed.get('direction', '?'),
                'entry_price': ed.get('entry_price', 0),
                'size_usd': ed.get('size_usd', 0),
                'confidence': 0,
                'grok_used': False,
                'grok_agreed': False,
                'outcome': None,
                'pnl': 0,
                'was_correct': None,
                'closed_at': None,
            }
        
        # Update with exit data if available
        if tid in all_trades and data['exit']:
            x = data['exit']
            xd = x.get('data', {})
            if xd.get('pnl'):
                all_trades[tid]['pnl'] = xd['pnl']
                all_trades[tid]['was_correct'] = xd.get('was_correct')
                all_trades[tid]['outcome'] = 'UP' if xd.get('was_correct') == (all_trades[tid]['direction'] == 'UP') else 'DOWN'
    
    # Print trades
    print(f"{'#':>3} | {'Timestamp':19} | {'Sym':3} | {'Dir':4} | {'Entry':5} | {'Size':>8} | {'Conf':>4} | {'Grok':4} | {'Status':8} | {'P&L':>10}")
    print("-" * 95)
    
    total_pnl = 0
    wins = 0
    losses = 0
    open_count = 0
    
    for tid in sorted(all_trades.keys()):
        t = all_trades[tid]
        
        # Status
        if t['was_correct'] is True:
            status = "WIN ✅"
            wins += 1
        elif t['was_correct'] is False:
            status = "LOSS ❌"
            losses += 1
        else:
            status = "OPEN ⏳"
            open_count += 1
        
        # Grok
        grok = ""
        if t['grok_used']:
            grok = "✓" if t['grok_agreed'] else "✗"
        
        # P&L
        pnl = t['pnl'] or 0
        total_pnl += pnl
        pnl_str = f"${pnl:+.2f}" if pnl != 0 else "-"
        
        ts = t['timestamp'][:19] if t['timestamp'] else '-'
        
        print(f"{t['id']:3d} | {ts:19} | {t['symbol']:3} | {t['direction']:4} | {t['entry_price']:.3f} | ${t['size_usd']:7.2f} | {t['confidence']*100:3.0f}% | {grok:4} | {status:8} | {pnl_str:>10}")
    
    # Summary
    print("-" * 95)
    print()
    print("=" * 40)
    print("              SUMMARY")
    print("=" * 40)
    print(f"  Total Trades:    {len(all_trades)}")
    print(f"  Open Positions:  {open_count}")
    print(f"  Closed Trades:   {wins + losses}")
    print(f"  Wins:            {wins}")
    print(f"  Losses:          {losses}")
    if wins + losses > 0:
        winrate = wins / (wins + losses) * 100
        print(f"  Win Rate:        {winrate:.1f}%")
    print(f"  Total P&L:       ${total_pnl:+.2f}")
    print("=" * 40)


if __name__ == "__main__":
    print_report()
