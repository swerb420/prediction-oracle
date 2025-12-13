#!/usr/bin/env python3
"""
Trade Analytics Dashboard
Analyze paper trades to find winning patterns and optimize thresholds.
"""
import sqlite3
from datetime import datetime
from collections import defaultdict


def get_trades():
    """Get all resolved trades from database."""
    conn = sqlite3.connect('data/polymarket_real.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    c.execute('''
        SELECT * FROM paper_trades 
        WHERE was_correct IS NOT NULL
        ORDER BY id
    ''')
    
    trades = [dict(row) for row in c.fetchall()]
    conn.close()
    return trades


def analyze_by_confidence(trades):
    """Analyze win rate by confidence level."""
    buckets = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})
    
    for t in trades:
        conf = t.get('confidence') or 0
        # Bucket by 5% increments
        bucket = int(conf * 20) * 5  # 50, 55, 60, 65, etc.
        bucket = max(50, min(bucket, 85))
        
        if t['was_correct']:
            buckets[bucket]["wins"] += 1
        else:
            buckets[bucket]["losses"] += 1
        buckets[bucket]["pnl"] += t.get('pnl') or 0
    
    print("\n" + "="*60)
    print("  WIN RATE BY CONFIDENCE LEVEL")
    print("="*60)
    print(f"{'Confidence':>12} | {'Wins':>6} | {'Losses':>6} | {'Rate':>8} | {'P&L':>10}")
    print("-"*60)
    
    for bucket in sorted(buckets.keys()):
        data = buckets[bucket]
        total = data["wins"] + data["losses"]
        rate = data["wins"] / total * 100 if total > 0 else 0
        print(f"{bucket:>10}%+ | {data['wins']:>6} | {data['losses']:>6} | {rate:>7.1f}% | ${data['pnl']:>+9.2f}")


def analyze_by_orderbook(trades):
    """Analyze win rate by orderbook signal strength."""
    buckets = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})
    
    for t in trades:
        ob = abs(t.get('orderbook_signal') or 0)
        # Bucket: 0.0-0.3 (weak), 0.3-0.6 (medium), 0.6-0.8 (strong), 0.8+ (very strong)
        if ob < 0.3:
            bucket = "weak (0-30%)"
        elif ob < 0.6:
            bucket = "medium (30-60%)"
        elif ob < 0.8:
            bucket = "strong (60-80%)"
        else:
            bucket = "very strong (80%+)"
        
        if t['was_correct']:
            buckets[bucket]["wins"] += 1
        else:
            buckets[bucket]["losses"] += 1
        buckets[bucket]["pnl"] += t.get('pnl') or 0
    
    print("\n" + "="*60)
    print("  WIN RATE BY ORDERBOOK SIGNAL STRENGTH")
    print("="*60)
    print(f"{'OB Strength':>20} | {'Wins':>6} | {'Losses':>6} | {'Rate':>8} | {'P&L':>10}")
    print("-"*60)
    
    for bucket in ["weak (0-30%)", "medium (30-60%)", "strong (60-80%)", "very strong (80%+)"]:
        if bucket in buckets:
            data = buckets[bucket]
            total = data["wins"] + data["losses"]
            rate = data["wins"] / total * 100 if total > 0 else 0
            print(f"{bucket:>20} | {data['wins']:>6} | {data['losses']:>6} | {rate:>7.1f}% | ${data['pnl']:>+9.2f}")


def analyze_by_grok(trades):
    """Analyze win rate when Grok was used vs not."""
    grok_used = {"wins": 0, "losses": 0, "pnl": 0}
    grok_agreed = {"wins": 0, "losses": 0, "pnl": 0}
    grok_disagreed = {"wins": 0, "losses": 0, "pnl": 0}
    no_grok = {"wins": 0, "losses": 0, "pnl": 0}
    
    for t in trades:
        pnl = t.get('pnl') or 0
        
        if t.get('grok_used'):
            if t['was_correct']:
                grok_used["wins"] += 1
            else:
                grok_used["losses"] += 1
            grok_used["pnl"] += pnl
            
            if t.get('grok_agreed'):
                if t['was_correct']:
                    grok_agreed["wins"] += 1
                else:
                    grok_agreed["losses"] += 1
                grok_agreed["pnl"] += pnl
            else:
                if t['was_correct']:
                    grok_disagreed["wins"] += 1
                else:
                    grok_disagreed["losses"] += 1
                grok_disagreed["pnl"] += pnl
        else:
            if t['was_correct']:
                no_grok["wins"] += 1
            else:
                no_grok["losses"] += 1
            no_grok["pnl"] += pnl
    
    print("\n" + "="*60)
    print("  WIN RATE BY GROK USAGE")
    print("="*60)
    print(f"{'Grok Status':>20} | {'Wins':>6} | {'Losses':>6} | {'Rate':>8} | {'P&L':>10}")
    print("-"*60)
    
    for name, data in [("No Grok", no_grok), ("Grok Used", grok_used), 
                        ("Grok Agreed", grok_agreed), ("Grok Disagreed", grok_disagreed)]:
        total = data["wins"] + data["losses"]
        if total > 0:
            rate = data["wins"] / total * 100
            print(f"{name:>20} | {data['wins']:>6} | {data['losses']:>6} | {rate:>7.1f}% | ${data['pnl']:>+9.2f}")


def analyze_by_symbol(trades):
    """Analyze win rate by crypto symbol."""
    symbols = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})
    
    for t in trades:
        sym = t['symbol']
        if t['was_correct']:
            symbols[sym]["wins"] += 1
        else:
            symbols[sym]["losses"] += 1
        symbols[sym]["pnl"] += t.get('pnl') or 0
    
    print("\n" + "="*60)
    print("  WIN RATE BY SYMBOL")
    print("="*60)
    print(f"{'Symbol':>10} | {'Wins':>6} | {'Losses':>6} | {'Rate':>8} | {'P&L':>10}")
    print("-"*60)
    
    for sym in ["BTC", "ETH", "SOL", "XRP"]:
        if sym in symbols:
            data = symbols[sym]
            total = data["wins"] + data["losses"]
            rate = data["wins"] / total * 100 if total > 0 else 0
            print(f"{sym:>10} | {data['wins']:>6} | {data['losses']:>6} | {rate:>7.1f}% | ${data['pnl']:>+9.2f}")


def analyze_by_window_timing(trades):
    """Analyze win rate by when in the 15M window we entered."""
    buckets = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0})
    
    for t in trades:
        secs = t.get('secs_into_window') or 0
        if secs < 60:
            bucket = "Early (0-60s)"
        elif secs < 300:
            bucket = "Early-Mid (1-5m)"
        elif secs < 600:
            bucket = "Mid (5-10m)"
        else:
            bucket = "Late (10-15m)"
        
        if t['was_correct']:
            buckets[bucket]["wins"] += 1
        else:
            buckets[bucket]["losses"] += 1
        buckets[bucket]["pnl"] += t.get('pnl') or 0
    
    print("\n" + "="*60)
    print("  WIN RATE BY ENTRY TIMING")
    print("="*60)
    print(f"{'Entry Timing':>20} | {'Wins':>6} | {'Losses':>6} | {'Rate':>8} | {'P&L':>10}")
    print("-"*60)
    
    for bucket in ["Early (0-60s)", "Early-Mid (1-5m)", "Mid (5-10m)", "Late (10-15m)"]:
        if bucket in buckets:
            data = buckets[bucket]
            total = data["wins"] + data["losses"]
            rate = data["wins"] / total * 100 if total > 0 else 0
            print(f"{bucket:>20} | {data['wins']:>6} | {data['losses']:>6} | {rate:>7.1f}% | ${data['pnl']:>+9.2f}")


def print_recommendations(trades):
    """Print tuning recommendations based on data."""
    print("\n" + "="*60)
    print("  RECOMMENDATIONS")
    print("="*60)
    
    # Find best confidence threshold
    conf_wins = defaultdict(int)
    conf_total = defaultdict(int)
    
    for t in trades:
        conf = int((t.get('confidence') or 0) * 100)
        conf_total[conf] += 1
        if t['was_correct']:
            conf_wins[conf] += 1
    
    # Find optimal OB threshold
    ob_wins = defaultdict(int)
    ob_total = defaultdict(int)
    
    for t in trades:
        ob = int(abs(t.get('orderbook_signal') or 0) * 10) / 10
        ob_total[ob] += 1
        if t['was_correct']:
            ob_wins[ob] += 1
    
    print("""
  Based on your data, consider:
  
  1. CONFIDENCE THRESHOLD: Check which confidence levels have best win rate
     - Only trade above the level with 60%+ win rate
     
  2. ORDERBOOK SIGNAL: Strong OB signals (>0.8) seem to predict well
     - Consider requiring OB > 0.7 for trades
     
  3. GROK VALIDATION: Track if Grok-validated trades win more
     - If yes, always require Grok agreement
     - If no, Grok might be adding noise
     
  4. TIMING: Check if early-window entries perform better
     - Markets may be more predictable early
     
  Run this daily to track improvements!
""")


def main():
    print("\n" + "#"*60)
    print("#" + " "*20 + "TRADE ANALYTICS" + " "*23 + "#")
    print("#"*60)
    
    trades = get_trades()
    
    if not trades:
        print("\n  No resolved trades yet. Run resolve_trades.py first!")
        return
    
    # Summary
    wins = sum(1 for t in trades if t['was_correct'])
    losses = len(trades) - wins
    total_pnl = sum(t.get('pnl') or 0 for t in trades)
    
    print(f"\n  Total Resolved Trades: {len(trades)}")
    print(f"  Wins: {wins} | Losses: {losses}")
    print(f"  Win Rate: {wins/len(trades)*100:.1f}%")
    print(f"  Total P&L: ${total_pnl:+.2f}")
    
    # Detailed analysis
    analyze_by_confidence(trades)
    analyze_by_orderbook(trades)
    analyze_by_grok(trades)
    analyze_by_symbol(trades)
    analyze_by_window_timing(trades)
    print_recommendations(trades)


if __name__ == "__main__":
    main()
