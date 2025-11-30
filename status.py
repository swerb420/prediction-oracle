#!/usr/bin/env python3
"""
Status Dashboard - Shows complete portfolio and LLM performance
"""
import sqlite3
from datetime import datetime

DB_PATH = '/root/prediction_oracle/paper_trades.db'

def main():
    db = sqlite3.connect(DB_PATH)
    c = db.cursor()
    
    print("\n" + "=" * 70)
    print("📊 PREDICTION ORACLE STATUS DASHBOARD")
    print(f"🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Open trades summary
    c.execute("""
        SELECT COUNT(*), SUM(bet_size), SUM(potential_profit)
        FROM trades WHERE resolved_at IS NULL
    """)
    open_stats = c.fetchone()
    
    # Resolved trades summary
    c.execute("""
        SELECT COUNT(*), SUM(bet_size), SUM(pnl),
               SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
               SUM(grok_correct), SUM(gpt_correct)
        FROM trades WHERE resolved_at IS NOT NULL
    """)
    resolved_stats = c.fetchone()
    
    print(f"\n💰 PORTFOLIO OVERVIEW")
    print("-" * 40)
    print(f"Open Trades:      {open_stats[0]}")
    print(f"Capital at Risk:  ${open_stats[1] or 0:.2f}")
    print(f"Potential Profit: ${open_stats[2] or 0:.2f}")
    
    if resolved_stats[0] and resolved_stats[0] > 0:
        total, bet_sum, pnl, wins, grok, gpt = resolved_stats
        print(f"\n📈 RESOLVED TRADES")
        print("-" * 40)
        print(f"Resolved Trades: {total}")
        print(f"Win Rate:        {wins}/{total} ({100*wins/total:.1f}%)")
        print(f"Realized P&L:    ${pnl:+.2f}")
        print(f"\n🤖 LLM ACCURACY")
        print("-" * 40)
        print(f"Grok:  {grok}/{total} ({100*grok/total:.1f}%)")
        print(f"GPT:   {gpt}/{total} ({100*gpt/total:.1f}%)")
        
        # By category
        c.execute("""
            SELECT category, COUNT(*), 
                   SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END),
                   SUM(pnl), SUM(grok_correct), SUM(gpt_correct)
            FROM trades WHERE resolved_at IS NOT NULL
            GROUP BY category ORDER BY COUNT(*) DESC
        """)
        cats = c.fetchall()
        
        if cats:
            print(f"\n📊 PERFORMANCE BY CATEGORY")
            print("-" * 70)
            print(f"{'Category':<15} {'Trades':<8} {'Win%':<8} {'P&L':<12} {'Grok%':<10} {'GPT%':<10}")
            print("-" * 70)
            for cat, tot, w, p, g, gp in cats:
                grok_pct = 100*g/tot if tot > 0 else 0
                gpt_pct = 100*gp/tot if tot > 0 else 0
                leader = "🏆Grok" if g > gp else ("🏆GPT" if gp > g else "Tie")
                print(f"{cat:<15} {tot:<8} {100*w/tot:<7.0f}% ${p:<+11.2f} {grok_pct:<9.0f}% {gpt_pct:<9.0f}% {leader}")
    else:
        print(f"\n⏳ No resolved trades yet - waiting for markets to close")
    
    # Upcoming resolutions
    c.execute("""
        SELECT question, hours_left, direction, bet_size, grok_dir, gpt_dir, category
        FROM trades WHERE resolved_at IS NULL
        ORDER BY hours_left LIMIT 10
    """)
    upcoming = c.fetchall()
    
    print(f"\n⏰ NEXT TO RESOLVE")
    print("-" * 70)
    for q, h, d, b, grok, gpt, cat in upcoming:
        consensus = "✓" if grok == gpt == d else "⚡"
        print(f"{h:>5.1f}h | ${b:>3.0f} {d:<3} | {consensus} | {cat:<10} | {q[:35]}...")
    
    # Show total by bet size
    c.execute("""
        SELECT 
            CASE WHEN bet_size >= 100 THEN '$100' 
                 WHEN bet_size >= 50 THEN '$50'
                 ELSE '$5' END as tier,
            COUNT(*), SUM(bet_size)
        FROM trades WHERE resolved_at IS NULL
        GROUP BY tier ORDER BY SUM(bet_size) DESC
    """)
    tiers = c.fetchall()
    
    print(f"\n💵 BET SIZE BREAKDOWN")
    print("-" * 40)
    for tier, count, total in tiers:
        print(f"{tier} bets: {count} trades = ${total:.0f}")
    
    db.close()
    print("\n" + "=" * 70)

if __name__ == "__main__":
    main()
