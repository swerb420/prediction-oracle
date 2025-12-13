#!/usr/bin/env python3
"""
Edge Finder - Analyze trades to find high-confidence winning patterns.
Focus on orderbook strength, timing, and spread analysis for compounding gains.
"""
import sqlite3
import pandas as pd
from datetime import datetime
import numpy as np

DB_PATH = 'data/polymarket_real.db'


def load_trades() -> pd.DataFrame:
    """Load all resolved trades with signal data."""
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query('''
        SELECT * FROM paper_trades 
        WHERE was_correct IS NOT NULL
        ORDER BY id
    ''', conn)
    conn.close()
    return df


def analyze_orderbook_edge(df: pd.DataFrame):
    """Find winning patterns based on orderbook strength."""
    print("\n" + "="*70)
    print("📊 ORDERBOOK EDGE ANALYSIS")
    print("="*70)
    
    # Only trades with orderbook data
    ob_trades = df[df['orderbook_signal'].notna()].copy()
    
    if len(ob_trades) == 0:
        print("No trades with orderbook data yet.")
        return
    
    ob_trades['ob_strength'] = ob_trades['orderbook_signal'].abs()
    
    # Bucket by orderbook strength
    thresholds = [0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
    
    print(f"\n{'OB Strength':<15} {'Trades':<8} {'Wins':<8} {'Win%':<10} {'Avg PnL':<12} {'Edge'}")
    print("-"*70)
    
    best_threshold = None
    best_edge = 0
    
    for thresh in thresholds:
        subset = ob_trades[ob_trades['ob_strength'] >= thresh]
        if len(subset) == 0:
            continue
        
        wins = subset['was_correct'].sum()
        total = len(subset)
        win_rate = wins / total * 100
        avg_pnl = subset['pnl'].mean()
        
        # Edge = how much better than 50% baseline
        edge = win_rate - 50
        
        marker = "⭐" if edge > best_edge else ""
        if edge > best_edge:
            best_edge = edge
            best_threshold = thresh
        
        print(f">= {thresh:<12.2f} {total:<8} {wins:<8} {win_rate:<10.1f}% ${avg_pnl:<11.2f} {marker}")
    
    if best_threshold:
        print(f"\n🎯 BEST EDGE: Orderbook >= {best_threshold:.2f} ({best_edge:+.1f}% edge)")


def analyze_spread_edge(df: pd.DataFrame):
    """Find edge in Polymarket spreads (YES/NO pricing)."""
    print("\n" + "="*70)
    print("💰 SPREAD/ENTRY PRICE EDGE")
    print("="*70)
    
    # Entry price analysis - lower entry = better odds
    price_trades = df[df['entry_price'].notna()].copy()
    
    if len(price_trades) == 0:
        print("No entry price data.")
        return
    
    # Bucket by entry price
    bins = [(0.40, 0.45), (0.45, 0.48), (0.48, 0.50), (0.50, 0.52), (0.52, 0.55)]
    
    print(f"\n{'Entry Price':<15} {'Trades':<8} {'Wins':<8} {'Win%':<10} {'Avg PnL':<12} {'Edge'}")
    print("-"*70)
    
    for low, high in bins:
        subset = price_trades[(price_trades['entry_price'] >= low) & (price_trades['entry_price'] < high)]
        if len(subset) == 0:
            continue
        
        wins = subset['was_correct'].sum()
        total = len(subset)
        win_rate = wins / total * 100
        avg_pnl = subset['pnl'].mean()
        edge = win_rate - 50
        
        print(f"{low:.2f}-{high:.2f}       {total:<8} {wins:<8} {win_rate:<10.1f}% ${avg_pnl:<11.2f} {edge:+.1f}%")


def analyze_timing_edge(df: pd.DataFrame):
    """Find edge based on when in the 15M window we trade."""
    print("\n" + "="*70)
    print("⏱️  TIMING EDGE (Seconds into Window)")
    print("="*70)
    
    timing_trades = df[df['secs_into_window'].notna()].copy()
    
    if len(timing_trades) == 0:
        print("No timing data yet.")
        return
    
    # Bucket by timing
    bins = [(0, 60), (60, 180), (180, 420), (420, 600), (600, 900)]
    labels = ['0-60s (early)', '60-180s', '180-420s', '420-600s', '600-900s (late)']
    
    print(f"\n{'Timing':<20} {'Trades':<8} {'Wins':<8} {'Win%':<10} {'Edge'}")
    print("-"*70)
    
    for (low, high), label in zip(bins, labels):
        subset = timing_trades[(timing_trades['secs_into_window'] >= low) & (timing_trades['secs_into_window'] < high)]
        if len(subset) == 0:
            continue
        
        wins = subset['was_correct'].sum()
        total = len(subset)
        win_rate = wins / total * 100
        edge = win_rate - 50
        
        marker = "⭐" if edge >= 20 else ""
        print(f"{label:<20} {total:<8} {wins:<8} {win_rate:<10.1f}% {edge:+.1f}% {marker}")


def analyze_symbol_edge(df: pd.DataFrame):
    """Find which symbols have the best edge."""
    print("\n" + "="*70)
    print("🪙 SYMBOL EDGE")
    print("="*70)
    
    print(f"\n{'Symbol':<10} {'Trades':<8} {'Wins':<8} {'Win%':<10} {'Total PnL':<12} {'Edge'}")
    print("-"*70)
    
    for symbol in ['BTC', 'ETH', 'SOL', 'XRP']:
        subset = df[df['symbol'] == symbol]
        if len(subset) == 0:
            continue
        
        wins = subset['was_correct'].sum()
        total = len(subset)
        win_rate = wins / total * 100
        total_pnl = subset['pnl'].sum()
        edge = win_rate - 50
        
        marker = "⭐" if edge >= 15 else ""
        print(f"{symbol:<10} {total:<8} {wins:<8} {win_rate:<10.1f}% ${total_pnl:<11.2f} {edge:+.1f}% {marker}")


def analyze_combined_edge(df: pd.DataFrame):
    """Find the holy grail - combined signal patterns that CRUSH it."""
    print("\n" + "="*70)
    print("🏆 COMBINED EDGE PATTERNS (High-Confidence Trades)")
    print("="*70)
    
    # Filter to trades with full data
    full_data = df[
        (df['orderbook_signal'].notna()) & 
        (df['entry_price'].notna())
    ].copy()
    
    if len(full_data) == 0:
        print("Need more trades with full signal data.")
        return
    
    full_data['ob_strength'] = full_data['orderbook_signal'].abs()
    
    # Test combined criteria
    patterns = [
        ("Strong OB (>0.85) + Good Entry (<0.51)", 
         (full_data['ob_strength'] >= 0.85) & (full_data['entry_price'] < 0.51)),
        ("Very Strong OB (>0.90) + Any Entry",
         (full_data['ob_strength'] >= 0.90)),
        ("Strong OB (>0.85) + Momentum Aligned",
         (full_data['ob_strength'] >= 0.85) & 
         ((full_data['orderbook_signal'] < 0) == (full_data['momentum_signal'] < 0))),
        ("Ultra Strong OB (>0.93)",
         (full_data['ob_strength'] >= 0.93)),
    ]
    
    print(f"\n{'Pattern':<45} {'N':<5} {'Wins':<5} {'Win%':<8} {'Edge'}")
    print("-"*70)
    
    for name, mask in patterns:
        subset = full_data[mask]
        if len(subset) == 0:
            continue
        
        wins = subset['was_correct'].sum()
        total = len(subset)
        win_rate = wins / total * 100
        edge = win_rate - 50
        
        stars = "⭐⭐⭐" if edge >= 40 else ("⭐⭐" if edge >= 25 else ("⭐" if edge >= 15 else ""))
        print(f"{name:<45} {total:<5} {wins:<5} {win_rate:<8.0f}% {edge:+.0f}% {stars}")


def calculate_kelly_criterion(df: pd.DataFrame):
    """Calculate optimal bet sizing using Kelly Criterion."""
    print("\n" + "="*70)
    print("📈 KELLY CRITERION - OPTIMAL BET SIZING")
    print("="*70)
    
    if len(df) < 5:
        print("Need more trades for Kelly calculation.")
        return
    
    # Overall stats
    wins = df['was_correct'].sum()
    total = len(df)
    win_prob = wins / total
    
    # Average win/loss amounts
    win_trades = df[df['was_correct'] == 1]
    loss_trades = df[df['was_correct'] == 0]
    
    if len(win_trades) == 0 or len(loss_trades) == 0:
        print("Need both wins and losses for Kelly.")
        return
    
    avg_win = win_trades['pnl'].mean()
    avg_loss = abs(loss_trades['pnl'].mean())
    
    # Kelly = (p * b - q) / b where p=win prob, q=loss prob, b=win/loss ratio
    b = avg_win / avg_loss if avg_loss > 0 else 1
    kelly = (win_prob * b - (1 - win_prob)) / b
    
    # Half-Kelly for safety
    half_kelly = kelly / 2
    
    print(f"\n  Win Rate: {win_prob*100:.1f}%")
    print(f"  Avg Win: ${avg_win:.2f}")
    print(f"  Avg Loss: ${avg_loss:.2f}")
    print(f"  Win/Loss Ratio: {b:.2f}")
    print(f"\n  📊 Full Kelly: {kelly*100:.1f}% of bankroll per trade")
    print(f"  🎯 Half Kelly (recommended): {half_kelly*100:.1f}% of bankroll per trade")
    
    if half_kelly > 0:
        starting_bank = 1000
        print(f"\n  With ${starting_bank} bankroll:")
        print(f"  → Bet ${starting_bank * half_kelly:.2f} per trade")


def suggest_improvements(df: pd.DataFrame):
    """Suggest specific parameter changes based on data."""
    print("\n" + "="*70)
    print("💡 SUGGESTED IMPROVEMENTS")
    print("="*70)
    
    ob_trades = df[df['orderbook_signal'].notna()]
    
    suggestions = []
    
    # Check if strong OB trades win more
    if len(ob_trades) >= 4:
        strong_ob = ob_trades[ob_trades['orderbook_signal'].abs() >= 0.90]
        if len(strong_ob) >= 2:
            strong_win_rate = strong_ob['was_correct'].mean()
            if strong_win_rate >= 0.70:
                suggestions.append(f"✅ Increase MIN_ORDERBOOK_SIGNAL to 0.90 (currently {len(strong_ob)} trades at {strong_win_rate*100:.0f}% win rate)")
            elif strong_win_rate < 0.50:
                suggestions.append("⚠️ Strong OB signals underperforming - check for market regime change")
    
    # Entry price analysis
    cheap_entries = df[df['entry_price'] < 0.48]
    if len(cheap_entries) >= 2:
        cheap_win_rate = cheap_entries['was_correct'].mean()
        if cheap_win_rate >= 0.65:
            suggestions.append(f"✅ Focus on entries < 0.48 ({cheap_win_rate*100:.0f}% win rate)")
    
    # Symbol analysis
    for symbol in ['BTC', 'ETH', 'SOL', 'XRP']:
        sym_trades = df[df['symbol'] == symbol]
        if len(sym_trades) >= 3:
            sym_win_rate = sym_trades['was_correct'].mean()
            if sym_win_rate >= 0.75:
                suggestions.append(f"✅ {symbol} is hot! ({sym_win_rate*100:.0f}% win rate)")
            elif sym_win_rate < 0.40:
                suggestions.append(f"⚠️ Consider skipping {symbol} ({sym_win_rate*100:.0f}% win rate)")
    
    if suggestions:
        for s in suggestions:
            print(f"\n  {s}")
    else:
        print("\n  Need more data to generate suggestions. Keep trading!")


def main():
    print("\n" + "="*70)
    print("🔍 EDGE FINDER - High-Confidence Trade Pattern Analysis")
    print("="*70)
    
    df = load_trades()
    
    if len(df) == 0:
        print("No resolved trades yet. Run resolve_trades.py first.")
        return
    
    total_trades = len(df)
    wins = df['was_correct'].sum()
    losses = total_trades - wins
    total_pnl = df['pnl'].sum()
    win_rate = wins / total_trades * 100
    
    print(f"\n📊 OVERALL STATS: {wins}W/{losses}L ({win_rate:.1f}%) | P&L: ${total_pnl:+.2f}")
    
    # Run all analyses
    analyze_orderbook_edge(df)
    analyze_spread_edge(df)
    analyze_timing_edge(df)
    analyze_symbol_edge(df)
    analyze_combined_edge(df)
    calculate_kelly_criterion(df)
    suggest_improvements(df)
    
    print("\n" + "="*70)
    print("🎯 KEY INSIGHT: Focus on trades with OB strength > 0.90 for highest edge")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
