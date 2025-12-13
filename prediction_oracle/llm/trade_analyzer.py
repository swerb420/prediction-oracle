#!/usr/bin/env python3
"""
Trade Analyzer - Deep analysis of trade patterns to find winning edges.

Analyzes:
1. Win rate by signal component (orderbook, momentum, ML, Grok)
2. Optimal entry conditions (timing, price levels, combinations)
3. Failure patterns (what went wrong on losses)
4. Grok accuracy analysis
5. Confidence calibration (are we over/under confident?)

Usage:
    python trade_analyzer.py              # Full analysis
    python trade_analyzer.py --winners    # Analyze only winning trades
    python trade_analyzer.py --losers     # Analyze only losing trades
    python trade_analyzer.py --grok       # Focus on Grok accuracy
"""

import argparse
import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import Optional

import pandas as pd
import numpy as np

DB_PATH = "./data/polymarket_real.db"


@dataclass
class TradePattern:
    """A discovered pattern in trades."""
    name: str
    win_rate: float
    sample_size: int
    avg_pnl: float
    conditions: dict
    description: str


class TradeAnalyzer:
    """Deep analysis of trading patterns."""
    
    def __init__(self, db_path: str = DB_PATH):
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {db_path}")
        self.df = self._load_trades()
    
    def _load_trades(self) -> pd.DataFrame:
        """Load all closed trades into a DataFrame."""
        conn = sqlite3.connect(self.db_path)
        
        # Get all columns from paper_trades
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(paper_trades)")
        columns = [row[1] for row in cursor.fetchall()]
        
        # Load closed trades
        df = pd.read_sql_query(
            "SELECT * FROM paper_trades WHERE closed_at IS NOT NULL",
            conn
        )
        conn.close()
        
        if len(df) == 0:
            print("⚠️ No closed trades found!")
            return df
        
        print(f"📊 Loaded {len(df)} closed trades")
        return df
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CORE ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def summary(self) -> dict:
        """Get overall trading summary."""
        if len(self.df) == 0:
            return {"error": "No trades to analyze"}
        
        wins = self.df[self.df['was_correct'] == 1]
        losses = self.df[self.df['was_correct'] == 0]
        
        return {
            "total_trades": len(self.df),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": len(wins) / len(self.df) if len(self.df) > 0 else 0,
            "total_pnl": self.df['pnl'].sum(),
            "avg_pnl_per_trade": self.df['pnl'].mean(),
            "avg_win_size": wins['pnl'].mean() if len(wins) > 0 else 0,
            "avg_loss_size": losses['pnl'].mean() if len(losses) > 0 else 0,
            "profit_factor": abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf'),
        }
    
    def analyze_by_column(self, column: str, bins: int = 5) -> pd.DataFrame:
        """Analyze win rate by a specific column."""
        if len(self.df) == 0 or column not in self.df.columns:
            return pd.DataFrame()
        
        # Handle numeric columns with binning
        if self.df[column].dtype in ['float64', 'int64']:
            self.df[f'{column}_bin'] = pd.cut(self.df[column], bins=bins)
            grouped = self.df.groupby(f'{column}_bin').agg({
                'was_correct': ['mean', 'count'],
                'pnl': 'sum'
            }).round(3)
            grouped.columns = ['win_rate', 'count', 'total_pnl']
            return grouped
        else:
            # Categorical columns
            grouped = self.df.groupby(column).agg({
                'was_correct': ['mean', 'count'],
                'pnl': 'sum'
            }).round(3)
            grouped.columns = ['win_rate', 'count', 'total_pnl']
            return grouped
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SIGNAL COMPONENT ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze_orderbook_edge(self) -> dict:
        """Analyze orderbook signal effectiveness."""
        if len(self.df) == 0 or 'orderbook_signal' not in self.df.columns:
            return {}
        
        results = {}
        
        # Create absolute orderbook strength column
        self.df['ob_strength'] = self.df['orderbook_signal'].abs()
        
        # Analyze by strength bins
        bins = [(0, 0.5), (0.5, 0.7), (0.7, 0.85), (0.85, 1.0)]
        for low, high in bins:
            subset = self.df[(self.df['ob_strength'] >= low) & (self.df['ob_strength'] < high)]
            if len(subset) > 0:
                results[f"ob_{low:.0%}-{high:.0%}"] = {
                    "count": len(subset),
                    "win_rate": subset['was_correct'].mean(),
                    "avg_pnl": subset['pnl'].mean(),
                    "total_pnl": subset['pnl'].sum(),
                }
        
        # Very high orderbook (our edge)
        high_ob = self.df[self.df['ob_strength'] >= 0.85]
        results["high_ob_edge"] = {
            "count": len(high_ob),
            "win_rate": high_ob['was_correct'].mean() if len(high_ob) > 0 else 0,
            "description": "OB >= 85% - Our primary edge"
        }
        
        return results
    
    def analyze_momentum_edge(self) -> dict:
        """Analyze momentum signal effectiveness."""
        if len(self.df) == 0 or 'momentum_signal' not in self.df.columns:
            return {}
        
        results = {}
        
        # Momentum aligned with direction
        # positive momentum + UP direction = aligned
        # negative momentum + DOWN direction = aligned
        
        # Check if we have 5min momentum
        if 'momentum_5min' in self.df.columns:
            mom_col = 'momentum_5min'
        else:
            mom_col = 'momentum_signal'
        
        # Create alignment check
        self.df['mom_aligned'] = (
            ((self.df[mom_col] > 0) & (self.df['direction'] == 'UP')) |
            ((self.df[mom_col] < 0) & (self.df['direction'] == 'DOWN'))
        )
        
        aligned = self.df[self.df['mom_aligned'] == True]
        not_aligned = self.df[self.df['mom_aligned'] == False]
        
        results["momentum_aligned"] = {
            "count": len(aligned),
            "win_rate": aligned['was_correct'].mean() if len(aligned) > 0 else 0,
            "avg_pnl": aligned['pnl'].mean() if len(aligned) > 0 else 0,
        }
        results["momentum_not_aligned"] = {
            "count": len(not_aligned),
            "win_rate": not_aligned['was_correct'].mean() if len(not_aligned) > 0 else 0,
            "avg_pnl": not_aligned['pnl'].mean() if len(not_aligned) > 0 else 0,
        }
        
        return results
    
    def analyze_grok_accuracy(self) -> dict:
        """Analyze Grok prediction accuracy."""
        if len(self.df) == 0:
            return {}
        
        results = {}
        
        # Trades with Grok
        grok_trades = self.df[self.df['grok_used'] == 1]
        no_grok = self.df[self.df['grok_used'] == 0]
        
        results["with_grok"] = {
            "count": len(grok_trades),
            "win_rate": grok_trades['was_correct'].mean() if len(grok_trades) > 0 else 0,
            "avg_pnl": grok_trades['pnl'].mean() if len(grok_trades) > 0 else 0,
        }
        
        results["without_grok"] = {
            "count": len(no_grok),
            "win_rate": no_grok['was_correct'].mean() if len(no_grok) > 0 else 0,
            "avg_pnl": no_grok['pnl'].mean() if len(no_grok) > 0 else 0,
        }
        
        # Grok agreed vs disagreed
        if len(grok_trades) > 0:
            grok_agreed = grok_trades[grok_trades['grok_agreed'] == 1]
            grok_disagreed = grok_trades[grok_trades['grok_agreed'] == 0]
            
            results["grok_agreed"] = {
                "count": len(grok_agreed),
                "win_rate": grok_agreed['was_correct'].mean() if len(grok_agreed) > 0 else 0,
            }
            
            results["grok_disagreed"] = {
                "count": len(grok_disagreed),
                "win_rate": grok_disagreed['was_correct'].mean() if len(grok_disagreed) > 0 else 0,
            }
        
        return results
    
    def analyze_entry_timing(self) -> dict:
        """Analyze best entry timing within windows."""
        if len(self.df) == 0 or 'secs_into_window' not in self.df.columns:
            return {}
        
        results = {}
        
        # Early, mid, late entry
        early = self.df[self.df['secs_into_window'] < 300]  # First 5 minutes
        mid = self.df[(self.df['secs_into_window'] >= 300) & (self.df['secs_into_window'] < 600)]
        late = self.df[self.df['secs_into_window'] >= 600]  # Last 5 minutes
        
        results["early_entry"] = {
            "count": len(early),
            "win_rate": early['was_correct'].mean() if len(early) > 0 else 0,
            "description": "First 5 minutes of window"
        }
        
        results["mid_entry"] = {
            "count": len(mid),
            "win_rate": mid['was_correct'].mean() if len(mid) > 0 else 0,
            "description": "Middle 5 minutes"
        }
        
        results["late_entry"] = {
            "count": len(late),
            "win_rate": late['was_correct'].mean() if len(late) > 0 else 0,
            "description": "Last 5 minutes"
        }
        
        return results
    
    def analyze_entry_price(self) -> dict:
        """Analyze entry price levels."""
        if len(self.df) == 0 or 'entry_price' not in self.df.columns:
            return {}
        
        results = {}
        
        # Entry price bins
        cheap = self.df[self.df['entry_price'] < 0.45]
        fair = self.df[(self.df['entry_price'] >= 0.45) & (self.df['entry_price'] <= 0.55)]
        expensive = self.df[self.df['entry_price'] > 0.55]
        
        results["cheap_entry"] = {
            "count": len(cheap),
            "win_rate": cheap['was_correct'].mean() if len(cheap) > 0 else 0,
            "description": "Entry < 45%"
        }
        
        results["fair_entry"] = {
            "count": len(fair),
            "win_rate": fair['was_correct'].mean() if len(fair) > 0 else 0,
            "description": "Entry 45-55%"
        }
        
        results["expensive_entry"] = {
            "count": len(expensive),
            "win_rate": expensive['was_correct'].mean() if len(expensive) > 0 else 0,
            "description": "Entry > 55%"
        }
        
        return results
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMBINATION ANALYSIS - Find the best combinations
    # ═══════════════════════════════════════════════════════════════════════════
    
    def find_winning_combinations(self) -> list[TradePattern]:
        """Find the most profitable signal combinations."""
        if len(self.df) == 0:
            return []
        
        patterns = []
        
        # Pattern 1: High OB + Low Entry Price
        if 'orderbook_signal' in self.df.columns:
            combo1 = self.df[
                (self.df['orderbook_signal'].abs() >= 0.85) & 
                (self.df['entry_price'] < 0.51)
            ]
            if len(combo1) >= 3:
                patterns.append(TradePattern(
                    name="High OB + Value Entry",
                    win_rate=combo1['was_correct'].mean(),
                    sample_size=len(combo1),
                    avg_pnl=combo1['pnl'].mean(),
                    conditions={"ob_strength": ">=0.85", "entry_price": "<0.51"},
                    description="Strong orderbook imbalance + entry below 51%"
                ))
        
        # Pattern 2: Momentum Aligned + OB Aligned
        if 'momentum_signal' in self.df.columns and 'orderbook_signal' in self.df.columns:
            self.df['signals_aligned'] = (
                (self.df['momentum_signal'] * self.df['orderbook_signal']) > 0
            )
            aligned = self.df[self.df['signals_aligned'] == True]
            if len(aligned) >= 3:
                patterns.append(TradePattern(
                    name="Momentum + OB Aligned",
                    win_rate=aligned['was_correct'].mean(),
                    sample_size=len(aligned),
                    avg_pnl=aligned['pnl'].mean(),
                    conditions={"momentum": "aligned with OB"},
                    description="Momentum and orderbook pointing same direction"
                ))
        
        # Pattern 3: Grok Confirmed + High OB
        grok_ob = self.df[
            (self.df['grok_agreed'] == 1) & 
            (self.df['orderbook_signal'].abs() >= 0.70)
        ]
        if len(grok_ob) >= 3:
            patterns.append(TradePattern(
                name="Grok Confirmed + Strong OB",
                win_rate=grok_ob['was_correct'].mean(),
                sample_size=len(grok_ob),
                avg_pnl=grok_ob['pnl'].mean(),
                conditions={"grok": "agreed", "ob_strength": ">=0.70"},
                description="Grok agrees + orderbook >= 70% imbalance"
            ))
        
        # Pattern 4: High Confidence + Early Entry
        high_conf_early = self.df[
            (self.df['confidence'] >= 0.60) & 
            (self.df['secs_into_window'] < 300)
        ]
        if len(high_conf_early) >= 3:
            patterns.append(TradePattern(
                name="High Confidence + Early Entry",
                win_rate=high_conf_early['was_correct'].mean(),
                sample_size=len(high_conf_early),
                avg_pnl=high_conf_early['pnl'].mean(),
                conditions={"confidence": ">=60%", "timing": "first 5 min"},
                description="60%+ confidence with early window entry"
            ))
        
        # Sort by win rate
        patterns.sort(key=lambda x: x.win_rate, reverse=True)
        return patterns
    
    def find_losing_patterns(self) -> list[TradePattern]:
        """Find patterns that tend to lose."""
        if len(self.df) == 0:
            return []
        
        patterns = []
        losses = self.df[self.df['was_correct'] == 0]
        
        if len(losses) < 3:
            return patterns
        
        # Pattern 1: Low OB + Expensive Entry
        bad_combo = self.df[
            (self.df['orderbook_signal'].abs() < 0.50) & 
            (self.df['entry_price'] > 0.55)
        ]
        if len(bad_combo) >= 3:
            patterns.append(TradePattern(
                name="Weak OB + Expensive Entry",
                win_rate=bad_combo['was_correct'].mean(),
                sample_size=len(bad_combo),
                avg_pnl=bad_combo['pnl'].mean(),
                conditions={"ob_strength": "<0.50", "entry_price": ">0.55"},
                description="⚠️ AVOID: Weak orderbook + paying premium"
            ))
        
        # Pattern 2: Signals Conflicting
        if 'signals_aligned' in self.df.columns:
            conflicting = self.df[self.df['signals_aligned'] == False]
            if len(conflicting) >= 3:
                patterns.append(TradePattern(
                    name="Signals Conflicting",
                    win_rate=conflicting['was_correct'].mean(),
                    sample_size=len(conflicting),
                    avg_pnl=conflicting['pnl'].mean(),
                    conditions={"momentum_vs_ob": "conflicting"},
                    description="⚠️ AVOID: Momentum and OB disagreeing"
                ))
        
        # Pattern 3: Grok Disagreed
        grok_no = self.df[self.df['grok_agreed'] == 0]
        if len(grok_no) >= 3:
            patterns.append(TradePattern(
                name="Grok Disagreed",
                win_rate=grok_no['was_correct'].mean(),
                sample_size=len(grok_no),
                avg_pnl=grok_no['pnl'].mean(),
                conditions={"grok": "disagreed"},
                description="⚠️ CAUTION: Traded against Grok's recommendation"
            ))
        
        return patterns
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFIDENCE CALIBRATION
    # ═══════════════════════════════════════════════════════════════════════════
    
    def analyze_confidence_calibration(self) -> dict:
        """Check if our confidence levels are well-calibrated."""
        if len(self.df) == 0 or 'confidence' not in self.df.columns:
            return {}
        
        results = {}
        
        # Bin by confidence
        bins = [(0.50, 0.55), (0.55, 0.60), (0.60, 0.65), (0.65, 0.70), (0.70, 1.0)]
        
        for low, high in bins:
            subset = self.df[(self.df['confidence'] >= low) & (self.df['confidence'] < high)]
            if len(subset) > 0:
                actual_win_rate = subset['was_correct'].mean()
                expected = (low + high) / 2
                calibration_error = actual_win_rate - expected
                
                results[f"conf_{low:.0%}-{high:.0%}"] = {
                    "count": len(subset),
                    "actual_win_rate": actual_win_rate,
                    "expected_win_rate": expected,
                    "calibration_error": calibration_error,
                    "is_calibrated": abs(calibration_error) < 0.10,
                    "status": "✅ Well calibrated" if abs(calibration_error) < 0.10 
                             else "⚠️ Over-confident" if calibration_error < 0 
                             else "📈 Under-confident"
                }
        
        return results
    
    # ═══════════════════════════════════════════════════════════════════════════
    # REPORTING
    # ═══════════════════════════════════════════════════════════════════════════
    
    def print_full_report(self):
        """Print comprehensive analysis report."""
        print("\n" + "=" * 70)
        print("  📊 TRADE ANALYSIS REPORT")
        print("=" * 70)
        
        # Summary
        summary = self.summary()
        print(f"\n📈 OVERALL PERFORMANCE")
        print(f"   Total Trades: {summary.get('total_trades', 0)}")
        print(f"   Wins: {summary.get('wins', 0)} | Losses: {summary.get('losses', 0)}")
        print(f"   Win Rate: {summary.get('win_rate', 0):.1%}")
        print(f"   Total P&L: ${summary.get('total_pnl', 0):+.2f}")
        print(f"   Avg P&L per Trade: ${summary.get('avg_pnl_per_trade', 0):+.2f}")
        print(f"   Profit Factor: {summary.get('profit_factor', 0):.2f}x")
        
        # Orderbook Analysis
        print(f"\n📊 ORDERBOOK EDGE")
        ob_analysis = self.analyze_orderbook_edge()
        for name, data in ob_analysis.items():
            if isinstance(data, dict) and 'win_rate' in data:
                print(f"   {name}: {data['count']} trades, {data['win_rate']:.1%} win rate")
        
        # Momentum Analysis
        print(f"\n🔄 MOMENTUM ANALYSIS")
        mom_analysis = self.analyze_momentum_edge()
        for name, data in mom_analysis.items():
            if isinstance(data, dict) and 'win_rate' in data:
                print(f"   {name}: {data['count']} trades, {data['win_rate']:.1%} win rate")
        
        # Grok Analysis
        print(f"\n🤖 GROK ACCURACY")
        grok_analysis = self.analyze_grok_accuracy()
        for name, data in grok_analysis.items():
            if isinstance(data, dict) and 'win_rate' in data:
                print(f"   {name}: {data['count']} trades, {data['win_rate']:.1%} win rate")
        
        # Entry Timing
        print(f"\n⏰ ENTRY TIMING")
        timing = self.analyze_entry_timing()
        for name, data in timing.items():
            if isinstance(data, dict) and 'win_rate' in data:
                print(f"   {name}: {data['count']} trades, {data['win_rate']:.1%} win rate")
        
        # Entry Price
        print(f"\n💰 ENTRY PRICE LEVELS")
        price = self.analyze_entry_price()
        for name, data in price.items():
            if isinstance(data, dict) and 'win_rate' in data:
                print(f"   {name}: {data['count']} trades, {data['win_rate']:.1%} win rate")
        
        # Winning Patterns
        print(f"\n🏆 WINNING PATTERNS")
        winning = self.find_winning_combinations()
        for i, pattern in enumerate(winning[:5], 1):
            print(f"   {i}. {pattern.name}: {pattern.win_rate:.1%} ({pattern.sample_size} trades)")
            print(f"      {pattern.description}")
        
        # Losing Patterns
        print(f"\n⚠️ PATTERNS TO AVOID")
        losing = self.find_losing_patterns()
        for pattern in losing[:3]:
            print(f"   ❌ {pattern.name}: {pattern.win_rate:.1%} ({pattern.sample_size} trades)")
            print(f"      {pattern.description}")
        
        # Confidence Calibration
        print(f"\n🎯 CONFIDENCE CALIBRATION")
        calibration = self.analyze_confidence_calibration()
        for name, data in calibration.items():
            if isinstance(data, dict) and 'status' in data:
                print(f"   {name}: Actual {data['actual_win_rate']:.1%} vs Expected {data['expected_win_rate']:.1%} {data['status']}")
        
        print("\n" + "=" * 70)
        print("  🎯 KEY RECOMMENDATIONS")
        print("=" * 70)
        
        # Generate recommendations
        recommendations = self._generate_recommendations()
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")
        
        print("\n" + "=" * 70)
    
    def _generate_recommendations(self) -> list[str]:
        """Generate actionable recommendations from analysis."""
        recs = []
        
        # Check orderbook edge
        ob_analysis = self.analyze_orderbook_edge()
        high_ob = ob_analysis.get('high_ob_edge', {})
        if high_ob.get('win_rate', 0) > 0.80 and high_ob.get('count', 0) >= 5:
            recs.append(f"🎯 STRONG EDGE: OB >= 85% has {high_ob['win_rate']:.0%} win rate! Increase position size on these.")
        
        # Check Grok value
        grok = self.analyze_grok_accuracy()
        grok_agreed = grok.get('grok_agreed', {})
        grok_disagreed = grok.get('grok_disagreed', {})
        if grok_agreed.get('win_rate', 0) > grok_disagreed.get('win_rate', 0) + 0.15:
            recs.append(f"🤖 When Grok AGREES: {grok_agreed.get('win_rate', 0):.0%} win rate. Skip trades where Grok disagrees.")
        
        # Check entry timing
        timing = self.analyze_entry_timing()
        best_timing = max(timing.items(), key=lambda x: x[1].get('win_rate', 0) if isinstance(x[1], dict) else 0)
        if best_timing[1].get('win_rate', 0) > 0.65:
            recs.append(f"⏰ Best entry timing: {best_timing[0]} ({best_timing[1]['win_rate']:.0%} win rate)")
        
        # Check entry price
        price = self.analyze_entry_price()
        cheap = price.get('cheap_entry', {})
        expensive = price.get('expensive_entry', {})
        if cheap.get('win_rate', 0) > expensive.get('win_rate', 0) + 0.10:
            recs.append(f"💰 Cheap entries (<45%) outperform expensive (>55%) by {(cheap.get('win_rate', 0) - expensive.get('win_rate', 0)):.0%}")
        
        # Check calibration
        calibration = self.analyze_confidence_calibration()
        for name, data in calibration.items():
            if isinstance(data, dict):
                if data.get('calibration_error', 0) < -0.15:
                    recs.append(f"⚠️ OVER-CONFIDENT in {name} range. Reduce position sizes or raise thresholds.")
                elif data.get('calibration_error', 0) > 0.15:
                    recs.append(f"📈 UNDER-CONFIDENT in {name} range. Increase position sizes!")
        
        if not recs:
            recs.append("📊 More data needed. Keep trading to build statistical significance.")
        
        return recs


def main():
    parser = argparse.ArgumentParser(description="Analyze trading patterns")
    parser.add_argument("--winners", action="store_true", help="Focus on winning trades")
    parser.add_argument("--losers", action="store_true", help="Focus on losing trades")
    parser.add_argument("--grok", action="store_true", help="Focus on Grok accuracy")
    parser.add_argument("--export", type=str, help="Export to CSV file")
    
    args = parser.parse_args()
    
    try:
        analyzer = TradeAnalyzer()
        
        if args.export:
            analyzer.df.to_csv(args.export, index=False)
            print(f"✅ Exported {len(analyzer.df)} trades to {args.export}")
        else:
            analyzer.print_full_report()
            
    except FileNotFoundError as e:
        print(f"❌ {e}")
        print("   Run some trades first with: python smart_signal_trader.py --trade")


if __name__ == "__main__":
    main()
