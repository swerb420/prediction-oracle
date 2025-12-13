#!/usr/bin/env python3
"""
Import data from the old prediction_oracle database into the new system.
This combines all existing paper trading data for analysis and fine-tuning.
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path


def import_old_trades():
    """Import trades from the old paper_trades.db"""
    
    old_db = Path("/root/prediction_oracle/paper_trades.db")
    new_db = Path("/root/prediction-oracle/prediction_oracle.db")
    
    if not old_db.exists():
        print(f"Old database not found: {old_db}")
        return
    
    # Connect to old database
    old_conn = sqlite3.connect(old_db)
    old_conn.row_factory = sqlite3.Row
    old_cursor = old_conn.cursor()
    
    # Connect to new database  
    new_conn = sqlite3.connect(new_db)
    new_cursor = new_conn.cursor()
    
    # Create legacy_trades table if not exists
    new_cursor.execute("""
        CREATE TABLE IF NOT EXISTS legacy_trades (
            id INTEGER PRIMARY KEY,
            trade_id TEXT UNIQUE,
            source TEXT,
            market_id TEXT,
            category TEXT,
            question TEXT,
            direction TEXT,
            entry_price REAL,
            bet_size REAL,
            potential_payout REAL,
            potential_profit REAL,
            edge REAL,
            llm_fair REAL,
            llm_confidence TEXT,
            llm_reason TEXT,
            grok_fair REAL,
            grok_dir TEXT,
            gpt_fair REAL,
            gpt_dir TEXT,
            hours_left REAL,
            closes_at TEXT,
            created_at TEXT,
            resolved_at TEXT,
            outcome TEXT,
            actual_price REAL,
            pnl REAL,
            grok_correct INTEGER,
            gpt_correct INTEGER,
            imported_at TEXT
        )
    """)
    
    # Get all trades from old database
    old_cursor.execute("SELECT * FROM trades")
    old_trades = old_cursor.fetchall()
    
    imported = 0
    skipped = 0
    
    for trade in old_trades:
        try:
            new_cursor.execute("""
                INSERT OR IGNORE INTO legacy_trades (
                    trade_id, source, market_id, category, question, direction,
                    entry_price, bet_size, potential_payout, potential_profit,
                    edge, llm_fair, llm_confidence, llm_reason,
                    grok_fair, grok_dir, gpt_fair, gpt_dir,
                    hours_left, closes_at, created_at, resolved_at,
                    outcome, actual_price, pnl, grok_correct, gpt_correct,
                    imported_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade['trade_id'], trade['source'], trade['market_id'],
                trade['category'], trade['question'], trade['direction'],
                trade['entry_price'], trade['bet_size'], trade['potential_payout'],
                trade['potential_profit'], trade['edge'], trade['llm_fair'],
                trade['llm_confidence'], trade['llm_reason'],
                trade['grok_fair'], trade['grok_dir'], trade['gpt_fair'], trade['gpt_dir'],
                trade['hours_left'], trade['closes_at'], trade['created_at'],
                trade['resolved_at'], trade['outcome'], trade['actual_price'],
                trade['pnl'], trade['grok_correct'], trade['gpt_correct'],
                datetime.utcnow().isoformat()
            ))
            
            if new_cursor.rowcount > 0:
                imported += 1
            else:
                skipped += 1
                
        except Exception as e:
            print(f"Error importing trade {trade['trade_id']}: {e}")
            skipped += 1
    
    new_conn.commit()
    
    print(f"\n=== Import Summary ===")
    print(f"Imported: {imported} trades")
    print(f"Skipped (duplicates): {skipped} trades")
    
    # Show stats
    new_cursor.execute("""
        SELECT 
            COUNT(*) as total,
            SUM(CASE WHEN outcome IS NOT NULL AND outcome != '' THEN 1 ELSE 0 END) as resolved,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
            ROUND(SUM(pnl), 2) as total_pnl
        FROM legacy_trades
    """)
    stats = new_cursor.fetchone()
    
    print(f"\n=== Legacy Data Stats ===")
    print(f"Total trades: {stats[0]}")
    print(f"Resolved: {stats[1]}")
    print(f"Wins: {stats[2]}")
    print(f"Losses: {stats[3]}")
    print(f"Total P&L: ${stats[4]}")
    
    # Category breakdown
    print(f"\n=== By Category ===")
    new_cursor.execute("""
        SELECT 
            category,
            COUNT(*) as trades,
            SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN pnl < 0 THEN 1 ELSE 0 END) as losses,
            ROUND(SUM(pnl), 2) as pnl
        FROM legacy_trades
        WHERE pnl IS NOT NULL
        GROUP BY category
        ORDER BY trades DESC
    """)
    
    for row in new_cursor.fetchall():
        win_rate = (row[2] / (row[2] + row[3]) * 100) if (row[2] + row[3]) > 0 else 0
        print(f"  {row[0]}: {row[1]} trades, {row[2]}W/{row[3]}L ({win_rate:.0f}%), P&L: ${row[4]}")
    
    old_conn.close()
    new_conn.close()


def export_finetuning_data():
    """Export data in JSONL format for fine-tuning."""
    
    new_db = Path("/root/prediction-oracle/prediction_oracle.db")
    output_path = Path("/root/prediction-oracle/finetuning_data.jsonl")
    
    conn = sqlite3.connect(new_db)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Get resolved trades with LLM predictions
    cursor.execute("""
        SELECT * FROM legacy_trades
        WHERE outcome IS NOT NULL 
        AND outcome != ''
        AND llm_reason IS NOT NULL
        AND llm_reason != ''
    """)
    
    trades = cursor.fetchall()
    
    with open(output_path, 'w') as f:
        for trade in trades:
            # Create training example
            system_prompt = """You are an expert prediction market analyst. Given market information, predict the probability of the YES outcome.
Respond with JSON: {"probability": 0.XX, "direction": "YES" or "NO", "confidence": "high/medium/low", "reasoning": "brief explanation"}"""
            
            user_prompt = f"""Market: {trade['question']}
Category: {trade['category']}
Current YES price: {trade['entry_price']:.2f}
Time remaining: {trade['hours_left']:.1f} hours
Source: {trade['source']}

What is the true probability of YES?"""
            
            # The "ideal" response based on actual outcome
            actual_outcome = trade['outcome']
            was_correct = (actual_outcome == trade['direction'])
            
            # If the trade was correct, use the original prediction
            # If wrong, we can still learn from the reasoning
            assistant_response = json.dumps({
                "probability": trade['llm_fair'],
                "direction": trade['direction'],
                "confidence": trade['llm_confidence'],
                "reasoning": trade['llm_reason'],
                "_actual_outcome": actual_outcome,
                "_was_correct": was_correct,
                "_pnl": trade['pnl']
            })
            
            entry = {
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_response}
                ],
                "_metadata": {
                    "market_id": trade['market_id'],
                    "category": trade['category'],
                    "actual_outcome": actual_outcome,
                    "was_correct": was_correct,
                    "pnl": trade['pnl'],
                    "edge": trade['edge']
                }
            }
            
            f.write(json.dumps(entry) + "\n")
    
    print(f"\n=== Fine-tuning Export ===")
    print(f"Exported {len(trades)} samples to {output_path}")
    
    conn.close()


if __name__ == "__main__":
    print("Importing old paper trading data...")
    import_old_trades()
    
    print("\n" + "="*50)
    print("Exporting fine-tuning data...")
    export_finetuning_data()
