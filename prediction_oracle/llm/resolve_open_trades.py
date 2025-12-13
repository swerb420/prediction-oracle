#!/usr/bin/env python3
"""Resolve all OPEN trades from the database using Binance historical data."""

import asyncio
import httpx
import sqlite3
from datetime import datetime, timezone, timedelta

DB_PATH = "/root/prediction-oracle/prediction_oracle/llm/data/polymarket_real.db"

async def get_binance_price_direction(symbol: str, window_start: datetime, window_end: datetime) -> tuple[str, float]:
    """Get the ACTUAL price direction from Binance klines."""
    start_ts = int(window_start.timestamp()) * 1000
    end_ts = int(window_end.timestamp()) * 1000
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(
                'https://api.binance.com/api/v3/klines',
                params={
                    'symbol': f'{symbol}USDT',
                    'interval': '1m',
                    'startTime': start_ts,
                    'endTime': end_ts,
                    'limit': 20
                }
            )
            klines = resp.json()
            
            if klines and len(klines) >= 2:
                open_price = float(klines[0][1])
                close_price = float(klines[-1][4])
                change_pct = (close_price - open_price) / open_price * 100
                direction = 'UP' if close_price > open_price else 'DOWN'
                return direction, change_pct
            
            return 'UNKNOWN', 0.0
    except Exception as e:
        print(f"Binance API error for {symbol}: {e}")
        return 'UNKNOWN', 0.0


def get_window_times(trade_timestamp: str) -> tuple[datetime, datetime]:
    """Get the 15-minute window start/end for a trade timestamp."""
    # Parse the timestamp
    ts = datetime.fromisoformat(trade_timestamp.replace('+00:00', ''))
    ts = ts.replace(tzinfo=timezone.utc)
    
    # Find the 15-min window: 00, 15, 30, 45
    minute = ts.minute
    window_minute = (minute // 15) * 15
    
    window_start = ts.replace(minute=window_minute, second=0, microsecond=0)
    window_end = window_start + timedelta(minutes=15)
    
    return window_start, window_end


async def resolve_open_trades():
    """Resolve all OPEN trades using Binance price data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all OPEN trades from last 3 hours
    cursor.execute("""
        SELECT id, symbol, direction, entry_price, size_usd, timestamp, confidence
        FROM paper_trades 
        WHERE was_correct IS NULL
        AND timestamp >= datetime('now', '-180 minutes')
        ORDER BY id
    """)
    
    open_trades = cursor.fetchall()
    print(f"Found {len(open_trades)} OPEN trades to resolve...")
    
    results = {"wins": 0, "losses": 0, "unknown": 0, "pnl": 0.0}
    resolved_trades = []
    
    for trade in open_trades:
        trade_id, symbol, direction, entry_price, size_usd, timestamp, confidence = trade
        
        window_start, window_end = get_window_times(timestamp)
        
        # Check if window has ended
        now = datetime.now(timezone.utc)
        if window_end > now:
            print(f"  #{trade_id} {symbol} {direction} - Window not ended yet ({window_end.strftime('%H:%M')} UTC)")
            continue
        
        # Get actual direction from Binance
        actual_direction, change_pct = await get_binance_price_direction(symbol, window_start, window_end)
        
        if actual_direction == 'UNKNOWN':
            results["unknown"] += 1
            print(f"  #{trade_id} {symbol} {direction} - Could not get price data")
            continue
        
        # Determine win/loss
        was_correct = 1 if actual_direction == direction else 0
        
        # Calculate PnL
        if was_correct:
            # Win: size * (1 - entry_price) / entry_price
            pnl = size_usd * (1 - entry_price) / entry_price
            results["wins"] += 1
        else:
            # Loss: -size
            pnl = -size_usd
            results["losses"] += 1
        
        results["pnl"] += pnl
        
        # Update the database
        cursor.execute("""
            UPDATE paper_trades 
            SET actual_outcome = ?, was_correct = ?, pnl = ?
            WHERE id = ?
        """, (actual_direction, was_correct, pnl, trade_id))
        
        result_str = "WIN" if was_correct else "LOSS"
        print(f"  #{trade_id} {symbol} {direction} @ {entry_price:.3f} -> Actual: {actual_direction} ({change_pct:+.2f}%) = {result_str} (${pnl:+.2f})")
        
        resolved_trades.append({
            "id": trade_id,
            "symbol": symbol,
            "direction": direction,
            "actual": actual_direction,
            "was_correct": was_correct,
            "pnl": pnl,
            "size": size_usd,
            "conf": confidence
        })
    
    conn.commit()
    conn.close()
    
    print("\n" + "="*60)
    print("RESOLUTION SUMMARY")
    print("="*60)
    print(f"Resolved: {results['wins'] + results['losses']} trades")
    print(f"Wins: {results['wins']}")
    print(f"Losses: {results['losses']}")
    if results['wins'] + results['losses'] > 0:
        wr = 100 * results['wins'] / (results['wins'] + results['losses'])
        print(f"Win Rate: {wr:.1f}%")
    print(f"PnL: ${results['pnl']:+.2f}")
    print(f"Still Unknown: {results['unknown']}")
    
    return results, resolved_trades


if __name__ == "__main__":
    asyncio.run(resolve_open_trades())
