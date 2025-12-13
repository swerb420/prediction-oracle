#!/usr/bin/env python3
"""
Auto-resolve open trades by fetching actual price data from Binance.
Run this periodically to update trade outcomes.
"""
import sqlite3
import httpx
import asyncio
from datetime import datetime, timezone, timedelta

# Timeframe window durations in seconds
TIMEFRAME_WINDOWS = {
    "15M": 900,
    "1H": 3600,
    "4H": 14400,
}


async def get_price_change(symbol: str, start_ts: int, end_ts: int) -> tuple[str, float]:
    """Get price direction and change % from Binance klines."""
    async with httpx.AsyncClient() as client:
        # Choose appropriate interval based on duration
        duration = end_ts - start_ts
        if duration <= 900:
            interval = '1m'
            limit = 20
        elif duration <= 3600:
            interval = '5m'
            limit = 15
        else:
            interval = '15m'
            limit = 20
            
        resp = await client.get(
            'https://api.binance.com/api/v3/klines',
            params={
                'symbol': f'{symbol}USDT',
                'interval': interval,
                'startTime': start_ts * 1000,
                'endTime': end_ts * 1000,
                'limit': limit
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


def get_window_times(timestamp_str: str, timeframe: str = "15M") -> tuple[datetime, datetime]:
    """Calculate window start/end from trade timestamp based on timeframe."""
    # Parse timestamp
    if '+' in timestamp_str:
        dt = datetime.fromisoformat(timestamp_str.replace('+00:00', ''))
    else:
        dt = datetime.fromisoformat(timestamp_str)
    
    dt = dt.replace(tzinfo=timezone.utc)
    
    window_secs = TIMEFRAME_WINDOWS.get(timeframe, 900)
    
    if timeframe == "15M":
        # Windows start at :00, :15, :30, :45
        window_start_min = (dt.minute // 15) * 15
        window_start = dt.replace(minute=window_start_min, second=0, microsecond=0)
    elif timeframe == "1H":
        # Windows start at top of hour
        window_start = dt.replace(minute=0, second=0, microsecond=0)
    elif timeframe == "4H":
        # Windows start at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00
        window_start_hour = (dt.hour // 4) * 4
        window_start = dt.replace(hour=window_start_hour, minute=0, second=0, microsecond=0)
    else:
        # Default to 15M
        window_start_min = (dt.minute // 15) * 15
        window_start = dt.replace(minute=window_start_min, second=0, microsecond=0)
    
    window_end = window_start + timedelta(seconds=window_secs)
    
    return window_start, window_end


async def resolve_trades():
    """Resolve all open trades that have expired."""
    conn = sqlite3.connect('data/polymarket_real.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    
    # Get open trades (include timeframe column with fallback to 15M for old trades)
    c.execute('''
        SELECT id, symbol, direction, entry_price, size_usd, timestamp,
               COALESCE(timeframe, '15M') as timeframe
        FROM paper_trades 
        WHERE actual_outcome IS NULL
        ORDER BY id
    ''')
    
    open_trades = list(c.fetchall())
    
    if not open_trades:
        print("No open trades to resolve.")
        return
    
    now = datetime.now(timezone.utc)
    resolved = 0
    wins = 0
    losses = 0
    total_pnl = 0.0
    
    print(f"\n{'='*70}")
    print(f"  RESOLVING {len(open_trades)} OPEN TRADES")
    print(f"{'='*70}\n")
    
    for trade in open_trades:
        timeframe = trade['timeframe']
        window_start, window_end = get_window_times(trade['timestamp'], timeframe)
        
        # Only resolve if window has ended (with 60s buffer)
        if now < window_end + timedelta(seconds=60):
            secs_left = (window_end - now).total_seconds()
            print(f"  #{trade['id']} {trade['symbol']}/{timeframe}: Window ends in {secs_left:.0f}s - WAITING")
            continue
        
        # Fetch actual outcome from Binance
        start_ts = int(window_start.timestamp())
        end_ts = int(window_end.timestamp())
        
        try:
            outcome, change_pct = await get_price_change(trade['symbol'], start_ts, end_ts)
        except Exception as e:
            print(f"  #{trade['id']} {trade['symbol']}: Error fetching price - {e}")
            continue
        
        if outcome == 'UNKNOWN':
            print(f"  #{trade['id']} {trade['symbol']}/{timeframe}: Could not determine outcome")
            continue
        
        # Calculate P&L
        won = (trade['direction'] == outcome)
        
        if won:
            pnl = trade['size_usd'] * (1.0 - trade['entry_price']) / trade['entry_price']
            wins += 1
        else:
            pnl = -trade['size_usd']
            losses += 1
        
        total_pnl += pnl
        resolved += 1
        
        # Update database
        c.execute('''
            UPDATE paper_trades 
            SET actual_outcome = ?, pnl = ?, was_correct = ?, closed_at = datetime('now')
            WHERE id = ?
        ''', (outcome, pnl, 1 if won else 0, trade['id']))
        
        result = "✅ WIN " if won else "❌ LOSS"
        print(f"  #{trade['id']} {trade['symbol']}/{timeframe} {trade['direction']} → {outcome} ({change_pct:+.2f}%) | {result} | ${pnl:+.2f}")
    
    conn.commit()
    
    # Get overall stats
    c.execute('SELECT COUNT(*) FROM paper_trades WHERE was_correct = 1')
    total_wins = c.fetchone()[0]
    c.execute('SELECT COUNT(*) FROM paper_trades WHERE was_correct = 0')  
    total_losses = c.fetchone()[0]
    c.execute('SELECT SUM(pnl) FROM paper_trades WHERE pnl IS NOT NULL')
    cumulative_pnl = c.fetchone()[0] or 0
    
    conn.close()
    
    print(f"\n{'='*70}")
    print(f"  SESSION: Resolved {resolved} | +{wins}W -{losses}L | P&L: ${total_pnl:+.2f}")
    print(f"  LIFETIME: {total_wins}W / {total_losses}L | Total P&L: ${cumulative_pnl:+.2f}")
    if total_wins + total_losses > 0:
        print(f"  WIN RATE: {total_wins/(total_wins+total_losses)*100:.1f}%")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    asyncio.run(resolve_trades())
