"""
Market Open Strategy for Polymarket 15M Crypto Markets

Based on 60-day Binance backtest across BTC, ETH, SOL:
- Verified patterns at US market open (9am-11am EST / 14:00-16:00 UTC)
- Day-of-week patterns with 58-67% win rates

GOLDEN WINDOWS (60%+ WR verified):
    Wednesday 9am EST (14:00 UTC): DOWN = 66.7% WR ← BEST
    Tuesday 10am EST (15:00 UTC): UP = 61.1% WR
    Friday 10am EST (15:00 UTC): DOWN = 60.2% WR
    Thursday 1pm EST (18:00 UTC): DOWN = 60.2% WR
    Friday 2pm EST (19:00 UTC): UP = 62.0% WR
    Tuesday 3pm EST (20:00 UTC): DOWN = 63.0% WR
    Wednesday 4pm EST (21:00 UTC): UP = 62.0% WR

AVOID:
    Any slot with <55% edge - random noise

Usage:
    from market_open_strategy import get_market_open_signal
    signal = get_market_open_signal()  # Returns {'direction': 'DOWN', 'confidence': 0.667, ...}
"""

from datetime import datetime, timezone
from typing import Optional, Dict, Any
import requests

# Trading rules derived from 60-day Binance backtest
# Format: (hour_utc, day_of_week): (direction, win_rate, sample_size)
MARKET_OPEN_RULES = {
    # Wednesday 9am EST = 14:00 UTC - BEST SLOT
    (14, 2): ('DOWN', 0.667, 108),
    
    # Tuesday slots
    (15, 1): ('UP', 0.611, 108),    # 10am EST
    (20, 1): ('DOWN', 0.630, 108),  # 3pm EST
    (22, 1): ('UP', 0.611, 108),    # 5pm EST
    
    # Wednesday slots
    (17, 2): ('UP', 0.630, 108),    # 12pm EST
    (21, 2): ('UP', 0.620, 108),    # 4pm EST
    
    # Thursday slots
    (13, 3): ('DOWN', 0.602, 108),  # 8am EST
    (18, 3): ('DOWN', 0.602, 108),  # 1pm EST
    
    # Friday slots
    (15, 4): ('DOWN', 0.602, 108),  # 10am EST
    (19, 4): ('UP', 0.620, 108),    # 2pm EST
    
    # Monday slots
    (13, 0): ('UP', 0.602, 108),    # 8am EST
}

# 10am EST playbook (15:00 UTC) - simplified version
TEN_AM_PLAYBOOK = {
    0: ('DOWN', 0.583),  # Monday
    1: ('UP', 0.611),    # Tuesday
    2: ('DOWN', 0.593),  # Wednesday
    3: ('DOWN', 0.593),  # Thursday
    4: ('DOWN', 0.602),  # Friday
}


def get_market_open_signal(hour: Optional[int] = None, 
                           dow: Optional[int] = None) -> Dict[str, Any]:
    """
    Get trading signal based on market open patterns.
    
    Args:
        hour: UTC hour (0-23), defaults to current
        dow: Day of week (0=Mon, 4=Fri), defaults to current
        
    Returns:
        {
            'has_signal': bool,
            'direction': 'UP' or 'DOWN',
            'confidence': float (0-1),
            'source': str,
            'reason': str
        }
    """
    if hour is None or dow is None:
        now = datetime.now(timezone.utc)
        hour = now.hour if hour is None else hour
        dow = now.weekday() if dow is None else dow
    
    # Skip weekends
    if dow >= 5:
        return {
            'has_signal': False,
            'direction': None,
            'confidence': 0,
            'source': 'market_open',
            'reason': 'Weekend - no market open patterns'
        }
    
    dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    
    # Check for specific hour/dow rule
    key = (hour, dow)
    if key in MARKET_OPEN_RULES:
        direction, win_rate, samples = MARKET_OPEN_RULES[key]
        est_hour = (hour - 5) % 24
        est_ampm = 'AM' if est_hour < 12 else 'PM'
        est_display = est_hour if est_hour <= 12 else est_hour - 12
        if est_display == 0:
            est_display = 12
            
        return {
            'has_signal': True,
            'direction': direction,
            'confidence': win_rate,
            'source': 'market_open',
            'reason': f"🔥 GOLDEN: {dow_names[dow]} {est_display}{est_ampm} EST = {win_rate*100:.1f}% {direction}"
        }
    
    # Check 10am EST (15:00 UTC) playbook
    if hour == 15 and dow in TEN_AM_PLAYBOOK:
        direction, win_rate = TEN_AM_PLAYBOOK[dow]
        return {
            'has_signal': True,
            'direction': direction,
            'confidence': win_rate,
            'source': '10am_playbook',
            'reason': f"✅ 10am EST {dow_names[dow]}: {direction} = {win_rate*100:.1f}% WR"
        }
    
    return {
        'has_signal': False,
        'direction': None,
        'confidence': 0,
        'source': 'market_open',
        'reason': f'No pattern for {dow_names[dow]} {hour}:00 UTC'
    }


def get_live_price_direction(symbol: str = 'BTCUSDT', 
                             minutes: int = 15) -> Dict[str, Any]:
    """
    Get current price momentum from Binance.
    
    Returns:
        {
            'current_trend': 'UP' or 'DOWN',
            'change_pct': float,
            'price': float
        }
    """
    try:
        url = 'https://api.binance.com/api/v3/klines'
        params = {
            'symbol': symbol,
            'interval': '15m',
            'limit': 2
        }
        resp = requests.get(url, params=params, timeout=5)
        candles = resp.json()
        
        if len(candles) >= 2:
            prev_close = float(candles[-2][4])
            curr_close = float(candles[-1][4])
            change_pct = (curr_close - prev_close) / prev_close * 100
            
            return {
                'current_trend': 'UP' if curr_close > prev_close else 'DOWN',
                'change_pct': change_pct,
                'price': curr_close,
                'symbol': symbol
            }
    except Exception as e:
        pass
    
    return {
        'current_trend': None,
        'change_pct': 0,
        'price': 0,
        'symbol': symbol
    }


def backtest_strategy(days: int = 60) -> Dict[str, Any]:
    """
    Backtest the market open strategy using Binance data.
    
    Returns performance metrics.
    """
    from collections import defaultdict
    
    url = 'https://api.binance.com/api/v3/klines'
    all_candles = []
    end_time = int(datetime.now().timestamp() * 1000)
    
    pages = max(1, days // 10)
    for _ in range(pages):
        params = {
            'symbol': 'BTCUSDT',
            'interval': '15m',
            'limit': 1000,
            'endTime': end_time
        }
        resp = requests.get(url, params=params, timeout=10)
        candles = resp.json()
        if not candles:
            break
        all_candles.extend(candles)
        end_time = candles[0][0] - 1
    
    # Dedupe and sort
    all_candles = list({c[0]: c for c in all_candles}.values())
    all_candles.sort(key=lambda x: x[0])
    
    # Simulate strategy
    results = {'wins': 0, 'losses': 0, 'profit': 0, 'trades': []}
    
    for c in all_candles:
        ts = datetime.fromtimestamp(c[0]/1000)
        hour = ts.hour
        dow = ts.weekday()
        
        signal = get_market_open_signal(hour, dow)
        if not signal['has_signal']:
            continue
        
        open_price = float(c[1])
        close_price = float(c[4])
        went_up = close_price > open_price
        
        bet = signal['direction']
        win = (bet == 'UP' and went_up) or (bet == 'DOWN' and not went_up)
        
        # $10 bet at 0.55 entry
        pnl = 10 * (1/0.55 - 1) if win else -10
        
        results['wins' if win else 'losses'] += 1
        results['profit'] += pnl
        results['trades'].append({
            'time': ts.isoformat(),
            'bet': bet,
            'win': win,
            'pnl': pnl
        })
    
    total = results['wins'] + results['losses']
    results['win_rate'] = results['wins'] / total if total else 0
    results['total_trades'] = total
    
    return results


if __name__ == '__main__':
    print("Market Open Strategy Module")
    print("="*60)
    
    # Current signal
    signal = get_market_open_signal()
    print(f"\nCurrent Signal:")
    print(f"  Has Signal: {signal['has_signal']}")
    print(f"  Direction: {signal['direction']}")
    print(f"  Confidence: {signal['confidence']*100:.1f}%" if signal['confidence'] else "  Confidence: N/A")
    print(f"  Reason: {signal['reason']}")
    
    # Live price
    print(f"\nLive BTC Price:")
    price_data = get_live_price_direction('BTCUSDT')
    print(f"  Trend: {price_data['current_trend']}")
    print(f"  Change: {price_data['change_pct']:+.2f}%")
    print(f"  Price: ${price_data['price']:,.2f}")
    
    # Quick backtest
    print(f"\nRunning 30-day backtest...")
    results = backtest_strategy(30)
    print(f"  Trades: {results['total_trades']}")
    print(f"  Win Rate: {results['win_rate']*100:.1f}%")
    print(f"  Profit: ${results['profit']:+.2f}")
