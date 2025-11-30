#!/usr/bin/env python3
import urllib.request, json, sqlite3
from datetime import datetime

db = sqlite3.connect('paper_trades.db')
c = db.cursor()
c.execute("SELECT market_id, question, bet_size, direction FROM trades WHERE source='MANIFOLD' AND resolved_at IS NULL ORDER BY bet_size DESC LIMIT 10")

print("Checking actual close dates for our bets...")
print("=" * 60)

for mid, q, bet, direction in c.fetchall():
    try:
        d = json.loads(urllib.request.urlopen(f'https://api.manifold.markets/v0/market/{mid}', timeout=10).read())
        ct = d.get('closeTime', 0)
        close_dt = datetime.fromtimestamp(ct/1000)
        hours_left = (close_dt - datetime.now()).total_seconds() / 3600
        
        if hours_left < 24:
            status = "SOON <24h"
        elif hours_left < 72:
            status = "3 days"
        else:
            status = f"{hours_left/24:.0f} days"
        
        print(f"${bet} {direction} | {status} | {q[:40]}...")
        print(f"   Closes: {close_dt.strftime('%Y-%m-%d %H:%M')}")
    except Exception as e:
        print(f"Error {mid}: {e}")

db.close()
