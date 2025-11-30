#!/usr/bin/env python3
import urllib.request, json, sqlite3
from datetime import datetime, timedelta

db = sqlite3.connect('/root/prediction_oracle/paper_trades.db')
c = db.cursor()

c.execute("SELECT market_id, question, bet_size, direction FROM trades WHERE source='MANIFOLD' AND resolved_at IS NULL")
manifold_trades = c.fetchall()

c.execute("SELECT source, question, bet_size, direction FROM trades WHERE source IN ('ESPN_NFL','WEATHER') AND resolved_at IS NULL")
other_trades = c.fetchall()

print("=" * 60)
print("TRADES ENDING IN NEXT 12 HOURS")
print("=" * 60)

now = datetime.now()
ending_soon = []

for mid, q, bet, direction in manifold_trades:
    try:
        d = json.loads(urllib.request.urlopen(f'https://api.manifold.markets/v0/market/{mid}', timeout=8).read())
        ct = d.get('closeTime', 0)
        close_dt = datetime.fromtimestamp(ct/1000)
        hours_left = (close_dt - now).total_seconds() / 3600
        
        if 0 < hours_left <= 12:
            ending_soon.append((hours_left, bet, direction, q, 'MANIFOLD'))
    except:
        pass

for source, q, bet, direction in other_trades:
    if 'Ravens' in q:
        ending_soon.append((2.0, bet, direction, q, source))
    elif 'WEATHER' in source:
        ending_soon.append((6.0, bet, direction, q, source))

ending_soon.sort(key=lambda x: x[0])

total_bet = 0
for hours, bet, direction, q, source in ending_soon:
    total_bet += bet
    print(f"{hours:5.1f}h | ${bet:>3.0f} {direction:<3} | {source:<10} | {q[:35]}...")

print()
print(f"TOTAL: {len(ending_soon)} bets | ${total_bet:.0f} at stake")
db.close()
