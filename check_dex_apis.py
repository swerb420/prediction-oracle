#!/usr/bin/env python3
"""
DEX Prediction Markets API Status Checker
Tests all available prediction market APIs and shows which ones work
"""

import requests
import json
from datetime import datetime

def test_polymarket():
    """Test Polymarket API"""
    try:
        resp = requests.get(
            'https://gamma-api.polymarket.com/markets',
            params={'limit': 3, 'active': 'true', 'closed': 'false'},
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            return True, len(data), data[0]['question'][:50] if data else 'N/A'
        return False, 0, f"Status: {resp.status_code}"
    except Exception as e:
        return False, 0, str(e)[:50]

def test_manifold():
    """Test Manifold Markets API"""
    try:
        resp = requests.get(
            'https://manifold.markets/api/v0/markets',
            params={'limit': 3},
            timeout=15,
            allow_redirects=True
        )
        if resp.status_code == 200:
            data = resp.json()
            return True, len(data), data[0]['question'][:50] if data else 'N/A'
        return False, 0, f"Status: {resp.status_code}"
    except Exception as e:
        return False, 0, str(e)[:50]

def test_metaculus():
    """Test Metaculus API"""
    try:
        resp = requests.get(
            'https://www.metaculus.com/api2/questions/',
            params={'limit': 3},
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            results = data.get('results', [])
            return True, len(results), results[0].get('title', 'N/A')[:50] if results else 'N/A'
        return False, 0, f"Status: {resp.status_code}"
    except Exception as e:
        return False, 0, str(e)[:50]

def test_espn_nfl():
    """Test ESPN NFL API"""
    try:
        resp = requests.get(
            'https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard',
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            events = data.get('events', [])
            return True, len(events), events[0].get('name', 'N/A')[:50] if events else 'No games today'
        return False, 0, f"Status: {resp.status_code}"
    except Exception as e:
        return False, 0, str(e)[:50]

def test_espn_nba():
    """Test ESPN NBA API"""
    try:
        resp = requests.get(
            'https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard',
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            events = data.get('events', [])
            return True, len(events), events[0].get('name', 'N/A')[:50] if events else 'No games today'
        return False, 0, f"Status: {resp.status_code}"
    except Exception as e:
        return False, 0, str(e)[:50]

def test_kalshi():
    """Test Kalshi API"""
    try:
        resp = requests.get(
            'https://trading-api.kalshi.com/trade-api/v2/markets',
            params={'limit': 3},
            timeout=15
        )
        if resp.status_code == 200:
            text = resp.text
            if 'moved' in text.lower():
                return False, 0, "API migrated to elections.kalshi.com"
            data = resp.json()
            return True, len(data.get('markets', [])), 'Working'
        return False, 0, f"Status: {resp.status_code}"
    except Exception as e:
        return False, 0, str(e)[:50]

def test_azuro():
    """Test Azuro GraphQL API"""
    try:
        query = '''
        {
            games(first: 3, where: { status: Created }) {
                id
                title
            }
        }
        '''
        resp = requests.post(
            'https://thegraph.onchainfeed.org/subgraphs/name/azuro-protocol/azuro-api-polygon-v3',
            json={'query': query},
            timeout=15
        )
        if resp.status_code == 200:
            data = resp.json()
            games = data.get('data', {}).get('games', [])
            return True, len(games), games[0].get('title', 'N/A')[:50] if games else 'No games'
        return False, 0, f"Status: {resp.status_code}"
    except Exception as e:
        return False, 0, str(e)[:50]

def main():
    print("=" * 70)
    print("🔮 DEX PREDICTION MARKETS API STATUS CHECK")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    apis = [
        ("POLYMARKET", "REAL $", "🟢", test_polymarket),
        ("MANIFOLD", "PAPER", "🔵", test_manifold),
        ("METACULUS", "NONE", "⚪", test_metaculus),
        ("ESPN NFL", "DATA", "🟢", test_espn_nfl),
        ("ESPN NBA", "DATA", "🟢", test_espn_nba),
        ("KALSHI", "REAL $", "🟡", test_kalshi),
        ("AZURO", "REAL $", "🟡", test_azuro),
    ]
    
    results = []
    
    for name, money_type, icon, test_func in apis:
        print(f"\n{icon} Testing {name}...", end=" ", flush=True)
        working, count, sample = test_func()
        status = "✅ WORKING" if working else "❌ FAILED"
        results.append((name, money_type, working, count, sample))
        print(status)
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 SUMMARY")
    print("=" * 70)
    
    print(f"\n{'Platform':<15} {'Money':<10} {'Status':<10} {'Markets':<10} {'Sample'}")
    print("-" * 70)
    
    for name, money_type, working, count, sample in results:
        status = "✅ OK" if working else "❌ FAIL"
        print(f"{name:<15} {money_type:<10} {status:<10} {count:<10} {sample[:35]}...")
    
    # Working count
    working_count = sum(1 for r in results if r[2])
    print("\n" + "-" * 70)
    print(f"🎯 {working_count}/{len(results)} APIs working")
    
    # Recommendations
    print("\n📌 RECOMMENDATIONS FOR YOUR ORACLE:")
    print("-" * 70)
    
    real_money = [(n, s) for n, m, w, c, s in results if m == "REAL $" and w]
    paper = [(n, s) for n, m, w, c, s in results if m in ["PAPER", "NONE"] and w]
    data = [(n, s) for n, m, w, c, s in results if m == "DATA" and w]
    
    if real_money:
        print(f"💰 Real Money Trading: {', '.join(n for n, s in real_money)}")
    if paper:
        print(f"📝 Paper Trading/Forecasting: {', '.join(n for n, s in paper)}")
    if data:
        print(f"📊 Sports Data Sources: {', '.join(n for n, s in data)}")
    
    print("\n" + "=" * 70)
    print("📖 See DEX_PREDICTION_MARKETS_API_GUIDE.md for full documentation")
    print("=" * 70)

if __name__ == "__main__":
    main()
