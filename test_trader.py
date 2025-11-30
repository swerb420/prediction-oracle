#!/usr/bin/env python3
"""Quick test of the trading system."""
print("=" * 60, flush=True)
print("PREDICTION ORACLE - PAPER TRADER TEST", flush=True)
print("=" * 60, flush=True)

import sys
import os
sys.path.insert(0, '/root/prediction_oracle')
os.chdir('/root/prediction_oracle')

print("Importing modules...", flush=True)

import asyncio
from datetime import datetime, timezone

# Test imports
try:
    from prediction_oracle.config import settings
    print(f"✓ Config loaded", flush=True)
    print(f"  - Kalshi API: {'SET' if settings.kalshi_api_key != 'your_kalshi_key' else 'NOT SET'}", flush=True)
    print(f"  - OpenAI API: {'SET' if settings.openai_api_key != 'your_openai_key' else 'NOT SET'}", flush=True)
except Exception as e:
    print(f"✗ Config error: {e}", flush=True)

try:
    from prediction_oracle.markets.real_polymarket import RealPolymarketClient
    print(f"✓ Polymarket client imported", flush=True)
except Exception as e:
    print(f"✗ Polymarket import error: {e}", flush=True)

try:
    from prediction_oracle.markets.real_kalshi import RealKalshiClient
    print(f"✓ Kalshi client imported", flush=True)
except Exception as e:
    print(f"✗ Kalshi import error: {e}", flush=True)

async def test_markets():
    print("\n" + "=" * 60, flush=True)
    print("FETCHING REAL MARKET DATA", flush=True)
    print("=" * 60, flush=True)
    
    # Test Polymarket
    print("\n[Polymarket - FREE API]", flush=True)
    try:
        poly = RealPolymarketClient()
        markets = await poly.list_markets(limit=10)
        print(f"✓ Got {len(markets)} markets", flush=True)
        
        for i, m in enumerate(markets[:5], 1):
            price = m.outcomes[0].price if m.outcomes else 0
            print(f"  {i}. {price:.0%} | {m.question[:50]}", flush=True)
        
        await poly.close()
    except Exception as e:
        print(f"✗ Polymarket error: {e}", flush=True)
    
    # Test Kalshi
    print("\n[Kalshi]", flush=True)
    try:
        kalshi = RealKalshiClient(demo_mode=True)
        if kalshi.has_credentials:
            print("✓ Has real API credentials", flush=True)
        else:
            print("⚠ No API credentials - using mock data", flush=True)
        
        markets = await kalshi.list_markets(limit=5)
        print(f"✓ Got {len(markets)} markets", flush=True)
        
        for i, m in enumerate(markets[:3], 1):
            price = m.outcomes[0].price if m.outcomes else 0
            print(f"  {i}. {price:.0%} | {m.question[:50]}", flush=True)
        
        await kalshi.close()
    except Exception as e:
        print(f"✗ Kalshi error: {e}", flush=True)

print("\nRunning async tests...", flush=True)
asyncio.run(test_markets())

print("\n" + "=" * 60, flush=True)
print("TEST COMPLETE", flush=True)
print("=" * 60, flush=True)
