#!/usr/bin/env python3
"""Quick test of enhanced modules."""
import asyncio
import sys
sys.path.insert(0, '/root/prediction-oracle')

from prediction_oracle.llm.multi_venue_client import MultiVenueClient

async def test():
    print('🔍 Testing multi-venue price fetch for BTC...')
    async with MultiVenueClient() as client:
        prices = await client.get_all_prices('BTC')
        print(f'   Got {len(prices)} venue prices:')
        for p in prices:
            print(f'   - {p.venue}: ${p.mid:,.2f} (spread: {p.spread_bps:.1f} bps)')
        
        if prices:
            features = client.compute_cross_venue_features(prices)
            if features:
                print(f'   Cross-venue avg: ${features.avg_mid_price:,.2f}')
                print(f'   Max arb spread: {features.max_arb_spread_bps:.1f} bps')
    print('✅ Multi-venue client works!')

if __name__ == "__main__":
    asyncio.run(test())
