#!/usr/bin/env python3
"""Debug real Polymarket data."""
import asyncio
import sys

async def main():
    print("Starting debug...", flush=True)
    sys.stdout.flush()
    
    try:
        from prediction_oracle.markets.real_polymarket import RealPolymarketClient
        print("Imported RealPolymarketClient", flush=True)
        
        client = RealPolymarketClient()
        print("Created client", flush=True)
        
        markets = await client.list_markets(limit=10)
        print(f"Got {len(markets)} markets", flush=True)
        
        for i, m in enumerate(markets[:5]):
            price = m.outcomes[0].price if m.outcomes else 0
            print(f"{i+1}. Price: {price:.2f} | Outcomes: {len(m.outcomes)} | Q: {m.question[:50]}", flush=True)
            
    except Exception as e:
        print(f"Error: {e}", flush=True)
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Running main...", flush=True)
    asyncio.run(main())
    print("Done", flush=True)
