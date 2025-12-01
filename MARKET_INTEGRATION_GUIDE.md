# Adding new prediction market venues

This project exposes a venue-agnostic interface so you can plug in additional markets (e.g., RainBet or other API-driven venues) without refactoring trading or research workflows. This guide walks through the recommended steps and patterns.

## 1) Implement a client that satisfies `BaseMarketClient`

Create a new module under `markets/` (for example `markets/rainbet_client.py`) with a class that subclasses `BaseMarketClient` and implements:

- `list_markets(category=None, min_volume=None, limit=None)` to return normalized `Market` objects.
- `get_market(market_id)` for detailed fetching.
- `get_positions()` for open positions.
- `place_order(request)` and `cancel_order(order_id)` for execution.
- `close()` to clean up session resources.

Use the shared data models in `markets/__init__.py` (`Market`, `Outcome`, `OrderRequest`, `OrderResult`, `Position`) to keep downstream consumers consistent.

### Normalization tips
- Convert venue-specific price formats into 0–1 probabilities for `Outcome.price`.
- Populate `volume_24h` and `liquidity` when available so existing filters continue working.
- Map resolution or close timestamps into timezone-aware `datetime` objects.

## 2) Register the client with the router

`MarketRouter` now supports pluggable clients via `register_client` and the `extra_clients` constructor argument. You can register your client at runtime without editing the router itself:

```python
from markets import Venue
from markets.router import MarketRouter
from markets.rainbet_client import RainBetClient

router = MarketRouter(
    mock_mode=False,
    extra_clients={Venue("RAINBET"): RainBetClient(api_key="...", mock_mode=False)},
)

rainbet_client = router.get_client(Venue("RAINBET"))
```

If you prefer a tighter integration, you can also extend the `Venue` enum in `markets/__init__.py` with your venue identifier and import the new client in `markets/router.py`.

## 3) Wire research/backtesting data

For offline analysis, add lightweight fetchers similar to the helpers in `extra_data_sources.py`. When your venue exposes historical data, normalize it to the same schema used by the strategy layer (market metadata plus price/time series) so you can reuse existing notebooks or scripts without special casing.

## 4) Testing checklist

- Unit-test price/volume normalization and serialization into `Market`/`Outcome`.
- Mock API responses for order placement/cancellation paths.
- Validate `MarketRouter.supported_venues` includes your new venue when registered.
- Run a smoke test in `mock_mode` to ensure trading loops can execute without live API calls.
