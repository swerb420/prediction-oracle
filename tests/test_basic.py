"""Tests for the prediction oracle."""

import pytest


@pytest.mark.asyncio
async def test_import():
    """Test that all modules can be imported."""
    from prediction_oracle import __version__
    from prediction_oracle.markets import Market, Venue
    from prediction_oracle.llm import LLMOracle
    from prediction_oracle.strategies import ConservativeStrategy, LongshotStrategy
    
    assert __version__ == "0.1.0"
    assert Venue.KALSHI
    assert Venue.POLYMARKET


@pytest.mark.asyncio
async def test_market_router():
    """Test market router with mock data."""
    from prediction_oracle.markets.router import MarketRouter
    from prediction_oracle.markets import Venue
    
    router = MarketRouter(mock_mode=True)
    
    # Test Kalshi client
    kalshi_client = router.get_client(Venue.KALSHI)
    markets = await kalshi_client.list_markets(limit=5)
    
    assert len(markets) == 5
    assert markets[0].venue == Venue.KALSHI
    
    # Test Polymarket client
    poly_client = router.get_client(Venue.POLYMARKET)
    markets = await poly_client.list_markets(limit=3)
    
    assert len(markets) == 3
    assert markets[0].venue == Venue.POLYMARKET
    
    await router.close_all()


@pytest.mark.asyncio
async def test_conservative_strategy():
    """Test conservative strategy market selection."""
    from prediction_oracle.markets.router import MarketRouter
    from prediction_oracle.markets import Venue
    from prediction_oracle.strategies import ConservativeStrategy
    
    router = MarketRouter(mock_mode=True)
    kalshi_client = router.get_client(Venue.KALSHI)
    markets = await kalshi_client.list_markets(limit=10)
    
    config = {
        "min_liquidity_usd": 100,
        "max_spread": 0.10,
        "implied_prob_range": [0.20, 0.75],
        "min_time_to_close_hours": 1,
    }
    
    strategy = ConservativeStrategy(config)
    selected = await strategy.select_markets(markets)
    
    # Should filter to some markets
    assert len(selected) > 0
    assert len(selected) <= len(markets)
    
    await router.close_all()


@pytest.mark.asyncio
async def test_bankroll_manager():
    """Test bankroll management."""
    from prediction_oracle.risk import BankrollManager
    
    bankroll = BankrollManager(initial_bankroll=1000.0)
    
    # Test allocation
    assert bankroll.allocate(100.0)
    assert bankroll.get_available() == 900.0
    
    # Test over-allocation
    assert not bankroll.allocate(1000.0)
    
    # Test release with profit
    bankroll.release(100.0, pnl=50.0)
    assert bankroll.current == 1050.0
    assert bankroll.total_pnl == 50.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
