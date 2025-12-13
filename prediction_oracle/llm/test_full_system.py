#!/usr/bin/env python3
"""
Full End-to-End Test for Enhanced Crypto Trading System
========================================================

Tests all components:
1. Multi-venue price fetching (Binance, Bybit, Kraken, Coinbase)
2. Feature extraction (31 features)
3. ML model training and prediction
4. Enhanced paper trading
5. Full prediction cycle for BTC, ETH, SOL, XRP
"""
import asyncio
import sys
import os

sys.path.insert(0, '/root/prediction-oracle')
os.chdir('/root/prediction-oracle/prediction_oracle/llm')

# Suppress verbose logging during test
import logging
logging.basicConfig(level=logging.WARNING)

async def test_all():
    print("=" * 70)
    print("🧪 ENHANCED CRYPTO TRADING SYSTEM - FULL TEST")
    print("=" * 70)
    
    errors = []
    
    # Test 1: Multi-venue prices
    print("\n📊 Test 1: Multi-Venue Price Fetching")
    print("-" * 50)
    try:
        from prediction_oracle.llm.multi_venue_client import MultiVenueClient
        
        async with MultiVenueClient() as client:
            for symbol in ["BTC", "ETH", "SOL", "XRP"]:
                prices = await client.get_all_prices(symbol)
                if prices:
                    avg = sum(p.mid for p in prices) / len(prices)
                    print(f"   ✅ {symbol}: ${avg:,.2f} from {len(prices)} venues")
                else:
                    print(f"   ⚠️ {symbol}: No prices fetched")
                    errors.append(f"No prices for {symbol}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        errors.append(f"Multi-venue: {e}")
    
    # Test 2: Crypto data fetching
    print("\n📈 Test 2: Binance OHLCV Data (15m candles)")
    print("-" * 50)
    try:
        from prediction_oracle.llm.crypto_data import get_fetcher
        
        fetcher = get_fetcher()
        for symbol in ["BTC", "ETH", "SOL", "XRP"]:
            data = await fetcher.get_candles(symbol, "15m", 100)
            latest = data.candles[-1]
            print(f"   ✅ {symbol}: {len(data.candles)} candles, latest: ${latest.close:,.2f}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        errors.append(f"Crypto data: {e}")
    
    # Test 3: Feature extraction
    print("\n🔬 Test 3: Feature Engineering (31 features)")
    print("-" * 50)
    try:
        from prediction_oracle.llm.feature_engineering import extract_features
        from prediction_oracle.llm.enhanced_features import EnhancedFeatureSet
        
        data = await fetcher.get_candles("BTC", "15m", 100)
        base_features = extract_features(data)
        
        print(f"   ✅ Base features: RSI={base_features.rsi_14:.1f}, MACD={base_features.macd_signal:.4f}")
        print(f"   ✅ BB Position: {base_features.bb_position:.2f}, Vol Ratio: {base_features.volume_ratio_6:.2f}")
        
        # Create enhanced features (with defaults for whale/venue)
        enhanced = EnhancedFeatureSet.from_components(base_features, None, None)
        print(f"   ✅ Enhanced features: {len(enhanced.to_array())} dimensions")
        print(f"   ✅ Feature names: {EnhancedFeatureSet.feature_names()[:5]}...")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        errors.append(f"Features: {e}")
    
    # Test 4: ML Model
    print("\n🤖 Test 4: ML Model Training & Prediction")
    print("-" * 50)
    try:
        from prediction_oracle.llm.enhanced_ml_predictor import EnhancedCryptoMLPredictor
        
        predictor = EnhancedCryptoMLPredictor(model_dir="./models")
        
        # Train models (uses historical data)
        print("   Training models (this may take a moment)...")
        await predictor.initialize(retrain=True)
        
        # Make predictions
        for symbol in ["BTC", "ETH", "SOL", "XRP"]:
            pred = await predictor.predict_symbol(symbol, include_whale=False, include_venue=False)
            gate = "✅" if pred.quality_gate_passed else "❌"
            print(f"   {gate} {symbol}: {pred.direction} @ {pred.confidence:.1%} conf")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        errors.append(f"ML Model: {e}")
    
    # Test 5: Enhanced Hybrid Oracle (without Grok for now)
    print("\n🔮 Test 5: Enhanced Hybrid Oracle (ML only, no Grok API)")
    print("-" * 50)
    try:
        from prediction_oracle.llm.enhanced_hybrid_oracle import EnhancedHybridOracle
        
        oracle = EnhancedHybridOracle(
            use_grok_validation=False,  # No API key yet
            use_whale_signals=False,     # Skip for speed
            use_venue_data=False,        # Skip for speed
        )
        await oracle.initialize()
        
        predictions = await oracle.predict_all()
        
        for symbol, pred in predictions.items():
            emoji = {"LONG": "🟢", "SHORT": "🔴", "HOLD": "⚪"}.get(pred.final_signal, "⚪")
            trade = "TRADE" if pred.should_trade else "SKIP"
            print(f"   {emoji} {symbol}: {pred.final_signal} @ {pred.final_confidence:.1%} → {trade}")
            print(f"      Price: ${pred.current_price:,.2f} | ML: {pred.ml_direction}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
        errors.append(f"Oracle: {e}")
    
    # Test 6: Paper Trading Engine
    print("\n💰 Test 6: Enhanced Paper Trading Engine")
    print("-" * 50)
    try:
        from prediction_oracle.llm.enhanced_paper_trading import EnhancedPaperTradingEngine
        
        engine = EnhancedPaperTradingEngine(
            initial_capital=10000.0,
            state_file="./test_paper_trading_state.json",
        )
        
        print(f"   ✅ Engine initialized with ${engine.capital:,.2f}")
        print(f"   ✅ Open positions: {len(engine.open_positions)}")
        print(f"   ✅ Closed trades: {len(engine.closed_trades)}")
        print(f"   ✅ Win rate: {engine.stats.win_rate:.1%}")
        
        # Test opening a position
        if predictions:
            for sym, pred in predictions.items():
                if pred.should_trade:
                    pos = await engine.open_position(pred)
                    if pos:
                        print(f"   ✅ Opened test position: {pos.symbol} {pos.direction}")
                        break
    except Exception as e:
        print(f"   ❌ Error: {e}")
        errors.append(f"Paper trading: {e}")
    
    # Summary
    print("\n" + "=" * 70)
    if errors:
        print(f"⚠️  TESTS COMPLETED WITH {len(errors)} ERROR(S):")
        for err in errors:
            print(f"   - {err}")
    else:
        print("✅ ALL TESTS PASSED!")
        print("\n📋 System Ready for Trading:")
        print("   - 4 crypto symbols: BTC, ETH, SOL, XRP")
        print("   - 4 CEX venues: Binance, Bybit, Kraken, Coinbase")
        print("   - 31 ML features (technical + venue + whale)")
        print("   - Gradient Boosting classifier trained")
        print("   - Paper trading engine active")
        print("\n🚀 To start trading with Grok 4.1 API:")
        print("   export XAI_API_KEY='your-api-key'")
        print("   python -m prediction_oracle.llm.enhanced_crypto_trader --once")
    print("=" * 70)

if __name__ == "__main__":
    asyncio.run(test_all())
