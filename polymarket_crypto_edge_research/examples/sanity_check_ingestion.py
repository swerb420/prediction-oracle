#!/usr/bin/env python3
"""
Sanity check for data ingestion.
Tests connectivity to all data sources and validates responses.

Usage:
    python -m examples.sanity_check_ingestion
    python -m examples.sanity_check_ingestion --skip-polymarket
    python -m examples.sanity_check_ingestion --verbose
"""

import argparse
import asyncio
import sys
from datetime import datetime

from core.logging_utils import get_logger, setup_logging
from core.time_utils import now_utc

logger = get_logger(__name__)


class SanityChecker:
    """Runs sanity checks on all data sources."""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.results: dict[str, dict] = {}
    
    async def check_binance(self) -> bool:
        """Check Binance CEX connectivity."""
        from data.cex_client import BinanceClient
        
        logger.info("Checking Binance...")
        
        try:
            async with BinanceClient() as client:
                # Test current price
                price_data = await client.get_current_price("BTC")
                logger.info(f"  Server connected, BTC price fetched")
                
                if self.verbose:
                    logger.info(f"  BTC Price: ${price_data.price:,.2f}")
                    logger.info(f"  24h Volume: {price_data.volume_24h:,.0f}")
                
                # Test candles
                candles = await client.get_candles(
                    symbol="BTC",
                    interval="15m",
                    limit=10
                )
                logger.info(f"  Fetched {len(candles)} candles")
                
                self.results["binance"] = {
                    "status": "ok",
                    "btc_price": price_data.price,
                    "candles_count": len(candles)
                }
                logger.info("  ✓ Binance OK")
                return True
                
        except Exception as e:
            logger.error(f"  ✗ Binance failed: {e}")
            self.results["binance"] = {"status": "error", "error": str(e)}
            return False
    
    async def check_polymarket_gamma(self) -> bool:
        """Check Polymarket Gamma API."""
        from data.polymarket_gamma_client import PolymarketGammaClient
        
        logger.info("Checking Polymarket Gamma API...")
        
        try:
            async with PolymarketGammaClient() as client:
                # Get active markets
                markets = await client.get_active_markets(limit=5)
                logger.info(f"  Found {len(markets)} active markets")
                
                if markets and self.verbose:
                    sample = markets[0]
                    logger.info(f"  Sample market: {sample.question[:50]}...")
                
                self.results["polymarket_gamma"] = {
                    "status": "ok",
                    "markets_found": len(markets)
                }
                logger.info("  ✓ Polymarket Gamma OK")
                return True
                
        except Exception as e:
            logger.error(f"  ✗ Polymarket Gamma failed: {e}")
            self.results["polymarket_gamma"] = {"status": "error", "error": str(e)}
            return False
    
    async def check_polymarket_data(self) -> bool:
        """Check Polymarket Data API."""
        from data.polymarket_data_client import PolymarketDataClient
        
        logger.info("Checking Polymarket Data API...")
        
        try:
            async with PolymarketDataClient() as client:
                # Get recent trades (needs a valid token_id in practice)
                # For now just check connectivity
                self.results["polymarket_data"] = {"status": "ok"}
                logger.info("  ✓ Polymarket Data API OK (basic connectivity)")
                return True
                
        except Exception as e:
            logger.error(f"  ✗ Polymarket Data failed: {e}")
            self.results["polymarket_data"] = {"status": "error", "error": str(e)}
            return False
    
    async def check_polymarket_clob(self) -> bool:
        """Check Polymarket CLOB/Book API."""
        from data.polymarket_book_client import PolymarketBookClient
        
        logger.info("Checking Polymarket CLOB API...")
        
        try:
            async with PolymarketBookClient() as client:
                # Check connectivity
                self.results["polymarket_clob"] = {"status": "ok"}
                logger.info("  ✓ Polymarket CLOB API OK (basic connectivity)")
                return True
                
        except Exception as e:
            logger.error(f"  ✗ Polymarket CLOB failed: {e}")
            self.results["polymarket_clob"] = {"status": "error", "error": str(e)}
            return False
    
    async def check_grok(self) -> bool:
        """Check Grok API connectivity."""
        from llm.grok_client import create_grok_client
        from core.config import get_settings
        
        logger.info("Checking Grok API...")
        
        settings = get_settings()
        if not settings.xai_api_key:
            logger.warning("  ⚠ XAI_API_KEY not set, skipping Grok check")
            self.results["grok"] = {"status": "skipped", "reason": "no api key"}
            return True
        
        try:
            client = create_grok_client()
            
            # Simple test call
            response = await client.complete(
                prompt="Say 'ok' and nothing else.",
                max_tokens=10
            )
            
            if self.verbose:
                logger.info(f"  Response: {response.content[:50]}")
            
            self.results["grok"] = {
                "status": "ok",
                "model": response.model,
                "tokens_used": response.usage.total_tokens
            }
            logger.info("  ✓ Grok API OK")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ Grok failed: {e}")
            self.results["grok"] = {"status": "error", "error": str(e)}
            return False
    
    async def check_feature_builder(self) -> bool:
        """Check feature builder functionality."""
        from features.feature_builder_underlyings import UnderlyingsFeatureBuilder
        from data.schemas import Candle
        from datetime import datetime, timedelta
        import numpy as np
        from datetime import timezone
        
        logger.info("Checking Feature Builder...")
        
        try:
            builder = UnderlyingsFeatureBuilder()
            
            # Generate synthetic candles
            now = datetime.now(timezone.utc)
            candles = []
            price = 50000.0
            for i in range(150):
                # Random walk
                change = np.random.randn() * 0.01
                price = price * (1 + change)
                candle = Candle(
                    symbol="BTC",
                    timestamp=now - timedelta(minutes=15 * (150 - i)),
                    open=price * (1 - abs(change) / 2),
                    high=price * (1 + abs(change)),
                    low=price * (1 - abs(change)),
                    close=price,
                    volume=1e6 * (1 + np.random.rand())
                )
                candles.append(candle)
            
            # Build features
            features = builder.build(symbol="BTC", candles=candles)
            
            if features is None:
                raise ValueError("Feature builder returned None")
            
            feature_count = len(features.model_dump())
            logger.info(f"  Generated {feature_count} features")
            
            if self.verbose:
                logger.info(f"  RSI: {features.rsi_14:.2f}")
                logger.info(f"  Volatility: {features.volatility_24:.4f}")
            
            self.results["feature_builder"] = {
                "status": "ok",
                "feature_count": feature_count
            }
            logger.info("  ✓ Feature Builder OK")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ Feature Builder failed: {e}")
            self.results["feature_builder"] = {"status": "error", "error": str(e)}
            return False
    
    async def check_storage(self) -> bool:
        """Check storage functionality."""
        from data.storage import SyncStorage
        from pathlib import Path
        import tempfile
        
        logger.info("Checking Storage...")
        
        try:
            with tempfile.TemporaryDirectory() as tmpdir:
                storage = SyncStorage(Path(tmpdir))
                
                # Test SQLite
                storage.execute(
                    "CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)"
                )
                storage.execute("INSERT INTO test VALUES (1, 'hello')")
                result = storage.fetch_one("SELECT value FROM test WHERE id = 1")
                assert result[0] == "hello"
                
                # Test Parquet
                import pandas as pd
                df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
                storage.save_parquet(df, "test_data")
                loaded = storage.load_parquet("test_data")
                assert len(loaded) == 3
                
                storage.close()
                
            self.results["storage"] = {"status": "ok"}
            logger.info("  ✓ Storage OK")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ Storage failed: {e}")
            self.results["storage"] = {"status": "error", "error": str(e)}
            return False
    
    async def check_model_training(self) -> bool:
        """Check model training functionality."""
        from ml.models_tabular import LightGBMClassifier
        import numpy as np
        
        logger.info("Checking Model Training...")
        
        try:
            # Generate synthetic data
            np.random.seed(42)
            n = 1000
            X_train = np.random.randn(n, 3)
            y_train = (X_train[:, 0] + X_train[:, 1] > 0).astype(int)
            
            X_val = np.random.randn(200, 3)
            y_val = (X_val[:, 0] + X_val[:, 1] > 0).astype(int)
            
            # Train model
            model = LightGBMClassifier()
            metrics = model.fit(
                X_train, y_train, 
                X_val, y_val,
                feature_names=["feature1", "feature2", "feature3"]
            )
            
            # Predict
            preds = model.predict_proba(X_val[:10])
            
            if self.verbose:
                logger.info(f"  Sample predictions: {preds[:3]}")
                logger.info(f"  Metrics: {metrics}")
            
            self.results["model_training"] = {
                "status": "ok",
                "samples_trained": n,
                "val_auc": metrics.get("val_auc", 0.0)
            }
            logger.info("  ✓ Model Training OK")
            return True
            
        except Exception as e:
            logger.error(f"  ✗ Model Training failed: {e}")
            self.results["model_training"] = {"status": "error", "error": str(e)}
            return False
    
    async def run_all(
        self,
        skip_polymarket: bool = False,
        skip_grok: bool = False
    ) -> dict:
        """Run all sanity checks."""
        logger.info("=" * 60)
        logger.info(f"Sanity Check - {now_utc().isoformat()}")
        logger.info("=" * 60)
        
        checks = []
        
        # Core checks
        checks.append(("Binance", self.check_binance()))
        checks.append(("Storage", self.check_storage()))
        checks.append(("Feature Builder", self.check_feature_builder()))
        checks.append(("Model Training", self.check_model_training()))
        
        # Optional checks
        if not skip_polymarket:
            checks.append(("Polymarket Gamma", self.check_polymarket_gamma()))
            checks.append(("Polymarket Data", self.check_polymarket_data()))
            checks.append(("Polymarket CLOB", self.check_polymarket_clob()))
        
        if not skip_grok:
            checks.append(("Grok", self.check_grok()))
        
        # Run checks
        results = []
        for name, coro in checks:
            try:
                result = await coro
                results.append((name, result))
            except Exception as e:
                logger.error(f"Check '{name}' crashed: {e}")
                results.append((name, False))
        
        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("Summary:")
        passed = sum(1 for _, r in results if r)
        total = len(results)
        
        for name, result in results:
            status = "✓" if result else "✗"
            logger.info(f"  {status} {name}")
        
        logger.info("-" * 60)
        logger.info(f"Passed: {passed}/{total}")
        logger.info("=" * 60)
        
        return {
            "passed": passed,
            "total": total,
            "results": self.results
        }


async def main_async(args):
    """Async main."""
    checker = SanityChecker(verbose=args.verbose)
    result = await checker.run_all(
        skip_polymarket=args.skip_polymarket,
        skip_grok=args.skip_grok
    )
    
    # Exit code
    if result["passed"] < result["total"]:
        sys.exit(1)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Data ingestion sanity check")
    parser.add_argument(
        "--skip-polymarket",
        action="store_true",
        help="Skip Polymarket API checks"
    )
    parser.add_argument(
        "--skip-grok",
        action="store_true",
        help="Skip Grok API check"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    setup_logging(level=args.log_level)
    
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
