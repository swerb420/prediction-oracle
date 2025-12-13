#!/usr/bin/env python3
"""
Daily retraining script.
Fetches latest data, retrains models, and updates registry.

Usage:
    python -m scripts.daily_retrain
    python -m scripts.daily_retrain --symbols BTC ETH SOL
    python -m scripts.daily_retrain --force-retrain
"""

import argparse
import asyncio
import sys
from datetime import datetime, timedelta
from pathlib import Path

from core.config import get_settings
from core.logging_utils import get_logger, setup_logging
from core.time_utils import now_utc

logger = get_logger(__name__)


async def fetch_latest_data(
    symbols: list[str],
    days_back: int = 30
) -> dict[str, list]:
    """
    Fetch latest price data for symbols.
    
    Args:
        symbols: List of crypto symbols
        days_back: Number of days of historical data
        
    Returns:
        Dict mapping symbol to list of OHLCV data
    """
    from data.cex_client import BinanceClient
    
    logger.info(f"Fetching data for {symbols} (last {days_back} days)")
    
    async with BinanceClient() as client:
        data = {}
        
        for symbol in symbols:
            try:
                klines = await client.get_historical_klines(
                    symbol=f"{symbol}USDT",
                    interval="15m",
                    days_back=days_back
                )
                data[symbol] = klines
                logger.info(f"Fetched {len(klines)} candles for {symbol}")
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                data[symbol] = []
        
        return data


async def build_training_dataset(
    price_data: dict[str, list],
    target_horizon_minutes: int = 15
) -> tuple:
    """
    Build training dataset from price data.
    
    Args:
        price_data: Dict of symbol -> OHLCV data
        target_horizon_minutes: Prediction horizon
        
    Returns:
        Tuple of (features, targets, symbols)
    """
    import numpy as np
    import pandas as pd
    from features.feature_builder_underlyings import CryptoFeatureBuilder
    
    logger.info("Building training dataset")
    
    all_features = []
    all_targets = []
    all_symbols = []
    
    for symbol, data in price_data.items():
        if len(data) < 100:
            logger.warning(f"Insufficient data for {symbol}, skipping")
            continue
        
        # Convert to DataFrame
        df = pd.DataFrame(data, columns=[
            'open_time', 'open', 'high', 'low', 'close', 
            'volume', 'close_time', 'quote_volume', 
            'trades', 'taker_buy_base', 'taker_buy_quote', 'ignore'
        ])
        df['timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
        
        # Build features
        builder = CryptoFeatureBuilder()
        
        for i in range(50, len(df) - target_horizon_minutes // 15):
            window = df.iloc[:i+1]
            
            try:
                features = builder.build_features(
                    symbol=symbol,
                    current_price=float(window.iloc[-1]['close']),
                    high_24h=float(window.tail(96)['high'].max()),
                    low_24h=float(window.tail(96)['low'].min()),
                    volume_24h=float(window.tail(96)['volume'].sum()),
                    change_24h_pct=float(
                        (window.iloc[-1]['close'] - window.iloc[-97]['close']) 
                        / window.iloc[-97]['close'] * 100
                    ) if len(window) > 97 else 0.0,
                    prices_1h=window.tail(4)['close'].tolist(),
                    volumes_1h=window.tail(4)['volume'].tolist()
                )
                
                # Target: price direction after horizon
                future_price = df.iloc[i + target_horizon_minutes // 15]['close']
                current_price = window.iloc[-1]['close']
                target = 1 if future_price > current_price else 0
                
                all_features.append(features.model_dump())
                all_targets.append(target)
                all_symbols.append(symbol)
                
            except Exception as e:
                continue
    
    logger.info(f"Built {len(all_features)} training samples")
    
    return all_features, all_targets, all_symbols


async def train_models(
    features: list[dict],
    targets: list[int],
    symbols: list[str],
    output_dir: Path
) -> dict[str, float]:
    """
    Train models for each symbol.
    
    Args:
        features: List of feature dicts
        targets: List of targets
        symbols: List of symbols
        output_dir: Output directory for models
        
    Returns:
        Dict mapping symbol to validation AUC
    """
    import numpy as np
    import pandas as pd
    from ml.trainer import Trainer
    from ml.models_tabular import TabularModel
    from ml.model_registry import ModelRegistry
    
    logger.info("Training models")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = ModelRegistry(output_dir)
    
    results = {}
    unique_symbols = list(set(symbols))
    
    for symbol in unique_symbols:
        logger.info(f"Training model for {symbol}")
        
        # Filter data for this symbol
        symbol_mask = [s == symbol for s in symbols]
        symbol_features = [f for f, m in zip(features, symbol_mask) if m]
        symbol_targets = [t for t, m in zip(targets, symbol_mask) if m]
        
        if len(symbol_features) < 100:
            logger.warning(f"Insufficient samples for {symbol}, skipping")
            continue
        
        try:
            # Convert to DataFrame
            X = pd.DataFrame(symbol_features)
            y = np.array(symbol_targets)
            
            # Train/val split
            split_idx = int(len(X) * 0.8)
            X_train, X_val = X.iloc[:split_idx], X.iloc[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
            
            # Train model
            model = TabularModel()
            model.train(X_train, y_train, symbol)
            
            # Evaluate
            from sklearn.metrics import roc_auc_score
            y_pred = model.predict_proba(X_val)
            if len(y_pred.shape) > 1:
                y_pred = y_pred[:, 1]
            
            auc = roc_auc_score(y_val, y_pred)
            logger.info(f"{symbol} validation AUC: {auc:.4f}")
            
            # Save to registry if good enough
            if auc > 0.52:  # Better than random
                model_path = output_dir / f"{symbol}_model.pkl"
                model.save(model_path)
                
                # Register
                registry.register_model(
                    symbol=symbol,
                    model_path=model_path,
                    metrics={"auc": auc, "samples": len(symbol_features)},
                    promote_if_better=True
                )
                
            results[symbol] = auc
            
        except Exception as e:
            logger.error(f"Training failed for {symbol}: {e}")
            results[symbol] = 0.0
    
    return results


async def run_daily_retrain(
    symbols: list[str],
    days_back: int = 30,
    force_retrain: bool = False,
    output_dir: Path = Path("models")
) -> None:
    """
    Run daily retraining pipeline.
    
    Args:
        symbols: Symbols to train
        days_back: Days of historical data
        force_retrain: Force retrain even if recent model exists
        output_dir: Output directory
    """
    from ml.model_registry import ModelRegistry
    
    logger.info("=" * 60)
    logger.info(f"Starting daily retrain at {now_utc().isoformat()}")
    logger.info(f"Symbols: {symbols}")
    logger.info("=" * 60)
    
    # Check if retrain needed
    registry = ModelRegistry(output_dir)
    
    if not force_retrain:
        all_fresh = True
        for symbol in symbols:
            metadata = registry.get_champion_metadata(symbol)
            if metadata:
                trained_at = datetime.fromisoformat(metadata.get("trained_at", "2000-01-01"))
                age_hours = (now_utc() - trained_at).total_seconds() / 3600
                if age_hours > 24:
                    all_fresh = False
                    break
            else:
                all_fresh = False
                break
        
        if all_fresh:
            logger.info("All models are fresh (< 24h old), skipping retrain")
            return
    
    # Fetch data
    price_data = await fetch_latest_data(symbols, days_back)
    
    # Build dataset
    features, targets, symbol_labels = await build_training_dataset(price_data)
    
    if not features:
        logger.error("No training data built, aborting")
        return
    
    # Train models
    results = await train_models(features, targets, symbol_labels, output_dir)
    
    # Summary
    logger.info("=" * 60)
    logger.info("Retrain Summary:")
    for symbol, auc in results.items():
        status = "✓" if auc > 0.52 else "✗"
        logger.info(f"  {status} {symbol}: AUC = {auc:.4f}")
    logger.info("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Daily model retraining")
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTC", "ETH", "SOL"],
        help="Symbols to train"
    )
    parser.add_argument(
        "--days-back",
        type=int,
        default=30,
        help="Days of historical data"
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retrain even if models are fresh"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Output directory for models"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level=args.log_level)
    
    # Run
    try:
        asyncio.run(run_daily_retrain(
            symbols=args.symbols,
            days_back=args.days_back,
            force_retrain=args.force_retrain,
            output_dir=args.output_dir
        ))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
