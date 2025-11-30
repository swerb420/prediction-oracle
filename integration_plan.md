# Full ML/Data Integration Plan

## Modules
- **data_collector.py**: Real-time API data collection for all sources
- **ml_predictor.py**: ML model for probability/edge prediction
- **trade_filter_ml.py**: Rule-based and ML-based trade filtering

## Pipeline
1. Collect market/event data using `data_collector.py`
2. Feature engineering for ML input
3. Predict edge/probability with `ml_predictor.py`
4. Filter trades using `trade_filter_ml.py`
5. Execute trades and log results
6. Use CPU for ML, offload heavy API calls

## Next Steps
- Update `multi_scanner.py` and `quick_resolve.py` to use new modules
- Train ML model with historical data
- Automate data collection and trade filtering
- Monitor performance and iterate
