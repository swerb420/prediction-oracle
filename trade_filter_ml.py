"""
trade_filter_ml.py - Rule-based and ML-based trade filtering
"""
from ml_predictor import MLPredictor

class TradeFilter:
    def __init__(self, model_path=None):
        self.predictor = MLPredictor(model_path)
    def filter(self, trade_data):
        # trade_data: dict with features
        prob = self.predictor.predict(trade_data['features'])
        # Example rule: Only trade if ML edge > 0.6
        if prob[1] > 0.6:
            return True
        return False
