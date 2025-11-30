"""
ml_predictor.py - ML model for probability/edge prediction
"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

class MLPredictor:
    def __init__(self, model_path=None):
        if model_path:
            self.model = joblib.load(model_path)
        else:
            self.model = RandomForestClassifier()
    def train(self, X, y):
        self.model.fit(X, y)
    def predict(self, features):
        return self.model.predict_proba([features])[0]
    def save(self, path):
        joblib.dump(self.model, path)
