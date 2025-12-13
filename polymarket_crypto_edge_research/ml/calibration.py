"""
Probability calibration for ML models.
Ensures predicted probabilities match actual frequencies.
"""

from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, Field

from core.logging_utils import get_logger

logger = get_logger(__name__)


class CalibrationMetrics(BaseModel):
    """Calibration quality metrics."""
    
    # Brier score (lower is better)
    brier_score_before: float
    brier_score_after: float
    
    # Expected Calibration Error (lower is better)
    ece_before: float
    ece_after: float
    
    # Reliability diagram data
    bin_accuracies: list[float] = Field(default_factory=list)
    bin_confidences: list[float] = Field(default_factory=list)
    bin_counts: list[int] = Field(default_factory=list)
    
    @property
    def improvement(self) -> float:
        """Relative improvement in ECE."""
        if self.ece_before == 0:
            return 0.0
        return (self.ece_before - self.ece_after) / self.ece_before


class IsotonicCalibrator:
    """
    Isotonic regression calibrator.
    Non-parametric, monotonic calibration.
    Best for larger datasets.
    """
    
    def __init__(self):
        self._calibrator: Any = None
        self._is_fitted: bool = False
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> CalibrationMetrics:
        """
        Fit isotonic regression on validation set.
        
        Args:
            probabilities: Uncalibrated probabilities (n_samples,)
            labels: True labels (n_samples,)
            
        Returns:
            CalibrationMetrics
        """
        from sklearn.isotonic import IsotonicRegression
        
        # Calculate before metrics
        brier_before = self._brier_score(probabilities, labels)
        ece_before, bin_acc, bin_conf, bin_counts = self._expected_calibration_error(
            probabilities, labels
        )
        
        # Fit calibrator
        self._calibrator = IsotonicRegression(
            y_min=0.0,
            y_max=1.0,
            out_of_bounds="clip"
        )
        self._calibrator.fit(probabilities, labels)
        self._is_fitted = True
        
        # Calculate after metrics
        calibrated = self.calibrate(probabilities)
        brier_after = self._brier_score(calibrated, labels)
        ece_after, _, _, _ = self._expected_calibration_error(calibrated, labels)
        
        metrics = CalibrationMetrics(
            brier_score_before=float(brier_before),
            brier_score_after=float(brier_after),
            ece_before=float(ece_before),
            ece_after=float(ece_after),
            bin_accuracies=bin_acc,
            bin_confidences=bin_conf,
            bin_counts=bin_counts
        )
        
        logger.info(
            f"Isotonic calibration: ECE {ece_before:.4f} -> {ece_after:.4f} "
            f"({metrics.improvement:.1%} improvement)"
        )
        
        return metrics
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply calibration to probabilities."""
        if not self._is_fitted:
            raise ValueError("Calibrator not fitted")
        return self._calibrator.predict(probabilities)
    
    @staticmethod
    def _brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
        """Calculate Brier score."""
        return float(np.mean((probs - labels) ** 2))
    
    @staticmethod
    def _expected_calibration_error(
        probs: np.ndarray,
        labels: np.ndarray,
        n_bins: int = 10
    ) -> tuple[float, list[float], list[float], list[int]]:
        """
        Calculate Expected Calibration Error.
        
        Returns:
            (ECE, bin_accuracies, bin_confidences, bin_counts)
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accuracies = []
        bin_confidences = []
        bin_counts = []
        
        ece = 0.0
        total = len(probs)
        
        for i in range(n_bins):
            mask = (probs >= bin_boundaries[i]) & (probs < bin_boundaries[i + 1])
            count = mask.sum()
            
            if count > 0:
                accuracy = labels[mask].mean()
                confidence = probs[mask].mean()
                ece += (count / total) * abs(accuracy - confidence)
            else:
                accuracy = 0.0
                confidence = (bin_boundaries[i] + bin_boundaries[i + 1]) / 2
            
            bin_accuracies.append(float(accuracy))
            bin_confidences.append(float(confidence))
            bin_counts.append(int(count))
        
        return ece, bin_accuracies, bin_confidences, bin_counts
    
    def save(self, path: Path) -> None:
        """Save calibrator to disk."""
        import joblib
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self._calibrator, path.with_suffix(".joblib"))
        logger.info(f"Calibrator saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load calibrator from disk."""
        import joblib
        
        path = Path(path)
        self._calibrator = joblib.load(path.with_suffix(".joblib"))
        self._is_fitted = True
        logger.info(f"Calibrator loaded from {path}")


class PlattCalibrator:
    """
    Platt scaling calibrator.
    Parametric (logistic) calibration.
    Works well with smaller datasets.
    """
    
    def __init__(self):
        self._a: float = 1.0
        self._b: float = 0.0
        self._is_fitted: bool = False
    
    @property
    def is_fitted(self) -> bool:
        return self._is_fitted
    
    def fit(
        self,
        probabilities: np.ndarray,
        labels: np.ndarray
    ) -> CalibrationMetrics:
        """
        Fit Platt scaling on validation set.
        
        Learns parameters A, B for:
        P_calibrated = 1 / (1 + exp(A * log(p/(1-p)) + B))
        """
        from sklearn.linear_model import LogisticRegression
        
        # Calculate before metrics
        brier_before = IsotonicCalibrator._brier_score(probabilities, labels)
        ece_before, bin_acc, bin_conf, bin_counts = IsotonicCalibrator._expected_calibration_error(
            probabilities, labels
        )
        
        # Transform to log-odds
        eps = 1e-10
        probs_clipped = np.clip(probabilities, eps, 1 - eps)
        log_odds = np.log(probs_clipped / (1 - probs_clipped))
        
        # Fit logistic regression
        lr = LogisticRegression(solver="lbfgs")
        lr.fit(log_odds.reshape(-1, 1), labels)
        
        self._a = float(lr.coef_[0][0])
        self._b = float(lr.intercept_[0])
        self._is_fitted = True
        
        # Calculate after metrics
        calibrated = self.calibrate(probabilities)
        brier_after = IsotonicCalibrator._brier_score(calibrated, labels)
        ece_after, _, _, _ = IsotonicCalibrator._expected_calibration_error(calibrated, labels)
        
        metrics = CalibrationMetrics(
            brier_score_before=float(brier_before),
            brier_score_after=float(brier_after),
            ece_before=float(ece_before),
            ece_after=float(ece_after),
            bin_accuracies=bin_acc,
            bin_confidences=bin_conf,
            bin_counts=bin_counts
        )
        
        logger.info(
            f"Platt calibration: ECE {ece_before:.4f} -> {ece_after:.4f} "
            f"(A={self._a:.3f}, B={self._b:.3f})"
        )
        
        return metrics
    
    def calibrate(self, probabilities: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to probabilities."""
        if not self._is_fitted:
            raise ValueError("Calibrator not fitted")
        
        eps = 1e-10
        probs_clipped = np.clip(probabilities, eps, 1 - eps)
        log_odds = np.log(probs_clipped / (1 - probs_clipped))
        
        calibrated_logits = self._a * log_odds + self._b
        return 1 / (1 + np.exp(-calibrated_logits))
    
    def save(self, path: Path) -> None:
        """Save calibrator parameters to disk."""
        import json
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path.with_suffix(".json"), "w") as f:
            json.dump({"a": self._a, "b": self._b}, f)
        
        logger.info(f"Platt calibrator saved to {path}")
    
    def load(self, path: Path) -> None:
        """Load calibrator parameters from disk."""
        import json
        
        path = Path(path)
        with open(path.with_suffix(".json")) as f:
            params = json.load(f)
        
        self._a = params["a"]
        self._b = params["b"]
        self._is_fitted = True
        
        logger.info(f"Platt calibrator loaded from {path}")


class TemperatureScaling:
    """
    Temperature scaling for neural network calibration.
    Learns a single temperature parameter T:
    P_calibrated = softmax(logits / T)
    """
    
    def __init__(self):
        self._temperature: float = 1.0
        self._is_fitted: bool = False
    
    @property
    def temperature(self) -> float:
        return self._temperature
    
    def fit(
        self,
        logits: np.ndarray,  # Raw logits, not probabilities
        labels: np.ndarray
    ) -> CalibrationMetrics:
        """
        Fit temperature parameter using NLL minimization.
        """
        from scipy.optimize import minimize_scalar
        
        def sigmoid(x: np.ndarray) -> np.ndarray:
            return 1 / (1 + np.exp(-x))
        
        def nll(T: float) -> float:
            """Negative log-likelihood with temperature."""
            scaled = sigmoid(logits / T)
            eps = 1e-10
            scaled = np.clip(scaled, eps, 1 - eps)
            return -np.mean(labels * np.log(scaled) + (1 - labels) * np.log(1 - scaled))
        
        # Calculate before metrics
        probs_before = sigmoid(logits)
        brier_before = IsotonicCalibrator._brier_score(probs_before, labels)
        ece_before, bin_acc, bin_conf, bin_counts = IsotonicCalibrator._expected_calibration_error(
            probs_before, labels
        )
        
        # Optimize temperature
        result = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
        self._temperature = float(result.x)
        self._is_fitted = True
        
        # Calculate after metrics
        probs_after = sigmoid(logits / self._temperature)
        brier_after = IsotonicCalibrator._brier_score(probs_after, labels)
        ece_after, _, _, _ = IsotonicCalibrator._expected_calibration_error(probs_after, labels)
        
        metrics = CalibrationMetrics(
            brier_score_before=float(brier_before),
            brier_score_after=float(brier_after),
            ece_before=float(ece_before),
            ece_after=float(ece_after),
            bin_accuracies=bin_acc,
            bin_confidences=bin_conf,
            bin_counts=bin_counts
        )
        
        logger.info(
            f"Temperature scaling: T={self._temperature:.3f}, "
            f"ECE {ece_before:.4f} -> {ece_after:.4f}"
        )
        
        return metrics
    
    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        scaled_logits = logits / self._temperature
        return 1 / (1 + np.exp(-scaled_logits))


def calibrate_probabilities(
    train_probs: np.ndarray,
    train_labels: np.ndarray,
    test_probs: np.ndarray,
    method: str = "isotonic"
) -> tuple[np.ndarray, CalibrationMetrics]:
    """
    Convenience function to calibrate probabilities.
    
    Args:
        train_probs: Training set probabilities for fitting
        train_labels: Training set labels
        test_probs: Probabilities to calibrate
        method: "isotonic" or "platt"
        
    Returns:
        (calibrated_probs, metrics)
    """
    if method == "isotonic":
        calibrator = IsotonicCalibrator()
    elif method == "platt":
        calibrator = PlattCalibrator()
    else:
        raise ValueError(f"Unknown calibration method: {method}")
    
    metrics = calibrator.fit(train_probs, train_labels)
    calibrated = calibrator.calibrate(test_probs)
    
    return calibrated, metrics
