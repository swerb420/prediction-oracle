"""Calibration and scoring helpers for probability forecasts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass
class CalibrationResult:
    """Container for common forecast quality metrics."""

    brier: float
    log_loss: float
    num_observations: int


def brier_score(predictions: Iterable[float], outcomes: Iterable[int]) -> float:
    """Compute the mean squared error between predicted probabilities and outcomes."""

    preds = np.asarray(list(predictions), dtype=float)
    obs = np.asarray(list(outcomes), dtype=float)
    if preds.size != obs.size:
        raise ValueError("Predictions and outcomes must have the same length")
    return float(np.mean((preds - obs) ** 2))


def log_loss(predictions: Iterable[float], outcomes: Iterable[int], eps: float = 1e-15) -> float:
    """Compute negative log-likelihood with numerical stability."""

    preds = np.clip(np.asarray(list(predictions), dtype=float), eps, 1 - eps)
    obs = np.asarray(list(outcomes), dtype=float)
    if preds.size != obs.size:
        raise ValueError("Predictions and outcomes must have the same length")
    return float(-np.mean(obs * np.log(preds) + (1 - obs) * np.log(1 - preds)))


def evaluate_forecasts(predictions: Iterable[float], outcomes: Iterable[int]) -> CalibrationResult:
    """Convenience wrapper returning both Brier score and log loss."""

    preds_list = list(predictions)
    outcomes_list = list(outcomes)
    return CalibrationResult(
        brier=brier_score(preds_list, outcomes_list),
        log_loss=log_loss(preds_list, outcomes_list),
        num_observations=len(preds_list),
    )
