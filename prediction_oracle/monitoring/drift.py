"""Feature and label drift monitoring."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

from ..markets import Market


@dataclass
class DriftSignal:
    feature_psi: float
    volume_shift: float
    timestamp: datetime


class DriftMonitor:
    def __init__(self, window: int = 100, alert_path: str = "storage/drift_alerts.jsonl"):
        self.window = window
        self.alert_path = Path(alert_path)
        self.history = deque(maxlen=window)

    def update(self, markets: Iterable[Market]) -> DriftSignal:
        volumes = [m.volume_24h for m in markets if getattr(m, "volume_24h", None) is not None]
        avg_volume = sum(volumes) / len(volumes) if volumes else 0.0
        price_levels = [o.price for m in markets for o in m.outcomes]
        mean_price = sum(price_levels) / len(price_levels) if price_levels else 0.0

        self.history.append((avg_volume, mean_price))
        baseline_volume, baseline_price = self.history[0]

        feature_psi = abs(mean_price - baseline_price)
        volume_shift = abs(avg_volume - baseline_volume)
        signal = DriftSignal(feature_psi=feature_psi, volume_shift=volume_shift, timestamp=datetime.utcnow())
        if feature_psi > 0.1 or volume_shift > baseline_volume * 0.5:
            self._write_alert(signal)
        return signal

    def _write_alert(self, signal: DriftSignal) -> None:
        self.alert_path.parent.mkdir(parents=True, exist_ok=True)
        with self.alert_path.open("a") as f:
            f.write(json.dumps(signal.__dict__, default=str) + "\n")

