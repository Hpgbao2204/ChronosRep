from __future__ import annotations
import numpy as np
from collections import defaultdict


class TTDTracker:
    def __init__(self, attacker_ids: set[int], tau: float):
        self._attacker_ids = attacker_ids
        self._tau = tau
        self._detected: dict[int, int] = {}

    def update(self, agents, t: int) -> None:
        for a in agents:
            if a.unique_id in self._attacker_ids:
                if a.unique_id not in self._detected and a.reputation < self._tau:
                    self._detected[a.unique_id] = t

    def ttd(self) -> float | None:
        if not self._detected:
            return None
        return float(np.mean(list(self._detected.values())))

    def detection_rate(self) -> float:
        if not self._attacker_ids:
            return 0.0
        return len(self._detected) / len(self._attacker_ids)


def reputation_distribution(agents) -> dict:
    reps = np.array([a.reputation for a in agents])
    return {
        "mean":   float(reps.mean()),
        "std":    float(reps.std()),
        "p10":    float(np.percentile(reps, 10)),
        "p50":    float(np.percentile(reps, 50)),
        "p90":    float(np.percentile(reps, 90)),
        "min":    float(reps.min()),
        "max":    float(reps.max()),
    }


class MetricsCollector:
    def __init__(self):
        self._history: dict[str, list] = defaultdict(list)

    def record(self, key: str, value: float) -> None:
        self._history[key].append(value)

    def series(self, key: str) -> list:
        return self._history[key]

    def last(self, key: str, default: float = 0.0) -> float:
        h = self._history.get(key, [])
        return h[-1] if h else default
