from __future__ import annotations
import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np

_CUSUM_H     = 4.0
_CUSUM_K     = 0.5
_ZSCORE_WIN  = 30
_ENTROPY_EPS = 1e-9


@dataclass
class _StreamState:
    window: deque = field(default_factory=lambda: deque(maxlen=_ZSCORE_WIN))
    cusum_pos: float = 0.0
    cusum_neg: float = 0.0
    alarm_count: int = 0
    last_entropy: float = 0.0
    t: int = 0


def _shannon_entropy(probs: np.ndarray) -> float:
    p = probs[probs > _ENTROPY_EPS]
    return float(-np.sum(p * np.log2(p)))


def _outcome_distribution(window: deque) -> np.ndarray:
    arr = np.array(list(window), dtype=float)
    if len(arr) == 0:
        return np.array([0.5, 0.5])
    p1 = arr.mean()
    p0 = 1.0 - p1
    return np.array([p0, p1])


def _zscore_normalize(value: float, window: deque) -> float:
    if len(window) < 2:
        return 0.0
    arr = np.array(list(window), dtype=float)
    mu, sigma = arr.mean(), arr.std()
    if sigma < _ENTROPY_EPS:
        return 0.0
    return (value - mu) / sigma


def _cusum_update(
    z: float,
    s_pos: float,
    s_neg: float,
    k: float = _CUSUM_K,
) -> tuple[float, float, bool]:
    s_pos = max(0.0, s_pos + z - k)
    s_neg = max(0.0, s_neg - z - k)
    alarm = (s_pos > _CUSUM_H) or (s_neg > _CUSUM_H)
    if alarm:
        s_pos = 0.0
        s_neg = 0.0
    return s_pos, s_neg, alarm


class BSM:
    def __init__(self, cusum_h: float = _CUSUM_H, cusum_k: float = _CUSUM_K):
        self._h = cusum_h
        self._k = cusum_k
        self._states: dict[int, _StreamState] = {}

    def _ensure(self, agent_id: int) -> _StreamState:
        if agent_id not in self._states:
            self._states[agent_id] = _StreamState()
        return self._states[agent_id]

    def monitor(self, agent_id: int, behavior_stream: list[int]) -> dict:
        s = self._ensure(agent_id)
        alarms: list[int] = []

        for outcome in behavior_stream:
            s.window.append(float(outcome))
            z = _zscore_normalize(float(outcome), s.window)
            s.cusum_pos, s.cusum_neg, alarm = _cusum_update(z, s.cusum_pos, s.cusum_neg, self._k)
            if alarm:
                s.alarm_count += 1
                alarms.append(s.t)
            s.t += 1

        dist = _outcome_distribution(s.window)
        h = _shannon_entropy(dist)
        s.last_entropy = h

        arr = np.array(list(s.window), dtype=float)
        mean_outcome = float(arr.mean()) if len(arr) > 0 else 0.5
        anomaly_score = float(s.alarm_count) / max(s.t, 1)

        return {
            "agent_id":      agent_id,
            "mean_outcome":  mean_outcome,
            "entropy":       h,
            "anomaly_score": anomaly_score,
            "alarm_steps":   alarms,
            "cusum_pos":     s.cusum_pos,
            "cusum_neg":     s.cusum_neg,
            "t":             s.t,
        }

    def anomaly_score(self, agent_id: int) -> float:
        s = self._states.get(agent_id)
        if s is None or s.t == 0:
            return 0.0
        return s.alarm_count / s.t
