from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field


@dataclass
class WienerState:
    path: list[float] = field(default_factory=list)
    cumulative: float = 0.0
    t: int = 0


class WienerProcess:
    def __init__(self, dt: float = 1.0, seed: int = 0):
        self._dt = dt
        self._rng = np.random.default_rng(seed)
        self._states: dict[int, WienerState] = {}

    def _ensure(self, pid: int) -> WienerState:
        if pid not in self._states:
            self._states[pid] = WienerState()
        return self._states[pid]

    def increment(self, pid: int) -> float:
        s = self._ensure(pid)
        dW = float(self._rng.standard_normal() * np.sqrt(self._dt))
        s.path.append(dW)
        s.cumulative += dW
        s.t += 1
        return dW

    def batch_increment(self, pids: list[int]) -> dict[int, float]:
        n = len(pids)
        if n == 0:
            return {}
        dWs = self._rng.standard_normal(n) * np.sqrt(self._dt)
        result: dict[int, float] = {}
        for i, pid in enumerate(pids):
            s = self._ensure(pid)
            dW = float(dWs[i])
            s.path.append(dW)
            s.cumulative += dW
            s.t += 1
            result[pid] = dW
        return result

    def realized_variance(self, pid: int, window: int = 20) -> float:
        s = self._states.get(pid)
        if s is None or len(s.path) < 2:
            return 0.0
        arr = np.array(s.path[-window:])
        return float(np.var(arr))

    def quadratic_variation(self, pid: int) -> float:
        s = self._states.get(pid)
        if s is None:
            return 0.0
        arr = np.array(s.path)
        return float(np.sum(arr ** 2))
