from __future__ import annotations
import numpy as np
from collections import deque
from dataclasses import dataclass, field


_DEFAULT_WINDOW = 30
_EWMA_LAMBDA    = 0.94
_MIN_VOL        = 1e-8


@dataclass
class VolState:
    returns: deque = field(default_factory=lambda: deque(maxlen=_DEFAULT_WINDOW))
    ewma_var: float = 0.0
    realized_var_hist: list[float] = field(default_factory=list)


def _realized_vol(returns: deque) -> float:
    if len(returns) < 2:
        return _MIN_VOL
    arr = np.asarray(returns, dtype=float)
    return float(max(_MIN_VOL, np.std(arr, ddof=1)))


def _ewma_vol(ewma_var: float, r_new: float, lam: float) -> float:
    return float(lam * ewma_var + (1 - lam) * r_new ** 2)


def _parkinson_vol(highs: list[float], lows: list[float]) -> float:
    if len(highs) < 2 or len(lows) < 2 or len(highs) != len(lows):
        return _MIN_VOL
    h = np.asarray(highs, dtype=float)
    lo = np.asarray(lows, dtype=float)
    valid = h > lo
    if not valid.any():
        return _MIN_VOL
    log_hl = np.log(h[valid] / lo[valid])
    return float(max(_MIN_VOL, np.sqrt(np.mean(log_hl ** 2) / (4.0 * np.log(2)))))


def _yang_zhang_vol(opens: list[float], closes: list[float], highs: list[float], lows: list[float]) -> float:
    n = min(len(opens), len(closes), len(highs), len(lows))
    if n < 3:
        return _MIN_VOL
    o = np.asarray(opens[:n], dtype=float)
    c = np.asarray(closes[:n], dtype=float)
    h = np.asarray(highs[:n], dtype=float)
    lo = np.asarray(lows[:n], dtype=float)
    k = 0.34 / (1.34 + (n + 1) / (n - 1))
    log_co = np.log(c[1:] / o[:-1])
    log_oc = np.log(o[1:] / c[:-1])
    log_hc = np.log(h[1:] / c[:-1])
    log_lc = np.log(lo[1:] / c[:-1])
    rs = log_hc * (log_hc - log_co[: len(log_hc)]) + log_lc * (log_lc - log_co[: len(log_lc)])
    sigma_rs = float(np.mean(rs))
    sigma_oc = float(np.var(log_oc, ddof=1)) if len(log_oc) > 1 else _MIN_VOL
    sigma_co = float(np.var(log_co, ddof=1)) if len(log_co) > 1 else _MIN_VOL
    return float(max(_MIN_VOL, np.sqrt(max(0.0, sigma_oc + k * sigma_co + (1 - k) * sigma_rs))))


class VolatilityEstimator:
    def __init__(self, window: int = _DEFAULT_WINDOW, ewma_lambda: float = _EWMA_LAMBDA):
        self._window = window
        self._lam = ewma_lambda
        self._states: dict[int, VolState] = {}

    def _get(self, pid: int) -> VolState:
        if pid not in self._states:
            self._states[pid] = VolState(returns=deque(maxlen=self._window))
        return self._states[pid]

    def update(self, pid: int, x_new: float, x_prev: float) -> None:
        r = x_new - x_prev
        st = self._get(pid)
        st.returns.append(r)
        st.ewma_var = _ewma_vol(st.ewma_var, r, self._lam)

    def realized_vol(self, pid: int) -> float:
        return _realized_vol(self._get(pid).returns)

    def ewma_vol(self, pid: int) -> float:
        return float(max(_MIN_VOL, np.sqrt(self._get(pid).ewma_var)))

    def parkinson_vol(self, highs: list[float], lows: list[float]) -> float:
        return _parkinson_vol(highs, lows)

    def yang_zhang_vol(
        self,
        opens: list[float],
        closes: list[float],
        highs: list[float],
        lows: list[float],
    ) -> float:
        return _yang_zhang_vol(opens, closes, highs, lows)

    def estimate(self, pid: int, method: str = "realized") -> float:
        if method == "ewma":
            return self.ewma_vol(pid)
        return self.realized_vol(pid)

    def regime(self, pid: int, low: float = 0.05, high: float = 0.20) -> str:
        v = self.realized_vol(pid)
        if v < low:
            return "low"
        if v < high:
            return "medium"
        return "high"

    def all_vols(self) -> dict[int, float]:
        return {pid: self.realized_vol(pid) for pid in self._states}

    def snapshot(self, pid: int) -> dict:
        v = self._get(pid)
        return {
            "realized": _realized_vol(v.returns),
            "ewma": self.ewma_vol(pid),
            "n_obs": len(v.returns),
        }
