from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from .wiener import WienerProcess
from .jump import JumpProcess

_X_LO = 0.0
_X_HI = 1.0


@dataclass
class OUParams:
    mu: float
    theta: float
    sigma: float


def _euler_maruyama_step(
    x: float,
    params: OUParams,
    dW: float,
    J: float,
    dt: float,
) -> float:
    drift = params.theta * (params.mu - x) * dt
    diff  = params.sigma * dW
    return float(np.clip(x + drift + diff + J, _X_LO, _X_HI))


def _milstein_step(
    x: float,
    params: OUParams,
    dW: float,
    J: float,
    dt: float,
) -> float:
    drift   = params.theta * (params.mu - x) * dt
    diff    = params.sigma * dW
    milstein_corr = 0.5 * params.sigma ** 2 * (dW ** 2 - dt)
    return float(np.clip(x + drift + diff + milstein_corr + J, _X_LO, _X_HI))


class SDESolver:
    def __init__(
        self,
        dt: float = 1.0,
        method: str = "euler_maruyama",
        gamma_threshold: float = 3.0,
        seed: int = 0,
    ):
        self._dt = dt
        self._method = method
        self._gamma_thr = gamma_threshold
        self._wiener = WienerProcess(dt=dt, seed=seed)
        self._jump   = JumpProcess(gamma_threshold=gamma_threshold, seed=seed + 1)
        self._x: dict[int, float] = {}

    def init(self, pid: int, x0: float) -> None:
        self._x[pid] = float(np.clip(x0, _X_LO, _X_HI))

    def step(self, pid: int, params: OUParams, gamma: float) -> float:
        if pid not in self._x:
            self.init(pid, params.mu)
        x = self._x[pid]
        dW = self._wiener.increment(pid)
        J  = self._jump.sample(pid, gamma, self._dt)
        if self._method == "milstein":
            x_new = _milstein_step(x, params, dW, J, self._dt)
        else:
            x_new = _euler_maruyama_step(x, params, dW, J, self._dt)
        self._x[pid] = x_new
        return x_new

    def batch_step(
        self,
        pids: list[int],
        params_map: dict[int, OUParams],
        gammas: dict[int, float],
    ) -> dict[int, float]:
        dWs = self._wiener.batch_increment(pids)
        result: dict[int, float] = {}
        for pid in pids:
            if pid not in self._x:
                self.init(pid, params_map[pid].mu)
            x = self._x[pid]
            p = params_map[pid]
            dW = dWs[pid]
            J = self._jump.sample(pid, gammas.get(pid, 0.0), self._dt)
            if self._method == "milstein":
                x_new = _milstein_step(x, p, dW, J, self._dt)
            else:
                x_new = _euler_maruyama_step(x, p, dW, J, self._dt)
            self._x[pid] = x_new
            result[pid] = x_new
        return result

    def trajectory(self, pid: int, params: OUParams, n_steps: int) -> list[float]:
        x = self._x.get(pid, params.mu)
        traj = [x]
        for _ in range(n_steps):
            dW = float(np.random.default_rng().standard_normal() * np.sqrt(self._dt))
            x = float(np.clip(
                x + params.theta * (params.mu - x) * self._dt + params.sigma * dW,
                _X_LO, _X_HI,
            ))
            traj.append(x)
        return traj

    def current(self, pid: int) -> float:
        return self._x.get(pid, 0.5)
