from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field

_DEFAULT_LAMBDA = 0.02
_DEFAULT_MU_J   = 0.0
_DEFAULT_SIGMA_J = 0.25


@dataclass
class JumpEvent:
    t: int
    pid: int
    amplitude: float
    direction: int


@dataclass
class JumpState:
    events: list[JumpEvent] = field(default_factory=list)
    total_mass: float = 0.0
    t: int = 0


class JumpProcess:
    def __init__(
        self,
        lambda_rate: float = _DEFAULT_LAMBDA,
        mu_j: float = _DEFAULT_MU_J,
        sigma_j: float = _DEFAULT_SIGMA_J,
        gamma_threshold: float = 3.0,
        seed: int = 0,
    ):
        self._lambda = lambda_rate
        self._mu_j = mu_j
        self._sigma_j = sigma_j
        self._gamma_thr = gamma_threshold
        self._rng = np.random.default_rng(seed)
        self._states: dict[int, JumpState] = {}

    def _ensure(self, pid: int) -> JumpState:
        if pid not in self._states:
            self._states[pid] = JumpState()
        return self._states[pid]

    def sample(self, pid: int, gamma: float, dt: float = 1.0) -> float:
        s = self._ensure(pid)
        s.t += 1
        if gamma <= self._gamma_thr:
            return 0.0
        n_jumps = int(self._rng.poisson(self._lambda * dt))
        if n_jumps == 0:
            return 0.0
        amplitudes = self._rng.normal(self._mu_j, self._sigma_j, n_jumps)
        total = float(amplitudes.sum())
        for amp in amplitudes:
            s.events.append(JumpEvent(
                t=s.t,
                pid=pid,
                amplitude=float(amp),
                direction=int(np.sign(amp)),
            ))
        s.total_mass += abs(total)
        return total

    def jump_intensity(self, pid: int, window_t: int = 50) -> float:
        s = self._states.get(pid)
        if s is None or s.t == 0:
            return 0.0
        recent = [e for e in s.events if s.t - e.t <= window_t]
        return len(recent) / max(window_t, 1)

    def cumulative_jump_mass(self, pid: int) -> float:
        s = self._states.get(pid)
        return s.total_mass if s else 0.0

    def last_jump(self, pid: int) -> JumpEvent | None:
        s = self._states.get(pid)
        if s and s.events:
            return s.events[-1]
        return None
