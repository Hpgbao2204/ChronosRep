from __future__ import annotations
import math
from dataclasses import dataclass
import numpy as np

_DECAY_MODES = ("exponential", "power_law", "hyperbolic")


@dataclass
class DecayConfig:
    mode: str = "exponential"
    lambda_exp: float = 0.01
    alpha_pow: float = 0.5
    half_life: float = 50.0


def _exponential_decay(r: float, dt: int, lam: float) -> float:
    return float(r * math.exp(-lam * dt))


def _power_law_decay(r: float, dt: int, alpha: float) -> float:
    if dt <= 0:
        return r
    return float(r / (1.0 + dt) ** alpha)


def _hyperbolic_decay(r: float, dt: int, half_life: float) -> float:
    if half_life <= 0:
        return 0.0
    return float(r * half_life / (half_life + dt))


class DecayScheduler:
    def __init__(self, config: DecayConfig | None = None):
        self._cfg = config or DecayConfig()
        self._last_active: dict[int, int] = {}
        self._t: int = 0

    def tick(self) -> None:
        self._t += 1

    def mark_active(self, agent_id: int) -> None:
        self._last_active[agent_id] = self._t

    def apply(self, agent_id: int, current_rep: float) -> float:
        last = self._last_active.get(agent_id, self._t)
        dt = max(0, self._t - last)
        if dt == 0:
            return current_rep
        mode = self._cfg.mode
        if mode == "exponential":
            decayed = _exponential_decay(current_rep, dt, self._cfg.lambda_exp)
        elif mode == "power_law":
            decayed = _power_law_decay(current_rep, dt, self._cfg.alpha_pow)
        elif mode == "hyperbolic":
            decayed = _hyperbolic_decay(current_rep, dt, self._cfg.half_life)
        else:
            decayed = current_rep
        return float(np.clip(decayed, 0.0, 1.0))

    def apply_all(self, reps: dict[int, float]) -> dict[int, float]:
        return {aid: self.apply(aid, r) for aid, r in reps.items()}
