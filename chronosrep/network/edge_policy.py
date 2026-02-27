from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class EdgeWeightConfig:
    recency_decay: float = 0.95
    outcome_momentum: float = 0.6
    issuer_trust_factor: float = 0.3
    volatility_penalty: float = 0.2


_DEFAULT_CFG = EdgeWeightConfig()


def _recency_factor(dt: int, decay: float) -> float:
    return decay ** dt


def _outcome_momentum(history: list[int], momentum: float) -> float:
    if not history:
        return 0.5
    weighted = 0.0
    total_w = 0.0
    for i, outcome in enumerate(reversed(history[-10:])):
        w = momentum ** i
        weighted += w * outcome
        total_w += w
    return weighted / total_w if total_w > 0 else 0.5


def _volatility_adjusted(base: float, volatility: float, penalty: float) -> float:
    return float(np.clip(base * (1.0 - penalty * volatility), 0.0, 1.0))


class EdgeWeightPolicy:
    def __init__(self, config: EdgeWeightConfig | None = None):
        self._cfg = config or _DEFAULT_CFG

    def compute(
        self,
        outcome_history: list[int],
        dt_since_last: int,
        issuer_trust: float,
        volatility: float,
    ) -> float:
        recency = _recency_factor(dt_since_last, self._cfg.recency_decay)
        momentum = _outcome_momentum(outcome_history, self._cfg.outcome_momentum)
        base = (
            (1.0 - self._cfg.issuer_trust_factor) * momentum
            + self._cfg.issuer_trust_factor * issuer_trust
        ) * recency
        return _volatility_adjusted(base, volatility, self._cfg.volatility_penalty)

    def batch_update(
        self,
        edges: dict[tuple[int, int], dict],
        current_t: int,
        agent_volatility: dict[int, float],
    ) -> dict[tuple[int, int], float]:
        result: dict[tuple[int, int], float] = {}
        for (u, v), meta in edges.items():
            dt = current_t - meta.get("last_t", current_t)
            hist = meta.get("outcome_history", [])
            trust = meta.get("issuer_trust", 0.7)
            vol = agent_volatility.get(u, 0.0)
            result[(u, v)] = self.compute(hist, dt, trust, vol)
        return result
