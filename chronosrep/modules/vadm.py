from __future__ import annotations
from collections import deque
from dataclasses import dataclass, field

import numpy as np

_GAMMA_THRESHOLD = 3.0
_X_LO = 0.0
_X_HI = 1.0


@dataclass
class _AgentState:
    x: float
    theta: float
    history: deque = field(default_factory=lambda: deque(maxlen=20))
    t: int = 0


def _noise_floor(sigma: float, theta: float) -> float:
    return sigma / max(np.sqrt(2.0 * theta), 1e-8)


def _gamma_ratio(epsilon: float, sigma: float, theta: float) -> float:
    nf = _noise_floor(sigma, theta)
    return abs(epsilon) / nf if nf > 0 else 0.0


def _euler_maruyama(
    x: float,
    mu: float,
    theta: float,
    sigma: float,
    dt: float,
    rng: np.random.Generator,
    jump_scale: float,
    gamma: float,
) -> float:
    drift = theta * (mu - x) * dt
    diffusion = sigma * rng.standard_normal() * np.sqrt(dt)
    jump = 0.0
    if gamma > _GAMMA_THRESHOLD:
        direction = -1.0 if x > mu else 1.0
        jump = direction * abs(rng.normal(0.0, jump_scale))
    return float(np.clip(x + drift + diffusion + jump, _X_LO, _X_HI))


def _update_theta(theta: float, epsilon: float, alpha: float, dt: float) -> float:
    grad = epsilon - theta * dt
    return float(np.clip(theta + alpha * grad, 1e-4, 10.0))


class VADM:
    def __init__(
        self,
        dt: float = 1.0,
        theta_0: float = 0.30,
        sigma: float = 0.03,
        jump_scale: float = 0.35,
        alpha: float = 0.05,
        window: int = 20,
        seed: int = 0,
    ):
        self._dt = dt
        self._theta_0 = theta_0
        self._sigma = sigma
        self._jump_scale = jump_scale
        self._alpha = alpha
        self._window = window
        self._rng = np.random.default_rng(seed)
        self._states: dict[int, _AgentState] = {}

    def _init(self, agent_id: int, mu: float) -> _AgentState:
        s = _AgentState(
            x=float(np.clip(mu, _X_LO, _X_HI)),
            theta=self._theta_0,
            history=deque(maxlen=self._window),
        )
        self._states[agent_id] = s
        return s

    def step(
        self,
        agent_id: int,
        irv: np.ndarray,
        r_static: float,
    ) -> tuple[float, float]:
        mu = float(np.clip(irv[0], _X_LO, _X_HI))
        s = self._states.get(agent_id) or self._init(agent_id, mu)

        epsilon = abs(r_static - s.x)
        gamma = _gamma_ratio(epsilon, self._sigma, s.theta)

        s.theta = _update_theta(s.theta, epsilon, self._alpha, self._dt)
        s.x = _euler_maruyama(
            s.x, mu, s.theta, self._sigma,
            self._dt, self._rng, self._jump_scale, gamma,
        )
        s.history.append(s.x)
        s.t += 1

        sigma_w = float(np.std(s.history)) if len(s.history) > 1 else 0.0
        return s.x, sigma_w

    def step_ou_only(
        self,
        x0: float,
        mu: float,
        theta: float,
        n_steps: int,
    ) -> list[float]:
        traj = [x0]
        x = x0
        for _ in range(n_steps):
            drift = theta * (mu - x) * self._dt
            diff  = self._sigma * self._rng.standard_normal() * np.sqrt(self._dt)
            x = float(np.clip(x + drift + diff, _X_LO, _X_HI))
            traj.append(x)
        return traj

    def decay(self, reputation: float, volatility: float, delta_t: int) -> float:
        drift = self._theta_0 * (0.5 - reputation) * delta_t
        diff  = volatility * self._rng.standard_normal() * np.sqrt(delta_t)
        return float(np.clip(reputation + drift + diff, _X_LO, _X_HI))
