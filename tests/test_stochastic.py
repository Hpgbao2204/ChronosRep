"""tests/test_stochastic.py — unit tests for stochastic submodule."""
from __future__ import annotations
import pytest
import numpy as np

from chronosrep.stochastic.wiener import WienerProcess
from chronosrep.stochastic.jump import JumpProcess
from chronosrep.stochastic.solver import SDESolver, OUParams
from chronosrep.stochastic.volatility import VolatilityEstimator


def test_wiener_increment_shape():
    wp = WienerProcess(dt=1.0, seed=0)
    dW = wp.increment(0)
    assert isinstance(dW, float)


def test_wiener_batch():
    wp = WienerProcess(dt=1.0, seed=1)
    dWs = wp.batch_increment([0, 1, 2])
    assert set(dWs.keys()) == {0, 1, 2}
    assert all(isinstance(v, float) for v in dWs.values())


def test_wiener_quadratic_variation():
    wp = WienerProcess(dt=1.0, seed=2)
    for _ in range(50):
        wp.increment(0)
    qv = wp.quadratic_variation(0)
    assert qv >= 0.0


def test_jump_no_fire_low_gamma():
    jp = JumpProcess(gamma_threshold=3.0, seed=3)
    j = jp.sample(0, gamma=0.5, dt=1.0)
    assert j == 0.0


def test_jump_fires_high_gamma():
    jp = JumpProcess(gamma_threshold=3.0, seed=4)
    fired = any(jp.sample(0, gamma=5.0, dt=1.0) != 0.0 for _ in range(200))
    assert fired


def test_solver_init_step_clipped():
    p = OUParams(mu=0.7, theta=0.3, sigma=0.05)
    solver = SDESolver(dt=1.0, seed=10)
    solver.init(0, 0.5)
    x = solver.step(0, p, gamma=0.0)
    assert 0.0 <= x <= 1.0


def test_solver_trajectory_length():
    p = OUParams(mu=0.5, theta=0.2, sigma=0.02)
    solver = SDESolver(dt=1.0, seed=20)
    traj = solver.trajectory(0, p, n_steps=30)
    assert len(traj) == 31
    assert all(0.0 <= v <= 1.0 for v in traj)


def test_volatility_realized_nonneg():
    ve = VolatilityEstimator(window=20)
    xs = np.cumsum(np.random.default_rng(0).normal(0, 0.05, 30)) + 0.5
    xs = np.clip(xs, 0, 1)
    for i in range(1, len(xs)):
        ve.update(0, float(xs[i]), float(xs[i - 1]))
    assert ve.realized_vol(0) >= 0.0


def test_volatility_ewma_positive():
    ve = VolatilityEstimator(window=20)
    for i in range(1, 30):
        ve.update(0, float(i * 0.01), float((i - 1) * 0.01))
    assert ve.ewma_vol(0) > 0.0


def test_volatility_regime():
    ve = VolatilityEstimator(window=10)
    for i in range(1, 15):
        ve.update(0, 0.5 + float(i) * 0.002, 0.5 + float(i - 1) * 0.002)
    regime = ve.regime(0)
    assert regime in ("low", "medium", "high")
