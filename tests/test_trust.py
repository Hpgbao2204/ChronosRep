"""tests/test_trust.py — unit tests for trust submodule."""
from __future__ import annotations
import pytest
import numpy as np

from chronosrep.trust.graph import TrustGraph
from chronosrep.trust.propagation import PropagationEngine
from chronosrep.trust.decay import DecayScheduler, DecayConfig


def test_trust_graph_record():
    g = TrustGraph()
    g.record_interaction(0, 1, success=True)
    assert g.edge_success_rate(0, 1) == pytest.approx(1.0)


def test_trust_graph_failure():
    g = TrustGraph()
    g.record_interaction(0, 1, success=True)
    g.record_interaction(0, 1, success=False)
    rate = g.edge_success_rate(0, 1)
    assert 0.0 < rate < 1.0


def test_trust_graph_isolate():
    g = TrustGraph()
    for _ in range(5):
        g.record_interaction(0, 1, success=True)
    g.isolate(0)
    assert g.edge_success_rate(0, 1) == pytest.approx(0.0)


def test_propagation_engine_converges():
    g = TrustGraph()
    for i in range(10):
        g.record_interaction(i, (i + 1) % 10, success=True)
    rep = {i: float(np.random.uniform(0.4, 0.9)) for i in range(10)}
    engine = PropagationEngine(damping=0.85, max_iter=30)
    for _ in range(5):
        rep = engine.propagate(g, rep)
    assert all(0.0 <= v <= 1.0 for v in rep.values())


def test_decay_exponential():
    cfg = DecayConfig(mode="exponential", rate=0.1)
    sched = DecayScheduler(cfg)
    v = sched.decay(1.0, elapsed=10)
    assert 0.0 < v < 1.0


def test_decay_power_law():
    cfg = DecayConfig(mode="power_law", rate=0.5)
    sched = DecayScheduler(cfg)
    v = sched.decay(1.0, elapsed=4)
    assert v == pytest.approx(1.0 / (1 + 4) ** 0.5)


def test_decay_hyperbolic():
    cfg = DecayConfig(mode="hyperbolic", rate=1.0)
    sched = DecayScheduler(cfg)
    v = sched.decay(1.0, elapsed=1)
    assert v == pytest.approx(0.5)
