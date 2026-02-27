"""tests/test_utils.py — unit tests for utils submodule."""
from __future__ import annotations
import pytest
import numpy as np

from chronosrep.utils.math_utils import (
    softmax, sigmoid, shannon_entropy, rolling_mean, rolling_std,
    ewma, z_score, clip_normalize, gini_coefficient, cosine_similarity, kl_divergence,
)
from chronosrep.utils.seed_manager import set_global_seed, make_rng, fork_rng, temp_seed
from chronosrep.utils.serialization import save_json, load_json
import tempfile, pathlib


def test_softmax_sums_to_one():
    x = np.array([1.0, 2.0, 3.0])
    s = softmax(x)
    assert np.isclose(s.sum(), 1.0)


def test_sigmoid_bounds():
    assert 0 < sigmoid(0.0) < 1
    assert sigmoid(100.0) > 0.99
    assert sigmoid(-100.0) < 0.01


def test_shannon_entropy_uniform():
    p = np.array([0.25, 0.25, 0.25, 0.25])
    assert shannon_entropy(p) == pytest.approx(2.0)


def test_rolling_mean_length():
    arr = [1.0, 2.0, 3.0, 4.0, 5.0]
    out = rolling_mean(arr, window=3)
    assert len(out) == len(arr)


def test_ewma_length():
    arr = list(range(10, dtype=float) if False else [float(i) for i in range(10)])
    out = ewma(arr, lam=0.9)
    assert len(out) == len(arr)


def test_gini_zero_for_equal():
    g = gini_coefficient([1.0, 1.0, 1.0, 1.0])
    assert g == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_identical():
    a = np.array([1.0, 2.0, 3.0])
    assert cosine_similarity(a, a) == pytest.approx(1.0)


def test_kl_divergence_self_zero():
    p = np.array([0.3, 0.3, 0.4])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-5)


def test_set_seed_reproducible():
    set_global_seed(42)
    a = np.random.rand(5)
    set_global_seed(42)
    b = np.random.rand(5)
    np.testing.assert_array_equal(a, b)


def test_fork_rng_different():
    r1 = fork_rng(42, 0)
    r2 = fork_rng(42, 1)
    a = r1.standard_normal(5)
    b = r2.standard_normal(5)
    assert not np.allclose(a, b)


def test_temp_seed_restores():
    np.random.seed(99)
    before = np.random.rand()
    np.random.seed(99)
    with temp_seed(12345):
        _ = np.random.rand(10)
    after = np.random.rand()
    assert before == pytest.approx(after)


def test_save_load_json(tmp_path):
    data = {"a": 1, "b": [1, 2, 3], "c": np.array([0.1, 0.2])}
    p = save_json(data, tmp_path / "test.json")
    loaded = load_json(p)
    assert loaded["a"] == 1
    assert loaded["c"] == pytest.approx([0.1, 0.2])
