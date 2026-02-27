from __future__ import annotations
import numpy as np


def softmax(x: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    e = np.exp((x - x.max()) / temperature)
    return e / e.sum()


def sigmoid(x: float | np.ndarray, k: float = 1.0, x0: float = 0.0) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-k * (np.asarray(x) - x0)))


def shannon_entropy(probs: np.ndarray) -> float:
    p = np.asarray(probs, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def rolling_mean(arr: list[float], window: int) -> list[float]:
    if not arr:
        return []
    out = []
    for i in range(len(arr)):
        w = arr[max(0, i - window + 1): i + 1]
        out.append(float(np.mean(w)))
    return out


def rolling_std(arr: list[float], window: int, ddof: int = 1) -> list[float]:
    if not arr:
        return []
    out = []
    for i in range(len(arr)):
        w = arr[max(0, i - window + 1): i + 1]
        if len(w) < 2:
            out.append(0.0)
        else:
            out.append(float(np.std(w, ddof=ddof)))
    return out


def ewma(arr: list[float], lam: float = 0.94) -> list[float]:
    if not arr:
        return []
    result = [arr[0]]
    for v in arr[1:]:
        result.append(lam * result[-1] + (1 - lam) * v)
    return result


def z_score(value: float, mean: float, std: float) -> float:
    if std < 1e-12:
        return 0.0
    return (value - mean) / std


def clip_normalize(arr: np.ndarray, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    arr = np.clip(arr, lo, hi)
    span = hi - lo
    return (arr - lo) / span if span > 0 else arr


def gini_coefficient(values: list[float] | np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[arr >= 0]
    if arr.sum() < 1e-12:
        return 0.0
    arr = np.sort(arr)
    n = len(arr)
    idx = np.arange(1, n + 1)
    return float((2 * (idx * arr).sum() / (n * arr.sum())) - (n + 1) / n)


def lorenz_curve(values: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = np.sort(np.asarray(values, dtype=float))
    arr = arr[arr >= 0]
    if arr.sum() < 1e-12:
        x = np.linspace(0, 1, len(arr) + 1)
        return x, x
    cumulative = np.cumsum(arr) / arr.sum()
    x = np.linspace(0, 1, len(arr) + 1)
    y = np.concatenate([[0.0], cumulative])
    return x, y


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom < 1e-12:
        return 0.0
    return float(np.dot(a, b) / denom)


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=float) + eps
    q = np.asarray(q, dtype=float) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))
