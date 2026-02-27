from __future__ import annotations
import random
import numpy as np
from contextlib import contextmanager


_GLOBAL_SEED: int | None = None


def set_global_seed(seed: int) -> None:
    global _GLOBAL_SEED
    _GLOBAL_SEED = seed
    random.seed(seed)
    np.random.seed(seed)


def get_global_seed() -> int | None:
    return _GLOBAL_SEED


def make_rng(seed: int | None = None) -> np.random.Generator:
    if seed is None and _GLOBAL_SEED is not None:
        seed = _GLOBAL_SEED
    return np.random.default_rng(seed)


def fork_rng(base_seed: int, fork_id: int) -> np.random.Generator:
    child_seed = (base_seed * 2654435761 + fork_id) & 0xFFFF_FFFF
    return np.random.default_rng(child_seed)


@contextmanager
def temp_seed(seed: int):
    py_state  = random.getstate()
    np_state  = np.random.get_state()
    random.seed(seed)
    np.random.seed(seed)
    try:
        yield
    finally:
        random.setstate(py_state)
        np.random.set_state(np_state)


def seeds_for_sweep(base: int, n: int) -> list[int]:
    rng = np.random.default_rng(base)
    return rng.integers(0, 2**31, size=n).tolist()
