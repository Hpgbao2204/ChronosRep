"""
trust_propagation_damping.py
Evaluates the sensitivity of the weighted PageRank trust propagation engine
to the damping factor alpha.  Three values are compared on an identical
ring topology, showing how alpha controls the balance between local and
global reputation influence.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chronosrep.trust.graph import TrustGraph
from chronosrep.trust.propagation import PropagationEngine

_OUT = Path(__file__).parent / "output" / "trust_propagation_damping.png"


def _build_ring(n: int, trust: float = 0.7) -> TrustGraph:
    g = TrustGraph()
    for i in range(n):
        g.record_interaction(i, (i + 1) % n, success=True)
    return g


def main(out_path: Path = _OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    N = 50
    T_steps = 60
    alphas = [0.70, 0.85, 0.95]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)

    for ax, alpha in zip(axes, alphas):
        np.random.seed(0)
        g = _build_ring(N)
        rep = {i: float(np.random.uniform(0.3, 0.9)) for i in range(N)}
        engine = PropagationEngine(damping=alpha, max_iter=20)

        series = [list(rep.values())]
        for _ in range(T_steps):
            rep = engine.propagate(g, rep)
            series.append(list(rep.values()))

        arr = np.array(series)
        for i in range(min(10, N)):
            ax.plot(arr[:, i], alpha=0.5, linewidth=0.8)
        ax.plot(arr.mean(axis=1), color="black", linewidth=2, label="mean")
        ax.set_title(f"α = {alpha}")
        ax.set_xlabel("Propagation step")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("Reputation")
    fig.suptitle("Trust Propagation — Damping Factor Sensitivity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
