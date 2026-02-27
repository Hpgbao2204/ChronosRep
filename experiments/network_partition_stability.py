"""
network_partition_stability.py
Measures the Louvain modularity of three synthetic network topologies
(Erdos-Renyi, Barabasi-Albert, Watts-Strogatz) as edges are continuously
rewired over time.  Shows how community structure stability differs by
topology class, which has implications for collusion-ring detection latency.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    import community as community_louvain
    _HAS_LOUVAIN = True
except ImportError:
    _HAS_LOUVAIN = False

from chronosrep.network.topology import TopologyBuilder
from chronosrep.network.partition import PartitionCache

_OUT = Path(__file__).parent / "output" / "network_partition_stability.png"


def _modularity(G: nx.Graph) -> float:
    if not _HAS_LOUVAIN or G.number_of_nodes() < 3:
        return 0.0
    try:
        part = community_louvain.best_partition(G)
        return community_louvain.modularity(part, G)
    except Exception:
        return 0.0


def _run_topology(mode: str, n: int, T: int) -> list[float]:
    builder = TopologyBuilder(seed=42)
    G = builder.build(mode, n)
    cache = PartitionCache()
    mods = []
    for t in range(T):
        if t % 5 == 0 and t > 0:
            edges = list(G.edges())
            n_rewire = max(1, int(0.02 * len(edges)))
            for _ in range(n_rewire):
                if edges:
                    u, v = edges[np.random.randint(len(edges))]
                    G.remove_edge(u, v)
                    a, b = np.random.choice(list(G.nodes()), 2, replace=False)
                    G.add_edge(int(a), int(b))
        ug = G.to_undirected() if G.is_directed() else G
        q = _modularity(ug)
        mods.append(q)
        if t % 10 == 0:
            import networkx.algorithms.community as nx_comm
            try:
                part_dict = community_louvain.best_partition(ug) if _HAS_LOUVAIN else {n: 0 for n in ug.nodes()}
                cache.update(t, part_dict, ug)
            except Exception:
                pass
    return mods


def main(out_path: Path = _OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    N = 80
    T = 100
    modes = ["erdos_renyi", "barabasi_albert", "watts_strogatz"]
    labels = ["Erdős-Rényi", "Barabási-Albert", "Watts-Strogatz"]

    fig, ax = plt.subplots(figsize=(9, 4))
    steps = np.arange(T)
    for mode, lbl in zip(modes, labels):
        np.random.seed(0)
        mods = _run_topology(mode, N, T)
        ax.plot(steps, mods, label=lbl, linewidth=1.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Louvain modularity Q")
    ax.set_title("Community Partition Stability — Louvain Modularity Across Network Topologies")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
