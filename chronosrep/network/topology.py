from __future__ import annotations
import random
import networkx as nx
from dataclasses import dataclass
from typing import Optional

_TOPOLOGY_MODES = ("random_erdos_renyi", "barabasi_albert", "watts_strogatz", "complete")


@dataclass
class TopologyConfig:
    mode: str = "barabasi_albert"
    p_er: float = 0.02
    m_ba: int = 3
    k_ws: int = 4
    p_ws: float = 0.1


class TopologyBuilder:
    def __init__(self, config: Optional[TopologyConfig] = None, seed: int = 0):
        self._cfg = config or TopologyConfig()
        self._seed = seed

    def build(self, n: int) -> nx.DiGraph:
        cfg = self._cfg
        rng = random.Random(self._seed)

        if cfg.mode == "random_erdos_renyi":
            G_und = nx.erdos_renyi_graph(n, cfg.p_er, seed=self._seed)
        elif cfg.mode == "barabasi_albert":
            G_und = nx.barabasi_albert_graph(n, cfg.m_ba, seed=self._seed)
        elif cfg.mode == "watts_strogatz":
            G_und = nx.watts_strogatz_graph(n, cfg.k_ws, cfg.p_ws, seed=self._seed)
        elif cfg.mode == "complete":
            G_und = nx.complete_graph(n)
        else:
            G_und = nx.erdos_renyi_graph(n, cfg.p_er, seed=self._seed)

        G = nx.DiGraph()
        G.add_nodes_from(range(n))
        for u, v in G_und.edges():
            if rng.random() < 0.5:
                G.add_edge(u, v, weight=rng.uniform(0.3, 1.0))
            else:
                G.add_edge(v, u, weight=rng.uniform(0.3, 1.0))
        return G

    def rewire(self, G: nx.DiGraph, p_rewire: float = 0.05) -> nx.DiGraph:
        rng = random.Random(self._seed + 1)
        edges = list(G.edges())
        nodes = list(G.nodes())
        for u, v in edges:
            if rng.random() < p_rewire:
                G.remove_edge(u, v)
                new_v = rng.choice(nodes)
                if new_v != u and not G.has_edge(u, new_v):
                    G.add_edge(u, new_v, weight=rng.uniform(0.3, 1.0))
        return G

    def inject_clique(self, G: nx.DiGraph, node_ids: list[int]) -> nx.DiGraph:
        for u in node_ids:
            for v in node_ids:
                if u != v:
                    G.add_edge(u, v, weight=1.0)
        return G
