from __future__ import annotations
from dataclasses import dataclass, field
from collections import defaultdict
import numpy as np

_EDGE_DECAY_BASE = 0.95
_MIN_WEIGHT = 1e-6


@dataclass
class _TrustEdge:
    weight: float
    outcome_sum: float
    interaction_count: int
    last_t: int


@dataclass
class _TrustNode:
    agent_id: int
    reputation: float
    volatility: float
    irv: np.ndarray
    isolated: bool = False
    t_last_active: int = 0


class TrustGraph:
    def __init__(self, decay_base: float = _EDGE_DECAY_BASE):
        self._decay_base = decay_base
        self._nodes: dict[int, _TrustNode] = {}
        self._edges: dict[tuple[int, int], _TrustEdge] = {}
        self._out_adj: dict[int, set[int]] = defaultdict(set)
        self._in_adj: dict[int, set[int]] = defaultdict(set)
        self._t: int = 0

    def upsert_node(self, agent_id: int, reputation: float, volatility: float, irv: np.ndarray) -> None:
        if agent_id in self._nodes:
            n = self._nodes[agent_id]
            n.reputation = reputation
            n.volatility = volatility
            n.irv = irv
            n.t_last_active = self._t
        else:
            self._nodes[agent_id] = _TrustNode(
                agent_id=agent_id,
                reputation=reputation,
                volatility=volatility,
                irv=irv.copy(),
                t_last_active=self._t,
            )

    def record_interaction(self, src: int, dst: int, outcome: int, penalized_signal: float) -> None:
        key = (src, dst)
        if key in self._edges:
            e = self._edges[key]
            e.weight = max(_MIN_WEIGHT, e.weight * self._decay_base + penalized_signal)
            e.outcome_sum += outcome
            e.interaction_count += 1
            e.last_t = self._t
        else:
            self._edges[key] = _TrustEdge(
                weight=penalized_signal,
                outcome_sum=float(outcome),
                interaction_count=1,
                last_t=self._t,
            )
            self._out_adj[src].add(dst)
            self._in_adj[dst].add(src)

    def isolate(self, agent_id: int) -> None:
        if agent_id in self._nodes:
            self._nodes[agent_id].isolated = True

    def neighbors_out(self, agent_id: int) -> list[int]:
        return list(self._out_adj.get(agent_id, set()))

    def neighbors_in(self, agent_id: int) -> list[int]:
        return list(self._in_adj.get(agent_id, set()))

    def edge_weight(self, src: int, dst: int) -> float:
        e = self._edges.get((src, dst))
        return e.weight if e else 0.0

    def edge_success_rate(self, src: int, dst: int) -> float:
        e = self._edges.get((src, dst))
        if e is None or e.interaction_count == 0:
            return 0.5
        return e.outcome_sum / e.interaction_count

    def tick(self) -> None:
        self._t += 1

    def active_node_ids(self) -> list[int]:
        return [nid for nid, n in self._nodes.items() if not n.isolated]

    def node(self, agent_id: int) -> _TrustNode | None:
        return self._nodes.get(agent_id)

    def num_nodes(self) -> int:
        return len(self._nodes)

    def num_edges(self) -> int:
        return len(self._edges)
