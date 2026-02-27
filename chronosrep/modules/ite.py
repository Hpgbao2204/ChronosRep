from __future__ import annotations
import random
from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx
import community as community_louvain
import numpy as np

_SUSPECT_DIN   = 0.60
_SUSPECT_EXTER = 0.20
_BETA          = 0.5
_GAMMA         = 0.5


@dataclass
class _CommStats:
    members: set = field(default_factory=set)
    din: float = 0.0
    ext_ratio: float = 1.0


def _fast_jaccard(nbrs_u: set, nbrs_v: set) -> float:
    inter = len(nbrs_u & nbrs_v)
    union = len(nbrs_u | nbrs_v)
    return inter / union if union > 0 else 0.0


def _is_suspect(stats: _CommStats) -> bool:
    return stats.din >= _SUSPECT_DIN and stats.ext_ratio <= _SUSPECT_EXTER


def _structural_penalty(sim: float, din: float) -> float:
    raw = _BETA * sim + _GAMMA * din
    return max(0.0, 1.0 - raw)


class ITE:
    def __init__(
        self,
        beta: float = _BETA,
        gamma: float = _GAMMA,
        recompute_interval: int = 10,
        k_interactions: int = 3,
        seed: int = 42,
    ):
        self._beta = beta
        self._gamma = gamma
        self._recompute_interval = recompute_interval
        self._k = k_interactions
        self._rng = random.Random(seed)

        self._graph: nx.DiGraph = nx.DiGraph()
        self._nbrs: dict[int, set] = defaultdict(set)
        self._partition: dict[int, int] = {}
        self._comm_stats: dict[int, _CommStats] = {}
        self._step: int = 0

    def _ensure(self, ids: list[int]) -> None:
        for aid in ids:
            if aid not in self._graph:
                self._graph.add_node(aid)
                self._nbrs.setdefault(aid, set())

    def _add_edge(self, u: int, v: int, outcome: int) -> None:
        if self._graph.has_edge(u, v):
            self._graph[u][v]["weight"] = self._graph[u][v]["weight"] + 1
            self._graph[u][v]["last"] = outcome
        else:
            self._graph.add_edge(u, v, weight=1, last=outcome)
        self._nbrs[u].add(v)
        self._nbrs[v].add(u)

    def _recompute(self) -> None:
        G_und = nx.Graph()
        G_und.add_nodes_from(self._graph.nodes())
        G_und.add_edges_from(self._graph.edges())
        if G_und.number_of_edges() == 0:
            self._partition = {n: n for n in G_und.nodes()}
        else:
            self._partition = community_louvain.best_partition(G_und)

        by_comm: dict[int, list] = defaultdict(list)
        for node, comm in self._partition.items():
            by_comm[comm].append(node)

        self._comm_stats = {}
        for cid, members in by_comm.items():
            n = len(members)
            mset = set(members)
            internal = sum(1 for u, v in self._graph.edges() if u in mset and v in mset)
            external = sum(1 for u, v in self._graph.edges() if u in mset and v not in mset)
            max_int = n * (n - 1)
            din = internal / max_int if max_int > 0 else 0.0
            total_out = internal + external
            ext_ratio = external / total_out if total_out > 0 else 1.0
            self._comm_stats[cid] = _CommStats(members=mset, din=din, ext_ratio=ext_ratio)

    def tick(self) -> None:
        self._step += 1
        if self._step % self._recompute_interval == 0:
            self._recompute()

    def generate_interactions(
        self, agent_id: int, all_ids: list[int]
    ) -> list[tuple[int, int]]:
        self._ensure([agent_id] + all_ids)
        candidates = [x for x in all_ids if x != agent_id]
        targets = self._rng.sample(candidates, min(self._k, len(candidates)))
        pairs: list[tuple[int, int]] = []
        for t in targets:
            outcome = self._rng.randint(0, 1)
            self._add_edge(agent_id, t, outcome)
            pairs.append((t, outcome))
        return pairs

    def force_interaction(self, u: int, v: int, outcome: int) -> None:
        self._ensure([u, v])
        self._add_edge(u, v, outcome)

    def penalized_evidence(
        self, endorser_id: int, target_id: int, raw_outcome: int
    ) -> float:
        e = float(raw_outcome)
        if not self._partition:
            return e
        c_e = self._partition.get(endorser_id)
        c_t = self._partition.get(target_id)
        if c_e is None or c_t is None or c_e != c_t:
            return e
        stats = self._comm_stats.get(c_e)
        if stats is None:
            return e
        if not _is_suspect(stats):
            return e
        nbrs_e = self._nbrs[endorser_id] | {endorser_id}
        nbrs_t = self._nbrs[target_id]   | {target_id}
        sim = _fast_jaccard(nbrs_e, nbrs_t)
        w = _structural_penalty(sim, stats.din)
        return w * e

    def get_neighbors(self, agent_id: int) -> list:
        return list(self._nbrs.get(agent_id, set()))
