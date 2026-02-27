from __future__ import annotations
import numpy as np
from .graph import TrustGraph

_DAMPING = 0.85
_MAX_ITER = 50
_TOL = 1e-6


def _pagerank_dict(graph: TrustGraph, node_ids: list[int]) -> dict[int, float]:
    n = len(node_ids)
    if n == 0:
        return {}
    idx = {nid: i for i, nid in enumerate(node_ids)}
    scores = np.full(n, 1.0 / n)
    for _ in range(_MAX_ITER):
        new_scores = np.full(n, (1.0 - _DAMPING) / n)
        for nid in node_ids:
            i = idx[nid]
            out_nbrs = graph.neighbors_out(nid)
            out_nbrs = [v for v in out_nbrs if v in idx]
            if not out_nbrs:
                for j in range(n):
                    new_scores[j] += _DAMPING * scores[i] / n
                continue
            total_w = sum(graph.edge_weight(nid, v) for v in out_nbrs)
            if total_w < 1e-12:
                total_w = 1.0
            for v in out_nbrs:
                j = idx[v]
                w = graph.edge_weight(nid, v) / total_w
                new_scores[j] += _DAMPING * scores[i] * w
        delta = float(np.abs(new_scores - scores).sum())
        scores = new_scores
        if delta < _TOL:
            break
    return {nid: float(scores[idx[nid]]) for nid in node_ids}


def _weighted_reputation_propagation(
    graph: TrustGraph,
    node_ids: list[int],
    base_reps: dict[int, float],
    alpha: float = 0.3,
) -> dict[int, float]:
    propagated = dict(base_reps)
    for nid in node_ids:
        in_nbrs = graph.neighbors_in(nid)
        in_nbrs = [v for v in in_nbrs if v in base_reps]
        if not in_nbrs:
            continue
        total_w = sum(graph.edge_weight(v, nid) for v in in_nbrs)
        if total_w < 1e-12:
            continue
        contrib = sum(
            graph.edge_weight(v, nid) * base_reps[v] for v in in_nbrs
        ) / total_w
        propagated[nid] = (1.0 - alpha) * base_reps[nid] + alpha * contrib
    return propagated


class PropagationEngine:
    def __init__(self, alpha: float = 0.3, use_pagerank: bool = True):
        self._alpha = alpha
        self._use_pagerank = use_pagerank
        self._pr_cache: dict[int, float] = {}

    def propagate(self, graph: TrustGraph, base_reps: dict[int, float]) -> dict[int, float]:
        node_ids = graph.active_node_ids()
        if not node_ids:
            return base_reps

        if self._use_pagerank:
            pr = _pagerank_dict(graph, node_ids)
            pr_max = max(pr.values()) if pr else 1.0
            if pr_max > 0:
                pr = {k: v / pr_max for k, v in pr.items()}
            self._pr_cache = pr
            adjusted = {
                nid: float(np.clip(base_reps.get(nid, 0.5) * (0.7 + 0.3 * pr.get(nid, 0.5)), 0.0, 1.0))
                for nid in node_ids
            }
        else:
            adjusted = dict(base_reps)

        return _weighted_reputation_propagation(graph, node_ids, adjusted, self._alpha)

    def pagerank_score(self, agent_id: int) -> float:
        return self._pr_cache.get(agent_id, 0.0)
