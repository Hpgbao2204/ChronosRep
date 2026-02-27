from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass, field
import networkx as nx


@dataclass
class PartitionSnapshot:
    t: int
    partition: dict[int, int]
    modularity: float
    n_communities: int
    community_sizes: dict[int, int]


class PartitionCache:
    def __init__(self, max_history: int = 20):
        self._max_history = max_history
        self._snapshots: list[PartitionSnapshot] = []
        self._current: dict[int, int] = {}

    def store(self, t: int, partition: dict[int, int], G_und: nx.Graph) -> PartitionSnapshot:
        try:
            mod = nx.community.modularity(
                G_und,
                [{n for n, c in partition.items() if c == cid}
                 for cid in set(partition.values())],
            )
        except Exception:
            mod = 0.0

        sizes: dict[int, int] = defaultdict(int)
        for c in partition.values():
            sizes[c] += 1

        snap = PartitionSnapshot(
            t=t,
            partition=dict(partition),
            modularity=mod,
            n_communities=len(set(partition.values())),
            community_sizes=dict(sizes),
        )
        self._snapshots.append(snap)
        if len(self._snapshots) > self._max_history:
            self._snapshots.pop(0)
        self._current = dict(partition)
        return snap

    def current_partition(self) -> dict[int, int]:
        return dict(self._current)

    def community_of(self, agent_id: int) -> int | None:
        return self._current.get(agent_id)

    def community_members(self, community_id: int) -> list[int]:
        return [n for n, c in self._current.items() if c == community_id]

    def modularity_history(self) -> list[float]:
        return [s.modularity for s in self._snapshots]

    def stability_score(self) -> float:
        h = self.modularity_history()
        if len(h) < 2:
            return 1.0
        import numpy as np
        return float(1.0 - np.std(h))

    def latest_snapshot(self) -> PartitionSnapshot | None:
        return self._snapshots[-1] if self._snapshots else None
