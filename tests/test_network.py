"""tests/test_network.py — unit tests for network submodule."""
from __future__ import annotations
import pytest
import networkx as nx

from chronosrep.network.topology import TopologyBuilder
from chronosrep.network.edge_policy import EdgeWeightPolicy
from chronosrep.network.partition import PartitionCache


def test_er_topology_nodes():
    builder = TopologyBuilder(seed=0)
    G = builder.build("erdos_renyi", n=50)
    assert G.number_of_nodes() == 50


def test_ba_topology_connected():
    builder = TopologyBuilder(seed=1)
    G = builder.build("barabasi_albert", n=30)
    ug = G.to_undirected() if G.is_directed() else G
    assert nx.is_connected(ug)


def test_ws_topology_nodes():
    builder = TopologyBuilder(seed=2)
    G = builder.build("watts_strogatz", n=20)
    assert G.number_of_nodes() == 20


def test_inject_clique():
    builder = TopologyBuilder(seed=3)
    G = builder.build("erdos_renyi", n=40)
    before = G.number_of_nodes()
    builder.inject_clique(G, size=5)
    assert G.number_of_nodes() == before + 5


def test_edge_weight_policy():
    policy = EdgeWeightPolicy()
    w = policy.compute(0, 1, outcomes=[1, 1, 0, 1], recency=0.9, issuer_trust=0.8, volatility=0.1)
    assert 0.0 <= w <= 1.0


def test_partition_cache_update():
    cache = PartitionCache()
    G = nx.karate_club_graph()
    part = {n: 0 if n < 17 else 1 for n in G.nodes()}
    cache.update(0, part, G)
    assert len(cache.snapshots()) == 1


def test_partition_stability_score():
    cache = PartitionCache()
    G = nx.karate_club_graph()
    for t in range(5):
        part = {n: n % 3 for n in G.nodes()}
        cache.update(t, part, G)
    s = cache.stability_score()
    assert 0.0 <= s <= 1.0
