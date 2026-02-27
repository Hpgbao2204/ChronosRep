from __future__ import annotations
import time
import csv
import json
from pathlib import Path
from dataclasses import dataclass, field, asdict
import numpy as np


@dataclass
class RunResult:
    scenario: str
    seed: int
    n_agents: int
    t_steps: int
    tau: float
    avg_reputation: float
    isolation_rate: float
    attacker_avg_rep: float
    ttd_mean: float
    ttd_std: float
    wall_time_s: float
    extra: dict = field(default_factory=dict)


@dataclass
class SweepConfig:
    n_agents_list:  list[int]   = field(default_factory=lambda: [100, 500, 1000])
    t_steps_list:   list[int]   = field(default_factory=lambda: [100, 300, 500])
    tau_list:       list[float] = field(default_factory=lambda: [0.3, 0.4, 0.5])
    seeds:          list[int]   = field(default_factory=lambda: list(range(5)))
    scenarios:      list[str]   = field(default_factory=lambda: [
        "baseline", "sleeper", "transgression", "collusion"
    ])


def _make_model(n_agents: int, t_steps: int, tau: float, seed: int):
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from chronosrep.model import ChronosRepModel
    np.random.seed(seed)
    return ChronosRepModel(n_agents=n_agents, t_steps=t_steps, tau=tau)


def _run_single(scenario: str, n_agents: int, t_steps: int, tau: float, seed: int) -> RunResult:
    from chronosrep.scenarios import (
        SleeperAgentScenario, TransgressionRecoveryScenario, CollusionFarmingScenario,
    )
    t0 = time.perf_counter()
    model = _make_model(n_agents, t_steps, tau, seed)
    if scenario == "sleeper":
        sc = SleeperAgentScenario()
        sc.setup(model)
    elif scenario == "transgression":
        sc = TransgressionRecoveryScenario()
        sc.setup(model)
    elif scenario == "collusion":
        sc = CollusionFarmingScenario()
        sc.setup(model)
    for step in range(t_steps):
        if scenario != "baseline":
            sc.inject(model)
        model.step()
    wall = time.perf_counter() - t0
    df = model.datacollector.get_model_vars_dataframe()
    avg_rep     = float(df["AvgReputation"].iloc[-1]) if "AvgReputation" in df else 0.0
    iso_rate    = float(df["IsolationRate"].iloc[-1]) if "IsolationRate" in df else 0.0
    atk_rep     = float(df["AttackerAvgReputation"].iloc[-1]) if "AttackerAvgReputation" in df else 0.0
    return RunResult(
        scenario=scenario,
        seed=seed,
        n_agents=n_agents,
        t_steps=t_steps,
        tau=tau,
        avg_reputation=avg_rep,
        isolation_rate=iso_rate,
        attacker_avg_rep=atk_rep,
        ttd_mean=0.0,
        ttd_std=0.0,
        wall_time_s=wall,
    )


class BenchmarkRunner:
    def __init__(self, config: SweepConfig | None = None):
        self._cfg = config or SweepConfig()
        self._results: list[RunResult] = []

    def run_single(self, scenario: str, n_agents: int, t_steps: int, tau: float, seed: int) -> RunResult:
        r = _run_single(scenario, n_agents, t_steps, tau, seed)
        self._results.append(r)
        return r

    def run_sweep(self, verbose: bool = True) -> list[RunResult]:
        cfg = self._cfg
        total = (
            len(cfg.scenarios)
            * len(cfg.n_agents_list)
            * len(cfg.t_steps_list)
            * len(cfg.tau_list)
            * len(cfg.seeds)
        )
        done = 0
        for sc in cfg.scenarios:
            for n in cfg.n_agents_list:
                for t in cfg.t_steps_list:
                    for tau in cfg.tau_list:
                        for seed in cfg.seeds:
                            r = _run_single(sc, n, t, tau, seed)
                            self._results.append(r)
                            done += 1
                            if verbose:
                                print(
                                    f"[{done}/{total}] sc={sc} n={n} t={t} "
                                    f"τ={tau} seed={seed} "
                                    f"rep={r.avg_reputation:.3f} iso={r.isolation_rate:.3f} "
                                    f"wall={r.wall_time_s:.2f}s"
                                )
        return self._results

    def summary(self) -> dict[str, dict[str, float]]:
        from collections import defaultdict
        by_sc: dict[str, list[RunResult]] = defaultdict(list)
        for r in self._results:
            by_sc[r.scenario].append(r)
        out = {}
        for sc, rs in by_sc.items():
            reps = [r.avg_reputation for r in rs]
            isos = [r.isolation_rate for r in rs]
            out[sc] = {
                "avg_rep_mean": float(np.mean(reps)),
                "avg_rep_std":  float(np.std(reps)),
                "iso_mean":     float(np.mean(isos)),
                "iso_std":      float(np.std(isos)),
                "n_runs":       len(rs),
            }
        return out

    def results(self) -> list[RunResult]:
        return list(self._results)
