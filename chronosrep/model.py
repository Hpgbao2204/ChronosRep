from __future__ import annotations

import mesa
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np

from chronosrep.modules import VCGen, IRV_PE, BSM, VADM, ITE


class ChronosAgent(mesa.Agent):
    def __init__(self, unique_id: int, model: "ChronosRepModel"):
        super().__init__(unique_id, model)
        self.irv: np.ndarray = np.zeros(5)
        self.reputation: float = 0.5
        self.volatility: float = 0.0
        self.evidence: list[float] = []
        self.isolated: bool = False
        self.is_attacker: bool = False

    def step(self):
        if self.isolated:
            return

        all_ids = [a.unique_id for a in self.model.schedule.agents if not a.isolated]

        credentials = self.model.vcgen.generate(self.unique_id, is_attacker=self.is_attacker)
        self.irv = self.model.irv_pe.process(self.unique_id, credentials)

        interactions = self.model.ite.generate_interactions(self.unique_id, all_ids)
        outcomes = [outcome for _, outcome in interactions]
        self.evidence = [
            self.model.ite.penalized_evidence(self.unique_id, target, outcome)
            for target, outcome in interactions
        ]

        bsm_result = self.model.bsm.monitor(self.unique_id, outcomes)
        anomaly = bsm_result["anomaly_score"]

        r_static = float(np.mean(self.evidence)) if self.evidence else float(self.irv[0])
        r_static = r_static * (1.0 - 0.5 * anomaly)

        self.reputation, self.volatility = self.model.vadm.step(
            self.unique_id, self.irv, r_static
        )

        if self.reputation < self.model.TAU:
            self.isolated = True
            self.model._newly_isolated.add(self.unique_id)


class ChronosRepModel(mesa.Model):
    N: int = 1000
    T: int = 500
    TAU: float = 0.4

    def __init__(self, scenario=None):
        super().__init__()
        self.schedule = RandomActivation(self)
        self.scenario = scenario
        self.current_step: int = 0

        self._newly_isolated: set[int] = set()
        self._isolation_step: dict[int, int] = {}
        self._attacker_ids: set[int] = set()

        self.vcgen = VCGen()
        self.irv_pe = IRV_PE()
        self.bsm = BSM()
        self.vadm = VADM()
        self.ite = ITE()

        for i in range(self.N):
            agent = ChronosAgent(i, self)
            self.schedule.add(agent)

        if self.scenario:
            self.scenario.setup(self)

        self.datacollector = DataCollector(
            model_reporters={
                "AvgReputation": lambda m: float(np.mean(
                    [a.reputation for a in m.schedule.agents if not a.isolated]
                )) if any(not a.isolated for a in m.schedule.agents) else 0.0,
                "IsolationRate": lambda m: sum(
                    1 for a in m.schedule.agents if a.isolated
                ) / m.N,
                "AttackerAvgReputation": lambda m: float(np.mean(
                    [a.reputation for a in m.schedule.agents
                     if a.unique_id in m._attacker_ids and not a.isolated]
                )) if any(
                    a.unique_id in m._attacker_ids and not a.isolated
                    for a in m.schedule.agents
                ) else 0.0,
            }
        )

    def step(self):
        self._newly_isolated.clear()
        self.current_step += 1

        if self.scenario:
            self.scenario.inject(self)

        self.schedule.step()
        self.ite.tick()

        for uid in self._newly_isolated:
            if uid not in self._isolation_step:
                self._isolation_step[uid] = self.current_step

        self.datacollector.collect(self)

    def time_to_detection(self, attacker_ids: set[int] | None = None) -> float | None:
        ids = attacker_ids or self._attacker_ids
        detected = [self._isolation_step[i] for i in ids if i in self._isolation_step]
        return float(np.mean(detected)) if detected else None

