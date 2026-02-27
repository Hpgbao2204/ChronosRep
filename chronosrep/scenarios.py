from __future__ import annotations

import random


class SleeperAgentScenario:
    N_ATTACKERS = 20
    DEFECT_STEP = 201

    def setup(self, model) -> None:
        all_ids = [a.unique_id for a in model.schedule.agents]
        model._attacker_ids = set(random.sample(all_ids, self.N_ATTACKERS))
        for a in model.schedule.agents:
            if a.unique_id in model._attacker_ids:
                a.is_attacker = True

    def inject(self, model) -> None:
        t = model.current_step
        outcome = 1 if t < self.DEFECT_STEP else 0
        active = {a.unique_id for a in model.schedule.agents if not a.isolated}
        for uid in model._attacker_ids:
            if uid not in active:
                continue
            targets = [x for x in active if x != uid]
            if targets:
                model.ite.force_interaction(uid, random.choice(targets), outcome)


class TransgressionRecoveryScenario:
    PHASE_MISBEHAVE_START = 101
    PHASE_REFORM_START = 201

    def setup(self, model) -> None:
        all_ids = [a.unique_id for a in model.schedule.agents]
        n = max(1, int(0.20 * model.N))
        model._attacker_ids = set(random.sample(all_ids, n))
        for a in model.schedule.agents:
            if a.unique_id in model._attacker_ids:
                a.is_attacker = True

    def inject(self, model) -> None:
        t = model.current_step
        if t < self.PHASE_MISBEHAVE_START:
            outcome = 1
        elif t < self.PHASE_REFORM_START:
            outcome = 0
        else:
            outcome = 1
            for a in model.schedule.agents:
                if a.unique_id in model._attacker_ids:
                    a.is_attacker = False

        active = {a.unique_id for a in model.schedule.agents if not a.isolated}
        for uid in model._attacker_ids:
            if uid not in active:
                continue
            targets = [x for x in active if x != uid]
            if targets:
                model.ite.force_interaction(uid, random.choice(targets), outcome)


class CollusionFarmingScenario:
    def setup(self, model) -> None:
        all_ids = [a.unique_id for a in model.schedule.agents]
        n = max(2, int(0.20 * model.N))
        model._attacker_ids = set(random.sample(all_ids, n))
        self._colluders = list(model._attacker_ids)
        for a in model.schedule.agents:
            if a.unique_id in model._attacker_ids:
                a.is_attacker = False

    def inject(self, model) -> None:
        active_colluders = [uid for uid in self._colluders
                            if not model.schedule._agents[uid].isolated
                            if uid in model.schedule._agents]
        for uid in active_colluders:
            for target in active_colluders:
                if target != uid:
                    model.ite.force_interaction(uid, target, 1)
