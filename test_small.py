import time
from chronosrep import ChronosRepModel
from chronosrep.scenarios import (
    SleeperAgentScenario,
    TransgressionRecoveryScenario,
    CollusionFarmingScenario,
)

N_SMALL = 30
T_SMALL = 15


class SmallModel(ChronosRepModel):
    N = N_SMALL
    T = T_SMALL


def run(model_cls, label, scenario=None):
    m = model_cls(scenario=scenario)
    t0 = time.perf_counter()
    for _ in range(model_cls.T):
        m.step()
    elapsed = time.perf_counter() - t0

    df = m.datacollector.get_model_vars_dataframe()
    avg_rep = df["AvgReputation"].iloc[-1]
    iso_rate = df["IsolationRate"].iloc[-1]
    ttd = m.time_to_detection()

    print(f"[{label}] agents={model_cls.N} steps={model_cls.T} time={elapsed:.2f}s  "
          f"AvgRep={avg_rep:.4f}  IsoRate={iso_rate:.3f}  TTD={ttd}")
    return elapsed


if __name__ == "__main__":
    print("=== Small smoke test (baseline) ===")
    e0 = run(SmallModel, "SMALL/baseline")

    print("\n=== Small smoke test (Scenario 1 - Sleeper) ===")
    run(SmallModel, "SMALL/sleeper", SleeperAgentScenario())

    print("\n=== Small smoke test (Scenario 2 - Transgression) ===")
    run(SmallModel, "SMALL/transgression", TransgressionRecoveryScenario())

    print("\n=== Small smoke test (Scenario 3 - Collusion) ===")
    run(SmallModel, "SMALL/collusion", CollusionFarmingScenario())

    est = e0 / (N_SMALL * T_SMALL) * (ChronosRepModel.N * ChronosRepModel.T)
    print(f"\nEstimated full run (1000 agents × 500 steps): ~{est:.0f}s")
    if est < 120:
        print("→ Fast enough. Run `python run_experiments.py` for full scale.")
    else:
        print("→ Consider reducing ITE k or recompute_interval before full run.")
