"""
reputation_inequality_gini.py
Tracks the Gini coefficient of the reputation distribution over time across
three simulation conditions: baseline (no attack), sleeper agent attack, and
collusion farming.  A rising Gini indicates increasing stratification between
honest and adversarial cohorts.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chronosrep.model import ChronosRepModel
from chronosrep.scenarios import SleeperAgentScenario, CollusionFarmingScenario
from chronosrep.utils import gini_coefficient

_OUT = Path(__file__).parent / "output" / "reputation_inequality_gini.png"


def _gini_series(model: ChronosRepModel, t_steps: int, scenario=None) -> list[float]:
    ginis = []
    for step in range(t_steps):
        if scenario is not None:
            scenario.inject(model)
        model.step()
        reps = [a.reputation for a in model.schedule.agents]
        ginis.append(gini_coefficient(reps))
    return ginis


def main(out_path: Path = _OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    T = 150
    N = 200
    np.random.seed(42)

    m_base = ChronosRepModel(n_agents=N, t_steps=T, tau=0.4)
    g_base = _gini_series(m_base, T)

    np.random.seed(42)
    m_sleep = ChronosRepModel(n_agents=N, t_steps=T, tau=0.4)
    sc_sleep = SleeperAgentScenario()
    sc_sleep.setup(m_sleep)
    g_sleep = _gini_series(m_sleep, T, sc_sleep)

    np.random.seed(42)
    m_coll = ChronosRepModel(n_agents=N, t_steps=T, tau=0.4)
    sc_coll = CollusionFarmingScenario()
    sc_coll.setup(m_coll)
    g_coll = _gini_series(m_coll, T, sc_coll)

    steps = np.arange(1, T + 1)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(steps, g_base,  label="Baseline",  linewidth=2)
    ax.plot(steps, g_sleep, label="Sleeper",   linewidth=2, linestyle="--")
    ax.plot(steps, g_coll,  label="Collusion", linewidth=2, linestyle=":")
    ax.set_xlabel("Step")
    ax.set_ylabel("Gini coefficient")
    ax.set_title("Reputation Inequality (Gini Coefficient) across Adversarial Conditions")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
