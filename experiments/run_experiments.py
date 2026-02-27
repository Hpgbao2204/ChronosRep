from __future__ import annotations
import time
import numpy as np
import matplotlib.pyplot as plt

from chronosrep import ChronosRepModel
from chronosrep.scenarios import (
    SleeperAgentScenario,
    TransgressionRecoveryScenario,
    CollusionFarmingScenario,
)
from chronosrep.analysis import MetricsCollector


def _run(model_cls, scenario=None) -> tuple[ChronosRepModel, float]:
    m = model_cls(scenario=scenario)
    t0 = time.perf_counter()
    for _ in range(model_cls.T):
        m.step()
    return m, time.perf_counter() - t0


def _rep_series(m: ChronosRepModel) -> list[float]:
    df = m.datacollector.get_model_vars_dataframe()
    return df["AvgReputation"].tolist()


def plot_reputation_curves(
    series_map: dict[str, list[float]],
    out_path: str = "experiments/reputation_evolution.png",
) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    styles = {
        "Baseline":     ("#2a9d8f", "-"),
        "Sleeper":      ("#e76f51", "--"),
        "Transgression":("#e9c46a", "-."),
        "Collusion":    ("#264653", ":"),
    }
    for label, series in series_map.items():
        color, ls = styles.get(label, ("#888888", "-"))
        ax.plot(series, label=label, color=color, linestyle=ls, linewidth=1.8)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Average Reputation")
    ax.set_title("ChronosRep — Reputation Evolution Across Scenarios")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[Plot] saved → {out_path}")


def main() -> None:
    N = ChronosRepModel.N
    T = ChronosRepModel.T

    print(f"Running baseline  (N={N}, T={T}) ...")
    m_base, t0 = _run(ChronosRepModel)
    print(f"  done in {t0:.1f}s  AvgRep={m_base.datacollector.get_model_vars_dataframe()['AvgReputation'].iloc[-1]:.4f}")

    print("Running Scenario 1 — Sleeper ...")
    m_sl, t1 = _run(ChronosRepModel, SleeperAgentScenario())
    df1 = m_sl.datacollector.get_model_vars_dataframe()
    print(f"  done in {t1:.1f}s  AvgRep={df1['AvgReputation'].iloc[-1]:.4f}  IsoRate={df1['IsolationRate'].iloc[-1]:.3f}")

    print("Running Scenario 2 — Transgression ...")
    m_tr, t2 = _run(ChronosRepModel, TransgressionRecoveryScenario())
    df2 = m_tr.datacollector.get_model_vars_dataframe()
    print(f"  done in {t2:.1f}s  AvgRep={df2['AvgReputation'].iloc[-1]:.4f}  IsoRate={df2['IsolationRate'].iloc[-1]:.3f}")

    print("Running Scenario 3 — Collusion ...")
    m_co, t3 = _run(ChronosRepModel, CollusionFarmingScenario())
    df3 = m_co.datacollector.get_model_vars_dataframe()
    print(f"  done in {t3:.1f}s  AvgRep={df3['AvgReputation'].iloc[-1]:.4f}  IsoRate={df3['IsolationRate'].iloc[-1]:.3f}")

    plot_reputation_curves({
        "Baseline":      _rep_series(m_base),
        "Sleeper":       _rep_series(m_sl),
        "Transgression": _rep_series(m_tr),
        "Collusion":     _rep_series(m_co),
    })

    from experiments.plot_fig16 import plot_fig16
    from experiments.plot_fig17 import plot_fig17
    plot_fig16()
    plot_fig17()


if __name__ == "__main__":
    main()
