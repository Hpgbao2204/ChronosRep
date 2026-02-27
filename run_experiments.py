from __future__ import annotations

import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from chronosrep import (
    ChronosRepModel,
    SleeperAgentScenario,
    TransgressionRecoveryScenario,
    CollusionFarmingScenario,
)
from chronosrep.modules.vadm import VADM

SCENARIOS = [
    ("Baseline",                   None),
    ("Scenario 1: Sleeper Attack", SleeperAgentScenario()),
    ("Scenario 2: Transgression",  TransgressionRecoveryScenario()),
    ("Scenario 3: Collusion Farm", CollusionFarmingScenario()),
]


def run_scenario(label: str, scenario) -> dict:
    print(f"  Running {label} ...", end=" ", flush=True)
    t0 = time.perf_counter()
    model = ChronosRepModel(scenario=scenario)
    for _ in range(ChronosRepModel.T):
        model.step()
    elapsed = time.perf_counter() - t0
    df = model.datacollector.get_model_vars_dataframe()
    ttd = model.time_to_detection()
    print(f"done in {elapsed:.1f}s  TTD={ttd}")
    return {"label": label, "df": df, "ttd": ttd, "model": model}


def plot_scenarios(results: list[dict]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("ChronosRep — Scenario Comparison (N=1000, T=500)", fontsize=13, fontweight="bold")

    metrics = [
        ("AvgReputation",         "Average Reputation",       axes[0]),
        ("IsolationRate",         "Isolation Rate",           axes[1]),
        ("AttackerAvgReputation", "Attacker Avg Reputation",  axes[2]),
    ]
    colors = ["#2196F3", "#F44336", "#4CAF50", "#FF9800"]

    for key, ylabel, ax in metrics:
        for r, color in zip(results, colors):
            series = r["df"][key]
            ax.plot(series.index, series.values, label=r["label"],
                    linewidth=1.5, color=color)
        ax.axhline(y=ChronosRepModel.TAU, color="gray", linestyle="--",
                   linewidth=0.8, alpha=0.7, label=f"τ={ChronosRepModel.TAU}")
        ax.set_xlabel("Simulation Step", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(ylabel, fontsize=11, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = "experiment_results.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Plot saved → {out}")
    plt.close()


def plot_figure16() -> None:
    """
    Figure 16: Resilience to Initialization Bias.
    Single honest agent, μ=0.8, X_0=0.2 (biased init).
    Plots OU mean-reversion under θ ∈ {0.1, 0.3, 0.8} for 100 steps.
    """
    MU = 0.8
    X0 = 0.2
    SIGMA = 0.03
    T = 100
    THETAS = [0.1, 0.3, 0.8]
    COLORS = ["#F44336", "#FF9800", "#4CAF50"]

    vadm = VADM(sigma=SIGMA)
    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(figsize=(9, 5))

    for theta, color in zip(THETAS, COLORS):
        x = X0
        trajectory = [x]
        for _ in range(T):
            drift = theta * (MU - x) * 1.0
            diffusion = SIGMA * rng.standard_normal()
            x = float(np.clip(x + drift + diffusion, 0.0, 1.0))
            trajectory.append(x)
        ax.plot(range(T + 1), trajectory, color=color, linewidth=1.8,
                label=rf"Correction Speed $\theta={theta}$")

    ax.axhline(y=MU, color="black", linestyle="--", linewidth=1.2, alpha=0.7,
               label=rf"True Identity Quality ($\mu={MU}$)")
    ax.set_xlim(0, T)
    ax.set_ylim(0.1, 1.05)
    ax.set_xlabel("Simulation Steps (Time)", fontsize=11)
    ax.set_ylabel("Trust Score", fontsize=11)
    ax.set_title("Figure 16: Mean Reversion — Resilience to Initialization Bias",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax.annotate(rf"Bad Initial Belief ($X_0={X0}$)",
                xy=(0, X0), xytext=(8, 0.28),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                fontsize=9)
    ax.annotate("Convergence to\nTrue Quality",
                xy=(72, MU - 0.01), xytext=(55, 0.65),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                fontsize=9)

    plt.tight_layout()
    out = "figure16_initialization_bias.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Figure 16 saved → {out}")
    plt.close()


def plot_figure17() -> None:
    """
    Figure 17: Forensic Analysis of Euler Finance Exploit.
    50 transaction steps: S_t=1 for steps 0-34, S_t=0 from step 35.
    Baseline = EWMA (λ=0.15) vs ChronosRep (OU-Jump).
    """
    ATTACK_STEP = 35
    N_STEPS = 50
    LAMBDA_EWMA = 0.15
    MU_NORMAL = 0.90
    SIGMA = 0.03
    JUMP_SCALE = 0.45

    trace = [1] * ATTACK_STEP + [0] * (N_STEPS - ATTACK_STEP)

    raw_evidence = []
    r_ewma = 0.9
    for s in trace:
        r_ewma = (1 - LAMBDA_EWMA) * r_ewma + LAMBDA_EWMA * s
        raw_evidence.append(r_ewma)

    rng = np.random.default_rng(7)
    x_ou = 0.92
    ou_trajectory = [x_ou]
    theta = 0.3

    for i, s in enumerate(trace):
        if s == 1:
            mu = MU_NORMAL
        else:
            mu = 0.0

        drift = theta * (mu - x_ou) * 1.0
        diffusion = SIGMA * rng.standard_normal()

        epsilon = abs(mu - x_ou)
        noise_floor = SIGMA / max(np.sqrt(2 * theta), 1e-6)
        gamma = epsilon / noise_floor
        jump = 0.0
        if gamma > 3.0:
            jump = -abs(float(rng.normal(loc=0.0, scale=JUMP_SCALE)))

        x_ou = float(np.clip(x_ou + drift + diffusion + jump, 0.0, 1.0))
        ou_trajectory.append(x_ou)

    ou_trajectory = ou_trajectory[:N_STEPS]

    baseline_full = [0.9] + [
        (1 - LAMBDA_EWMA) * prev + LAMBDA_EWMA * s
        for prev, s in zip([0.9] + raw_evidence[:-1], trace)
    ]
    baseline_full = baseline_full[:N_STEPS]

    fig, ax = plt.subplots(figsize=(10, 5))

    steps = list(range(N_STEPS))
    ax.plot(steps, raw_evidence, color="gray", linestyle=":",
            linewidth=1.2, label="Raw Behavioral Evidence (Normalized)", alpha=0.7)
    ax.plot(steps, baseline_full, color="#2196F3", linestyle="--",
            linewidth=2.0, label="Baseline (Static Decay)")
    ax.plot(steps, ou_trajectory, color="#F44336", linewidth=2.2,
            label="ChronosRep (OU Jump Diffusion)")

    ax.axvline(x=ATTACK_STEP, color="#FF9800", linestyle="-", linewidth=1.5, alpha=0.8)
    ax.text(ATTACK_STEP - 0.5, 0.52,
            "Euler Finance Exploited\n(Block 16817996)",
            color="#FF9800", fontsize=8.5, ha="right", fontweight="bold")

    ax.annotate("Instant Collapse\n(Jump Process)",
                xy=(ATTACK_STEP, ou_trajectory[ATTACK_STEP - 1] * 0.25),
                xytext=(ATTACK_STEP + 3, 0.25),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                fontsize=8.5)
    ax.annotate("Lag in Baseline",
                xy=(ATTACK_STEP + 8, baseline_full[min(ATTACK_STEP + 8, N_STEPS - 1)]),
                xytext=(ATTACK_STEP + 10, 0.82),
                arrowprops=dict(arrowstyle="->", color="black", lw=1.2),
                fontsize=8.5)

    ax.set_xlim(0, N_STEPS - 1)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Transaction Sequence", fontsize=11)
    ax.set_ylabel("Trust Score", fontsize=11)
    ax.set_title("Figure 17: Comparative Trust Dynamics under the Euler Finance Exploit Trace",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    out = "figure17_euler_finance_exploit.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"  Figure 17 saved → {out}")
    plt.close()


if __name__ == "__main__":
    print("=== Figure 16: Initialization Bias Resilience ===")
    plot_figure16()

    print("\n=== Figure 17: Euler Finance Exploit Forensics ===")
    plot_figure17()

    print(f"\n=== ChronosRep Full Experiment (N={ChronosRepModel.N}, T={ChronosRepModel.T}) ===")
    results = []
    for label, scenario in SCENARIOS:
        results.append(run_scenario(label, scenario))

    print("\n=== Summary ===")
    for r in results:
        df = r["df"]
        print(
            f"  {r['label']:<35} "
            f"FinalAvgRep={df['AvgReputation'].iloc[-1]:.4f}  "
            f"FinalIsoRate={df['IsolationRate'].iloc[-1]:.3f}  "
            f"TTD={r['ttd']}"
        )

    plot_scenarios(results)
    print("\nDone.")

