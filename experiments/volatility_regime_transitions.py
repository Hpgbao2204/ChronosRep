"""
volatility_regime_transitions.py
Simulates an OU-Jump agent trajectory across three behavioural regimes —
calm, stress, and active attack — and compares the realised volatility
estimate against the EWMA volatility filter.  Regime boundaries are
highlighted to show how the volatility estimator adapts to structural shifts.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chronosrep.stochastic.solver import SDESolver, OUParams
from chronosrep.stochastic.volatility import VolatilityEstimator

_OUT = Path(__file__).parent / "output" / "volatility_regime_transitions.png"


def main(out_path: Path = _OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(99)
    T = 200
    pid = 0

    solver = SDESolver(dt=1.0, method="euler_maruyama", seed=99)
    vol_est = VolatilityEstimator(window=20)
    solver.init(pid, 0.5)

    xs:     list[float] = []
    ewmas:  list[float] = []
    rvols:  list[float] = []
    regimes: list[str]  = []

    params_calm   = OUParams(mu=0.75, theta=0.3, sigma=0.03)
    params_stress = OUParams(mu=0.40, theta=0.5, sigma=0.15)
    params_attack = OUParams(mu=0.20, theta=0.8, sigma=0.25)

    prev = solver.current(pid)
    for t in range(T):
        if t < 60:
            p, gamma, regime = params_calm,   0.0, "calm"
        elif t < 120:
            p, gamma, regime = params_stress, 2.5, "stress"
        else:
            p, gamma, regime = params_attack, 4.5, "attack"

        x = solver.step(pid, p, gamma)
        vol_est.update(pid, x, prev)
        prev = x

        xs.append(x)
        ewmas.append(vol_est.ewma_vol(pid))
        rvols.append(vol_est.realized_vol(pid))
        regimes.append(regime)

    steps = np.arange(T)
    colors = {"calm": "#d4efdf", "stress": "#fdebd0", "attack": "#fadbd8"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

    for ax in (ax1, ax2):
        for i, r in enumerate(regimes):
            ax.axvspan(i, i + 1, color=colors[r], alpha=0.4, linewidth=0)

    ax1.plot(steps, xs, color="navy", linewidth=1.2, label="$X_t$")
    ax1.axhline(0.4, color="gray", linestyle="--", linewidth=1, label="τ = 0.4")
    ax1.set_ylabel("SDE state / reputation")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_title("OU-Jump Trajectory and Volatility Across Behavioural Regimes")

    ax2.plot(steps, rvols, color="darkorange", linewidth=1.8, label="Realised volatility")
    ax2.plot(steps, ewmas, color="purple",     linewidth=1.8, linestyle="--", label="EWMA volatility")
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Volatility")
    patches = [mpatches.Patch(color=v, label=k, alpha=0.6) for k, v in colors.items()]
    ax2.legend(handles=patches + list(ax2.get_lines()), fontsize=7)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
