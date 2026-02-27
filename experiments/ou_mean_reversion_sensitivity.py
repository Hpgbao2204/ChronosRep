"""
ou_mean_reversion_sensitivity.py
Compares OU mean reversion trajectories across three values of the reversion
speed parameter theta, demonstrating robustness to initialization bias.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from chronosrep.modules.vadm import VADM

_OUT = Path(__file__).parent / "output" / "ou_mean_reversion_sensitivity.png"


def run(out_path: Path = _OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vadm = VADM(sigma=0.03, seed=7)
    mu = 0.8
    x0 = 0.2
    n_steps = 100
    thetas = [0.1, 0.3, 0.8]
    t_axis = list(range(n_steps + 1))

    fig, ax = plt.subplots(figsize=(9, 5))
    for theta in thetas:
        traj = vadm.step_ou_only(x0=x0, mu=mu, theta=theta, n_steps=n_steps)
        ax.plot(t_axis, traj, label=f"θ = {theta}")

    ax.axhline(mu, color="black", linestyle="--", linewidth=1.0, label=f"μ = {mu}")
    ax.axhline(x0, color="gray",  linestyle=":",  linewidth=0.8, label=f"X₀ = {x0}")
    ax.set_xlabel("Time Step t")
    ax.set_ylabel("Trust Score $X_t$")
    ax.set_title("OU Mean Reversion — Sensitivity to Reversion Speed θ")
    ax.legend()
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    run()
