"""
revocation_velocity_analysis.py
Simulates credential revocation dynamics in a mixed honest/attacker population.
Tracks per-epoch revocation velocity and the cumulative revocation accumulation
curve for each cohort, demonstrating the divergence that emerges as attackers
accumulate disqualifying credential events over time.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chronosrep.identity.revocation import RevocationIndex

_OUT = Path(__file__).parent / "output" / "revocation_velocity_analysis.png"


def main(out_path: Path = _OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.random.seed(7)
    T = 100
    N_honest   = 80
    N_attacker = 20
    idx = RevocationIndex()

    honest_ids   = list(range(N_honest))
    attacker_ids = list(range(N_honest, N_honest + N_attacker))

    velocities: list[float] = []
    atk_accum:  list[float] = []
    hon_accum:  list[float] = []

    for step in range(T):
        epoch = step // 10
        for aid in honest_ids:
            if np.random.random() < 0.01:
                idx.revoke(aid, f"vc-{aid}-{step}", epoch)
        for aid in attacker_ids:
            if np.random.random() < 0.15:
                idx.revoke(aid, f"vc-{aid}-{step}", epoch)

        velocities.append(idx.revocation_velocity(epoch))
        atk_acc = np.mean([idx.accumulation_vector(aid, T, 10)[-1] for aid in attacker_ids])
        hon_acc = np.mean([idx.accumulation_vector(aid, T, 10)[-1] for aid in honest_ids])
        atk_accum.append(float(atk_acc))
        hon_accum.append(float(hon_acc))

    steps = np.arange(T)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    ax1.plot(steps, velocities, color="crimson", linewidth=1.8)
    ax1.set_ylabel("Revocation velocity (per epoch)")
    ax1.set_title("Credential Revocation Dynamics — Honest vs. Attacker Cohorts")
    ax1.grid(alpha=0.3)

    ax2.plot(steps, atk_accum, label="Attacker", color="red",       linewidth=2)
    ax2.plot(steps, hon_accum, label="Honest",   color="steelblue", linewidth=2)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Cumulative revocations")
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved → {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
