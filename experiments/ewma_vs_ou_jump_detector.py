"""
ewma_vs_ou_jump_detector.py
Compares a baseline EWMA reputation tracker against the ChronosRep OU-Jump
detector on a simulated flash-loan exploit sequence.  The attacker behaves
honestly for the first phase, then switches to malicious evidence at a
known injection step.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

_EWMA_LAMBDA = 0.15
_N_STEPS     = 51
_ATTACK_STEP = 35
_OUT = Path(__file__).parent / "output" / "ewma_vs_ou_jump_detector.png"


def _ewma_trace(evidence: list[float], lam: float) -> list[float]:
    rep = 0.5
    traj = [rep]
    for e in evidence:
        rep = (1 - lam) * rep + lam * e
        traj.append(rep)
    return traj


def _ou_jd_trace(evidence: list[float]) -> list[float]:
    from chronosrep.modules.vadm import VADM
    vadm = VADM(theta_0=0.4, sigma=0.03, jump_scale=0.40, seed=17)
    import numpy as np
    irv_honest  = np.array([0.80, 0.02, 0.05, 0.82, 0.80])
    irv_attack  = np.array([0.05, 0.75, 0.10, 0.08, 0.05])
    rep, _ = vadm.step(0, irv_honest, 0.80)
    traj = [rep]
    for t, e in enumerate(evidence):
        irv = irv_attack if t >= _ATTACK_STEP - 1 else irv_honest
        rep, _ = vadm.step(0, irv, float(e))
        traj.append(rep)
    return traj


def _build_euler_evidence() -> list[float]:
    ev: list[float] = []
    for t in range(_N_STEPS):
        if t < _ATTACK_STEP:
            ev.append(1.0)
        else:
            ev.append(0.0)
    return ev


def run(out_path: Path = _OUT) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    evidence = _build_euler_evidence()
    t_axis   = list(range(_N_STEPS + 1))

    traj_ewma = _ewma_trace(evidence, _EWMA_LAMBDA)
    traj_ou   = _ou_jd_trace(evidence)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(t_axis, traj_ewma, label="Baseline (EWMA, λ=0.15)",
            color="#e08020", linewidth=2.0, linestyle="--")
    ax.plot(t_axis, traj_ou,   label="ChronosRep (OU-Jump)",
            color="#1060c0", linewidth=2.0)
    ax.axvline(_ATTACK_STEP, color="red", linewidth=1.2, linestyle=":",
               label=f"Attack onset (step {_ATTACK_STEP})")
    ax.annotate("Detection lag\n(baseline)", xy=(_ATTACK_STEP + 4, traj_ewma[_ATTACK_STEP + 4]),
                xytext=(_ATTACK_STEP + 8, 0.65),
                arrowprops=dict(arrowstyle="->", color="#e08020"),
                color="#e08020", fontsize=9)
    ax.annotate("Rapid\ncollapse", xy=(_ATTACK_STEP + 1, traj_ou[_ATTACK_STEP + 1]),
                xytext=(_ATTACK_STEP + 5, 0.25),
                arrowprops=dict(arrowstyle="->", color="#1060c0"),
                color="#1060c0", fontsize=9)
    ax.set_xlabel("Transaction Step")
    ax.set_ylabel("Trust Score")
    ax.set_title("EWMA Baseline vs. ChronosRep OU-Jump Detector — Flash Loan Exploit Sequence")
    ax.legend()
    ax.set_ylim(-0.05, 1.10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    run()
