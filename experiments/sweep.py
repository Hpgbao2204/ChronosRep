"""
sweep.py – Parameter sweep runner.
Runs BenchmarkRunner over a SweepConfig and writes results to results/.
"""
from __future__ import annotations
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from chronosrep.evaluation import BenchmarkRunner, SweepConfig, ResultsWriter
from chronosrep.utils import set_global_seed


def main():
    set_global_seed(42)
    cfg = SweepConfig(
        n_agents_list=[50, 100],
        t_steps_list=[50, 100],
        tau_list=[0.35, 0.40, 0.45],
        seeds=[0, 1, 2],
        scenarios=["baseline", "sleeper", "collusion"],
    )
    runner = BenchmarkRunner(config=cfg)
    print("Starting sweep …")
    results = runner.run_sweep(verbose=True)
    summary = runner.summary()

    writer = ResultsWriter(output_dir=ROOT / "results")
    paths = writer.write_all(results)
    for k, p in paths.items():
        print(f"  {k:10s} → {p}")

    print("\nSummary:")
    for sc, stats in summary.items():
        print(f"  {sc:14s}  rep={stats['avg_rep_mean']:.3f}±{stats['avg_rep_std']:.3f}  "
              f"iso={stats['iso_mean']:.3f}±{stats['iso_std']:.3f}  n={stats['n_runs']}")


if __name__ == "__main__":
    main()
