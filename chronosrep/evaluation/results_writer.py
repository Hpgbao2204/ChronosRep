from __future__ import annotations
import csv
import json
from pathlib import Path
from dataclasses import asdict
from .benchmark import RunResult
from .scenario_matrix import ScenarioMatrix


class ResultsWriter:
    def __init__(self, output_dir: str | Path = "results"):
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)

    def write_csv(self, results: list[RunResult], filename: str = "benchmark.csv") -> Path:
        path = self._out / filename
        if not results:
            return path
        rows = [asdict(r) for r in results]
        for row in rows:
            row.pop("extra", None)
        fieldnames = list(rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        return path

    def write_json(self, results: list[RunResult], filename: str = "benchmark.json") -> Path:
        path = self._out / filename
        with open(path, "w") as f:
            json.dump([asdict(r) for r in results], f, indent=2)
        return path

    def write_summary(self, summary: dict, filename: str = "summary.json") -> Path:
        path = self._out / filename
        with open(path, "w") as f:
            json.dump(summary, f, indent=2)
        return path

    def write_matrix(self, matrix: ScenarioMatrix, filename: str = "matrix.json") -> Path:
        path = self._out / filename
        data = [{"coords": c.coords, "metrics": c.metrics} for c in matrix.cells()]
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def write_all(self, results: list[RunResult], matrix: ScenarioMatrix | None = None) -> dict[str, Path]:
        paths: dict[str, Path] = {}
        paths["csv"]  = self.write_csv(results)
        paths["json"] = self.write_json(results)
        paths["summary"] = self.write_summary(_summarize(results))
        if matrix is not None:
            paths["matrix"] = self.write_matrix(matrix)
        return paths


def _summarize(results: list[RunResult]) -> dict:
    from collections import defaultdict
    import numpy as np
    by_sc: dict[str, list] = defaultdict(list)
    for r in results:
        by_sc[r.scenario].append(r)
    out = {}
    for sc, rs in by_sc.items():
        out[sc] = {
            "avg_rep":   float(np.mean([r.avg_reputation for r in rs])),
            "iso_rate":  float(np.mean([r.isolation_rate for r in rs])),
            "n_runs":    len(rs),
        }
    return out
