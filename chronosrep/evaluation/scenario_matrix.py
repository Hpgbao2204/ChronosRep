from __future__ import annotations
import itertools
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MatrixDimension:
    name: str
    values: list


@dataclass
class MatrixCell:
    coords: dict
    metrics: dict = field(default_factory=dict)


class ScenarioMatrix:
    def __init__(self, dimensions: list[MatrixDimension] | None = None):
        self._dims = dimensions or [
            MatrixDimension("n_agents", [100, 500, 1000]),
            MatrixDimension("tau",      [0.3, 0.4, 0.5]),
            MatrixDimension("scenario", ["baseline", "sleeper", "collusion"]),
        ]
        self._cells: list[MatrixCell] = self._build_grid()

    def _build_grid(self) -> list[MatrixCell]:
        names  = [d.name for d in self._dims]
        values = [d.values for d in self._dims]
        return [MatrixCell(coords=dict(zip(names, combo))) for combo in itertools.product(*values)]

    def cells(self) -> list[MatrixCell]:
        return self._cells

    def fill(self, coords: dict, metrics: dict) -> None:
        for cell in self._cells:
            if cell.coords == coords:
                cell.metrics.update(metrics)
                return
        self._cells.append(MatrixCell(coords=coords, metrics=metrics))

    def slice(self, **fixed_dims) -> list[MatrixCell]:
        result = []
        for cell in self._cells:
            if all(cell.coords.get(k) == v for k, v in fixed_dims.items()):
                result.append(cell)
        return result

    def pivot_table(self, row_dim: str, col_dim: str, metric: str) -> dict:
        rows = sorted({c.coords[row_dim] for c in self._cells if row_dim in c.coords})
        cols = sorted({c.coords[col_dim] for c in self._cells if col_dim in c.coords})
        table: dict[str, dict] = {}
        for r in rows:
            table[str(r)] = {}
            for c in cols:
                matches = [
                    cell.metrics.get(metric)
                    for cell in self._cells
                    if cell.coords.get(row_dim) == r
                    and cell.coords.get(col_dim) == c
                    and metric in cell.metrics
                ]
                if matches:
                    table[str(r)][str(c)] = float(np.mean(matches))
        return table

    def coverage(self) -> float:
        filled = sum(1 for c in self._cells if c.metrics)
        return filled / len(self._cells) if self._cells else 0.0
