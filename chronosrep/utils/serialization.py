from __future__ import annotations
import json
import pickle
from pathlib import Path
from dataclasses import asdict
import numpy as np


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


def save_json(obj, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, cls=NumpyEncoder, indent=indent)
    return path


def load_json(path: str | Path) -> object:
    with open(path) as f:
        return json.load(f)


def save_pickle(obj, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    return path


def load_pickle(path: str | Path) -> object:
    with open(path, "rb") as f:
        return pickle.load(f)


def dataclass_to_dict(obj) -> dict:
    try:
        return asdict(obj)
    except TypeError:
        return vars(obj)


def agent_state_snapshot(agents: list) -> list[dict]:
    rows = []
    for a in agents:
        row = {
            "unique_id":  a.unique_id,
            "reputation": float(getattr(a, "reputation", 0.0)),
            "isolated":   bool(getattr(a, "isolated", False)),
            "is_attacker": bool(getattr(a, "is_attacker", False)),
        }
        irv = getattr(a, "irv", None)
        if irv is not None:
            row["irv"] = irv.tolist() if isinstance(irv, np.ndarray) else irv
        rows.append(row)
    return rows


def save_checkpoint(model, path: str | Path, step: int | None = None) -> Path:
    path = Path(path)
    if step is not None:
        path = path.with_stem(f"{path.stem}_step{step:05d}")
    agents_snapshot = agent_state_snapshot(model.schedule.agents)
    payload = {
        "step": step,
        "agents": agents_snapshot,
    }
    return save_json(payload, path)
