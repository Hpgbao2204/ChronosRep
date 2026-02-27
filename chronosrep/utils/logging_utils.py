from __future__ import annotations
import logging
import sys


_FMT = "%(asctime)s %(levelname)-8s [%(name)s] %(message)s"
_DATE = "%Y-%m-%dT%H:%M:%S"


def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter(_FMT, datefmt=_DATE))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


class StepLogger:
    def __init__(self, name: str, log_every: int = 50, level: int = logging.INFO):
        self._log = get_logger(name, level)
        self._every = log_every
        self._step = 0

    def tick(self, **metrics) -> None:
        self._step += 1
        if self._step % self._every == 0:
            parts = " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in metrics.items())
            self._log.info(f"step={self._step:>5d} {parts}")

    def force(self, msg: str) -> None:
        self._log.info(f"step={self._step:>5d} {msg}")
