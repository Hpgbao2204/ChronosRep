from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from chronosrep.core.types import VCRecord, IRV, BehaviorEvent


class BaseModule(ABC):
    @abstractmethod
    def reset(self) -> None: ...


class BaseScenario(ABC):
    @abstractmethod
    def setup(self, model) -> None: ...

    @abstractmethod
    def inject(self, model) -> None: ...
