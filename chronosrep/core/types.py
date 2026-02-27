from __future__ import annotations
from dataclasses import dataclass, field
from typing import FrozenSet, Mapping
import numpy as np


FrameOfDiscernment = FrozenSet[str]

MassFunction = Mapping[FrozenSet[str], float]

_THETA: FrameOfDiscernment = frozenset({"trusted", "untrusted", "unknown"})
_EMPTY: FrozenSet[str] = frozenset()

_SINGLETONS = {
    "trusted":   frozenset({"trusted"}),
    "untrusted": frozenset({"untrusted"}),
    "unknown":   frozenset({"unknown"}),
}

_VC_TYPES = ("KYC", "DID_DOC", "BEHAVIORAL", "DELEGATED", "REVOCATION_LIST")
_ISSUER_TIERS = ("ROOT_CA", "INTERMEDIATE_CA", "LEAF_ISSUER")


@dataclass(frozen=True)
class VCRecord:
    vc_id: str
    vc_type: str
    issuer_tier: str
    issuer_trust: float
    subject_id: int
    revoked: bool
    attributes: dict
    bba: dict


IRV = np.ndarray


@dataclass
class BehaviorEvent:
    t: int
    agent_id: int
    target_id: int
    outcome: int
    context: str = "default"
    penalized_signal: float = field(default=0.0)
