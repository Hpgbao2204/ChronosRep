from __future__ import annotations
import math
from itertools import chain, combinations
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from chronosrep.modules.vcgen import VCRecord

_FRAME = frozenset({"trusted", "untrusted", "unknown"})
_S_TR  = frozenset({"trusted"})
_S_UN  = frozenset({"untrusted"})
_S_UK  = frozenset({"unknown"})
_EMPTY = frozenset()

_VC_TYPE_WEIGHT = {
    "KYC":        1.00,
    "DID_DOC":    0.85,
    "BEHAVIORAL": 0.90,
    "DELEGATED":  0.70,
    "GOVERNANCE": 0.75,
}

_TIER_DISCOUNT = {
    "ROOT_CA":         1.00,
    "INTERMEDIATE_CA": 0.90,
    "LEAF_ISSUER":     0.75,
}

_CHAIN_DEPTH_PENALTY = 0.05


def _powerset(s):
    return [
        frozenset(c)
        for c in chain.from_iterable(combinations(list(s), r) for r in range(1, len(s) + 1))
    ]


_SUBSETS = _powerset(_FRAME)


def _belief_entropy(bba: dict) -> float:
    h = 0.0
    for A, m in bba.items():
        if m <= 0.0:
            continue
        card = len(A)
        denom = (1 << card) - 1
        if denom <= 0:
            continue
        h -= m * math.log2(m / denom)
    return h


def _conflict_coeff(m1: dict, m2: dict) -> float:
    K = 0.0
    for A in _SUBSETS:
        for B in _SUBSETS:
            if A & B == _EMPTY:
                K += m1.get(A, 0.0) * m2.get(B, 0.0)
    return min(K, 1.0)


def _chronosrep_combine(m1: dict, m2: dict) -> dict:
    K = _conflict_coeff(m1, m2)
    if K >= 1.0:
        return {_FRAME: 1.0}
    norm = 1.0 - K
    fused: dict = {}
    for A in _SUBSETS:
        mass = sum(
            m1.get(B, 0.0) * m2.get(C, 0.0)
            for B in _SUBSETS
            for C in _SUBSETS
            if B & C == A
        )
        if mass > 0.0:
            fused[A] = mass / norm
    return fused


def _condition_bba(bba: dict, condition_set: frozenset) -> dict:
    if not condition_set:
        return bba
    conditioned: dict = {}
    for A in _SUBSETS:
        intersection = A & condition_set
        if not intersection:
            continue
        conditioned[intersection] = conditioned.get(intersection, 0.0) + bba.get(A, 0.0)
    total = sum(conditioned.values())
    if total == 0.0:
        return {_FRAME: 1.0}
    return {k: v / total for k, v in conditioned.items()}


def _downweight_bba(bba: dict, w: float) -> dict:
    vacuous = 1.0 - w
    result: dict = {}
    for A, m in bba.items():
        result[A] = m * w + (vacuous if A == _FRAME else 0.0)
    return result


def _pignistic_transform(bba: dict) -> dict:
    betp: dict = {e: 0.0 for e in ["trusted", "untrusted", "unknown"]}
    for A, m in bba.items():
        card = len(A)
        if card == 0:
            continue
        share = m / card
        for elem in A:
            betp[elem] = betp.get(elem, 0.0) + share
    return betp


def _belief(bba: dict, hyp: frozenset) -> float:
    return sum(m for A, m in bba.items() if A and A <= hyp)


def _plausibility(bba: dict, hyp: frozenset) -> float:
    return sum(m for A, m in bba.items() if A & hyp)


def _effective_weight(vc) -> float:
    type_w  = _VC_TYPE_WEIGHT.get(vc.vc_type, 0.7)
    tier_d  = _TIER_DISCOUNT.get(vc.issuer_tier, 0.7)
    depth_p = max(0.0, 1.0 - vc.chain_depth * _CHAIN_DEPTH_PENALTY)
    return vc.issuer_trust * type_w * tier_d * depth_p


class IRV_PE:
    def __init__(self, eta: float = 2.5, revocation_penalty: float = 0.10):
        self.eta = eta
        self.revocation_penalty = revocation_penalty

    def process(self, agent_id: int, credentials: list) -> np.ndarray:
        active = [vc for vc in credentials if not vc.revoked]
        if not active:
            active = credentials

        bbas: list[dict] = []
        for vc in active:
            bba = dict(vc.bba)
            h = _belief_entropy(bba)
            w = _effective_weight(vc)
            if vc.revoked:
                w *= (1.0 - self.revocation_penalty)
            if h > self.eta:
                w *= self.eta / h
            w = max(0.01, min(w, 1.0))
            if w < 1.0:
                bba = _downweight_bba(bba, w)
            bbas.append(bba)

        if not bbas:
            return np.zeros(5)

        fused = bbas[0]
        for nxt in bbas[1:]:
            fused = _chronosrep_combine(fused, nxt)

        betp = _pignistic_transform(fused)

        bel_tr  = _belief(fused, _S_TR)
        bel_un  = _belief(fused, _S_UN)
        bel_uk  = _belief(fused, _S_UK)
        pl_tr   = _plausibility(fused, _S_TR)
        betp_tr = betp.get("trusted", 0.0)

        irv = np.clip(
            np.array([bel_tr, bel_un, bel_uk, pl_tr, betp_tr]),
            0.0, 1.0,
        )
        return irv
