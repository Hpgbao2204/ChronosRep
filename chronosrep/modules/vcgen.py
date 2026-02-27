from __future__ import annotations
import hashlib
import random
import time
from dataclasses import dataclass, field
from itertools import chain, combinations
from typing import Optional

_FRAME = frozenset({"trusted", "untrusted", "unknown"})
_S_TR = frozenset({"trusted"})
_S_UN = frozenset({"untrusted"})
_S_UK = frozenset({"unknown"})
_S_TR_UN = frozenset({"trusted", "untrusted"})
_S_TR_UK = frozenset({"trusted", "unknown"})
_S_UN_UK = frozenset({"untrusted", "unknown"})

_VC_TYPES = ("KYC", "DID_DOC", "BEHAVIORAL", "DELEGATED", "GOVERNANCE")
_ISSUER_TIERS = ("ROOT_CA", "INTERMEDIATE_CA", "LEAF_ISSUER")
_ATTR_KEYS_KYC = ("identity_hash", "nationality_code", "risk_band", "kyc_level")
_ATTR_KEYS_DID = ("did_method", "pub_key_alg", "rotation_count", "linked_domain")
_ATTR_KEYS_BEH = ("tx_count_30d", "avg_tx_value", "flagged_ratio", "peer_score")
_ATTR_KEYS_DEL = ("delegator_id", "delegation_depth", "scope_bitmask", "expiry_epoch")
_ATTR_KEYS_GOV = ("dao_id", "vote_weight", "proposal_count", "slash_count")

_ATTR_KEY_MAP = {
    "KYC":        _ATTR_KEYS_KYC,
    "DID_DOC":    _ATTR_KEYS_DID,
    "BEHAVIORAL": _ATTR_KEYS_BEH,
    "DELEGATED":  _ATTR_KEYS_DEL,
    "GOVERNANCE": _ATTR_KEYS_GOV,
}

_TIER_TRUST_RANGE = {
    "ROOT_CA":         (0.88, 1.00),
    "INTERMEDIATE_CA": (0.70, 0.90),
    "LEAF_ISSUER":     (0.45, 0.75),
}

_REVOCATION_PROB_HONEST   = 0.02
_REVOCATION_PROB_ATTACKER = 0.35


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
    issuance_ts: int
    chain_depth: int


def _powerset_subsets():
    s = list(_FRAME)
    return tuple(
        frozenset(c)
        for c in chain.from_iterable(combinations(s, r) for r in range(1, len(s) + 1))
    )


_ALL_SUBSETS = _powerset_subsets()


def _build_honest_bba(rng: random.Random) -> dict:
    m_tr = rng.uniform(0.50, 0.78)
    m_uk = rng.uniform(0.04, 0.18)
    m_fr = rng.uniform(0.03, 0.10)
    residual = max(0.0, 1.0 - m_tr - m_uk - m_fr)
    m_un = residual * rng.uniform(0.00, 0.12)
    m_tr_uk = residual * rng.uniform(0.00, 0.08)
    raw = {
        _S_TR:    m_tr,
        _S_UN:    m_un,
        _S_UK:    m_uk,
        _S_TR_UK: m_tr_uk,
        _FRAME:   m_fr,
    }
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def _build_attacker_bba(rng: random.Random) -> dict:
    m_un = rng.uniform(0.48, 0.72)
    m_uk = rng.uniform(0.10, 0.24)
    m_fr = rng.uniform(0.05, 0.14)
    residual = max(0.0, 1.0 - m_un - m_uk - m_fr)
    m_tr = residual * rng.uniform(0.00, 0.08)
    m_un_uk = residual * rng.uniform(0.00, 0.10)
    raw = {
        _S_TR:    m_tr,
        _S_UN:    m_un,
        _S_UK:    m_uk,
        _S_UN_UK: m_un_uk,
        _FRAME:   m_fr,
    }
    total = sum(raw.values())
    return {k: v / total for k, v in raw.items()}


def _build_attributes(vc_type: str, rng: random.Random, is_attacker: bool) -> dict:
    keys = _ATTR_KEY_MAP[vc_type]
    attrs: dict = {}
    for k in keys:
        if "hash" in k or "key" in k or "did" in k:
            seed_val = rng.getrandbits(128)
            attrs[k] = hashlib.sha3_256(seed_val.to_bytes(16, "big")).hexdigest()[:32]
        elif "count" in k or "depth" in k or "rotation" in k:
            attrs[k] = rng.randint(0, 200 if is_attacker else 50)
        elif "ratio" in k or "score" in k or "weight" in k:
            attrs[k] = round(rng.uniform(0.0, 0.25 if is_attacker else 0.08), 4)
        elif "epoch" in k:
            attrs[k] = int(time.time()) + rng.randint(86400, 31536000)
        elif "bitmask" in k:
            attrs[k] = rng.getrandbits(16)
        elif "band" in k or "level" in k or "method" in k or "alg" in k:
            choices_map = {
                "risk_band":      ["LOW", "MEDIUM", "HIGH"] if is_attacker else ["LOW", "MEDIUM"],
                "kyc_level":      [1, 2, 3],
                "did_method":     ["did:web", "did:key", "did:ion", "did:ethr"],
                "pub_key_alg":    ["Ed25519", "secp256k1", "P-256"],
                "nationality_code": ["VN", "US", "SG", "EU", "UK"],
                "linked_domain":  ["example.com", "defi.xyz", "anon.io"],
                "dao_id":         ["dao_alpha", "dao_beta", "dao_gamma"],
            }
            pool = choices_map.get(k, ["A", "B", "C"])
            attrs[k] = rng.choice(pool)
        else:
            attrs[k] = round(rng.uniform(0.0, 1.0), 4)
    return attrs


def _issuer_tier_for_vc_type(vc_type: str, rng: random.Random) -> str:
    weights = {
        "KYC":        [0.40, 0.40, 0.20],
        "DID_DOC":    [0.20, 0.45, 0.35],
        "BEHAVIORAL": [0.05, 0.30, 0.65],
        "DELEGATED":  [0.10, 0.35, 0.55],
        "GOVERNANCE": [0.30, 0.50, 0.20],
    }
    return rng.choices(_ISSUER_TIERS, weights=weights[vc_type], k=1)[0]


def _issuer_trust(tier: str, rng: random.Random) -> float:
    lo, hi = _TIER_TRUST_RANGE[tier]
    return round(rng.uniform(lo, hi), 4)


def _chain_depth(tier: str, rng: random.Random) -> int:
    base = {"ROOT_CA": 0, "INTERMEDIATE_CA": 1, "LEAF_ISSUER": 2}[tier]
    return base + rng.randint(0, 1)


class VCGen:
    def __init__(self, seed_offset: int = 0):
        self._seed_offset = seed_offset

    def generate(
        self,
        agent_id: int,
        n_credentials: int = 5,
        is_attacker: bool = False,
    ) -> list[VCRecord]:
        rng = random.Random(agent_id + self._seed_offset * 100003)
        vc_types = rng.choices(_VC_TYPES, k=n_credentials)
        records: list[VCRecord] = []
        for i, vt in enumerate(vc_types):
            tier = _issuer_tier_for_vc_type(vt, rng)
            trust = _issuer_trust(tier, rng)
            if is_attacker:
                trust *= rng.uniform(0.5, 0.85)
            bba_fn = _build_attacker_bba if is_attacker else _build_honest_bba
            rev_prob = _REVOCATION_PROB_ATTACKER if is_attacker else _REVOCATION_PROB_HONEST
            records.append(VCRecord(
                vc_id=f"vc-{agent_id:04d}-{i}-{vt[:3]}",
                vc_type=vt,
                issuer_tier=tier,
                issuer_trust=round(trust, 4),
                subject_id=agent_id,
                revoked=rng.random() < rev_prob,
                attributes=_build_attributes(vt, rng, is_attacker),
                bba=bba_fn(rng),
                issuance_ts=int(time.time()) - rng.randint(0, 31536000),
                chain_depth=_chain_depth(tier, rng),
            ))
        return records
