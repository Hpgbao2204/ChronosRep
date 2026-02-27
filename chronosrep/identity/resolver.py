from __future__ import annotations
import hashlib
import random
import time
from dataclasses import dataclass, field
from typing import Optional

_DID_METHODS = ("did:web", "did:key", "did:ion", "did:ethr", "did:peer")
_RESOLUTION_LATENCY_MS_RANGE = (1, 8)


@dataclass
class ResolutionResult:
    did: str
    subject_id: int
    method: str
    resolved: bool
    pub_key_fingerprint: str
    rotation_count: int
    status: str
    resolution_ts: int
    linked_domain: Optional[str]


def _derive_did(subject_id: int, method: str) -> str:
    raw = f"{method}:{subject_id:08d}".encode()
    suffix = hashlib.sha3_256(raw).hexdigest()[:24]
    return f"{method}:{suffix}"


def _key_fingerprint(subject_id: int, rotation: int) -> str:
    raw = f"pubkey:{subject_id}:{rotation}".encode()
    return hashlib.blake2b(raw, digest_size=12).hexdigest()


class DIDResolver:
    def __init__(self, seed: int = 0):
        self._rng = random.Random(seed)
        self._cache: dict[str, ResolutionResult] = {}
        self._subject_to_did: dict[int, str] = {}
        self._rotation_log: dict[int, list[str]] = {}

    def register_subject(self, subject_id: int) -> str:
        if subject_id in self._subject_to_did:
            return self._subject_to_did[subject_id]
        method = self._rng.choice(_DID_METHODS)
        did = _derive_did(subject_id, method)
        self._subject_to_did[subject_id] = did
        fp = _key_fingerprint(subject_id, 0)
        self._rotation_log[subject_id] = [fp]
        result = ResolutionResult(
            did=did,
            subject_id=subject_id,
            method=method,
            resolved=True,
            pub_key_fingerprint=fp,
            rotation_count=0,
            status="ACTIVE",
            resolution_ts=int(time.time()),
            linked_domain=f"agent-{subject_id:04d}.did.example",
        )
        self._cache[did] = result
        return did

    def resolve(self, subject_id: int) -> Optional[ResolutionResult]:
        did = self._subject_to_did.get(subject_id)
        if did is None:
            return None
        return self._cache.get(did)

    def rotate_key(self, subject_id: int) -> Optional[str]:
        did = self._subject_to_did.get(subject_id)
        if did is None or did not in self._cache:
            return None
        r = self._cache[did]
        new_rot = r.rotation_count + 1
        new_fp = _key_fingerprint(subject_id, new_rot)
        self._rotation_log.setdefault(subject_id, []).append(new_fp)
        updated = ResolutionResult(
            did=did,
            subject_id=subject_id,
            method=r.method,
            resolved=True,
            pub_key_fingerprint=new_fp,
            rotation_count=new_rot,
            status=r.status,
            resolution_ts=int(time.time()),
            linked_domain=r.linked_domain,
        )
        self._cache[did] = updated
        return new_fp

    def deactivate(self, subject_id: int) -> bool:
        did = self._subject_to_did.get(subject_id)
        if did and did in self._cache:
            self._cache[did].status = "DEACTIVATED"
            return True
        return False

    def rotation_history(self, subject_id: int) -> list[str]:
        return list(self._rotation_log.get(subject_id, []))
