from __future__ import annotations
import hashlib
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


_DID_METHODS = ("did:web", "did:key", "did:ion", "did:ethr", "did:peer")
_STATUS_ACTIVE   = "ACTIVE"
_STATUS_REVOKED  = "REVOKED"
_STATUS_EXPIRED  = "EXPIRED"
_STATUS_SUSPENDED = "SUSPENDED"


@dataclass
class DIDDocument:
    did: str
    subject_id: int
    method: str
    pub_key_fingerprint: str
    created_ts: int
    updated_ts: int
    status: str
    linked_credentials: list[str] = field(default_factory=list)
    rotation_count: int = 0


@dataclass
class RegistryEntry:
    vc_id: str
    subject_id: int
    vc_type: str
    issuer_tier: str
    issuer_trust: float
    issuance_ts: int
    expiry_ts: Optional[int]
    status: str
    attributes_hash: str


def _fingerprint(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=16).hexdigest()


class CredentialRegistry:
    def __init__(self):
        self._entries: dict[str, RegistryEntry] = {}
        self._by_subject: dict[int, list[str]] = defaultdict(list)
        self._by_type: dict[str, list[str]] = defaultdict(list)
        self._total_registered: int = 0
        self._total_revoked: int = 0

    def register(self, vc) -> RegistryEntry:
        import json
        attr_bytes = json.dumps(vc.attributes, sort_keys=True).encode()
        entry = RegistryEntry(
            vc_id=vc.vc_id,
            subject_id=vc.subject_id,
            vc_type=vc.vc_type,
            issuer_tier=vc.issuer_tier,
            issuer_trust=vc.issuer_trust,
            issuance_ts=vc.issuance_ts,
            expiry_ts=None,
            status=_STATUS_REVOKED if vc.revoked else _STATUS_ACTIVE,
            attributes_hash=_fingerprint(attr_bytes),
        )
        self._entries[vc.vc_id] = entry
        self._by_subject[vc.subject_id].append(vc.vc_id)
        self._by_type[vc.vc_type].append(vc.vc_id)
        self._total_registered += 1
        if vc.revoked:
            self._total_revoked += 1
        return entry

    def lookup(self, vc_id: str) -> Optional[RegistryEntry]:
        return self._entries.get(vc_id)

    def subject_credentials(self, subject_id: int) -> list[RegistryEntry]:
        ids = self._by_subject.get(subject_id, [])
        return [self._entries[i] for i in ids if i in self._entries]

    def revoke(self, vc_id: str) -> bool:
        e = self._entries.get(vc_id)
        if e and e.status == _STATUS_ACTIVE:
            e.status = _STATUS_REVOKED
            self._total_revoked += 1
            return True
        return False

    def active_count(self, subject_id: int) -> int:
        return sum(
            1 for e in self.subject_credentials(subject_id)
            if e.status == _STATUS_ACTIVE
        )

    def revocation_ratio(self, subject_id: int) -> float:
        creds = self.subject_credentials(subject_id)
        if not creds:
            return 0.0
        return sum(1 for e in creds if e.status == _STATUS_REVOKED) / len(creds)

    def stats(self) -> dict:
        return {
            "total_registered": self._total_registered,
            "total_revoked":    self._total_revoked,
            "active":           self._total_registered - self._total_revoked,
        }
