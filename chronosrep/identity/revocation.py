from __future__ import annotations
import time
from dataclasses import dataclass, field
from collections import defaultdict

_EPOCH_DURATION_S = 3600


@dataclass
class RevocationEntry:
    vc_id: str
    subject_id: int
    revoked_at_t: int
    revoked_at_ts: int
    reason: str


class RevocationIndex:
    def __init__(self):
        self._revoked: dict[str, RevocationEntry] = {}
        self._by_subject: dict[int, list[str]] = defaultdict(list)
        self._by_epoch: dict[int, list[str]] = defaultdict(list)
        self._t: int = 0

    def tick(self) -> None:
        self._t += 1

    def revoke(self, vc_id: str, subject_id: int, reason: str = "unspecified") -> RevocationEntry:
        epoch = self._t // _EPOCH_DURATION_S
        entry = RevocationEntry(
            vc_id=vc_id,
            subject_id=subject_id,
            revoked_at_t=self._t,
            revoked_at_ts=int(time.time()),
            reason=reason,
        )
        self._revoked[vc_id] = entry
        self._by_subject[subject_id].append(vc_id)
        self._by_epoch[epoch].append(vc_id)
        return entry

    def is_revoked(self, vc_id: str) -> bool:
        return vc_id in self._revoked

    def subject_revocations(self, subject_id: int) -> list[RevocationEntry]:
        return [self._revoked[vid] for vid in self._by_subject.get(subject_id, []) if vid in self._revoked]

    def revocation_velocity(self, subject_id: int, window_t: int = 50) -> float:
        entries = self.subject_revocations(subject_id)
        recent = [e for e in entries if self._t - e.revoked_at_t <= window_t]
        return len(recent) / max(window_t, 1)

    def epoch_revocation_count(self, epoch: int) -> int:
        return len(self._by_epoch.get(epoch, []))

    def total_revoked(self) -> int:
        return len(self._revoked)

    def accumulation_vector(self, subject_id: int, n_bins: int = 10) -> list[int]:
        entries = self.subject_revocations(subject_id)
        if not entries:
            return [0] * n_bins
        min_t = min(e.revoked_at_t for e in entries)
        max_t = max(e.revoked_at_t for e in entries) + 1
        bin_size = max(1, (max_t - min_t) // n_bins)
        bins = [0] * n_bins
        for e in entries:
            idx = min((e.revoked_at_t - min_t) // bin_size, n_bins - 1)
            bins[idx] += 1
        return bins
