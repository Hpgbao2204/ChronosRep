"""tests/test_identity.py — unit tests for identity submodule."""
from __future__ import annotations
import pytest
import numpy as np

from chronosrep.identity.registry import CredentialRegistry
from chronosrep.identity.resolver import DIDResolver
from chronosrep.identity.revocation import RevocationIndex


def test_registry_register_and_get():
    reg = CredentialRegistry()
    from chronosrep.modules.vcgen import VCGen
    vcgen = VCGen()
    creds = vcgen.generate(agent_id=1, n_credentials=3)
    for c in creds:
        reg.register(c)
    assert len(reg.get_by_subject(1)) == 3


def test_registry_revoke():
    reg = CredentialRegistry()
    from chronosrep.modules.vcgen import VCGen
    vcgen = VCGen()
    creds = vcgen.generate(agent_id=2, n_credentials=2)
    for c in creds:
        reg.register(c)
    reg.revoke(creds[0].vc_id)
    active = [c for c in reg.get_by_subject(2) if not c.revoked]
    assert len(active) == 1


def test_registry_revocation_ratio():
    reg = CredentialRegistry()
    from chronosrep.modules.vcgen import VCGen
    vcgen = VCGen()
    creds = vcgen.generate(agent_id=3, n_credentials=4)
    for c in creds:
        reg.register(c)
    reg.revoke(creds[0].vc_id)
    ratio = reg.revocation_ratio(3)
    assert ratio == pytest.approx(0.25)


def test_resolver_resolve():
    resolver = DIDResolver(seed=0)
    doc, result = resolver.resolve(42)
    assert result.success
    assert doc is not None
    assert doc.did.startswith("did:")


def test_revocation_index_velocity():
    idx = RevocationIndex()
    for i in range(10):
        idx.revoke(0, f"vc-{i}", epoch=0)
    v = idx.revocation_velocity(0)
    assert v >= 10


def test_revocation_accumulation():
    idx = RevocationIndex()
    for ep in range(5):
        idx.revoke(0, f"vc-{ep}", epoch=ep)
    acc = idx.accumulation_vector(0, n_epochs=5, epoch_size=1)
    assert acc[-1] >= 1
