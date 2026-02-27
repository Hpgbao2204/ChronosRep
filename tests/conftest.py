"""
conftest.py – shared pytest fixtures for ChronosRep tests.
"""
from __future__ import annotations
import pytest
import numpy as np


@pytest.fixture(autouse=True)
def fix_seed():
    np.random.seed(0)
    yield


@pytest.fixture
def tiny_model():
    from chronosrep.model import ChronosRepModel
    return ChronosRepModel(n_agents=20, t_steps=10, tau=0.4)


@pytest.fixture
def vcgen():
    from chronosrep.modules.vcgen import VCGen
    return VCGen()


@pytest.fixture
def irv_pe():
    from chronosrep.modules.irv_pe import IRV_PE
    return IRV_PE()


@pytest.fixture
def vadm():
    from chronosrep.modules.vadm import VADM
    return VADM()


@pytest.fixture
def bsm():
    from chronosrep.modules.bsm import BSM
    return BSM()
