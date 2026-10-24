import numpy as np
import pytest

from bilacl.synthetic.phantom import make_bilateral_case


@pytest.fixture(scope="session")
def syn_case(tmp_path_factory):
    """Small synthetic bilateral case (coarser grid keeps tests fast)."""
    d = tmp_path_factory.mktemp("cases") / "SYN_test"
    return make_bilateral_case(d, seed=3, injury="midsubstance", shape=(96, 88, 80), spacing=(0.9, 0.9, 1.2))


@pytest.fixture
def rng():
    return np.random.default_rng(0)
