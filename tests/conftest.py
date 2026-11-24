import numpy as np
import pytest

from hime.engine import Config, central, evaluate, run
from hime.oracle import make_world
from hime.pipeline import prepare


@pytest.fixture(scope="session")
def small():
    w = make_world(n=1500, N=30, d_i=8, seed=7, extent=20_000.0)
    pb = prepare(w, r=None)
    uav = np.random.default_rng(0).uniform(0, 20_000.0, (4, 2))
    return w, pb, uav


def zscore(est, true, se):
    return abs(est - true) / max(se, 1e-300)
