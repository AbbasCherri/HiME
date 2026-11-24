import numpy as np

from hime.experts import sigmoid
from hime.oracle import llr_gauss, llr_joint, make_world
from tests.conftest import zscore


def test_P2_deterministic():
    a, b = make_world(n=300, N=10, seed=3), make_world(n=300, N=10, seed=3)
    assert all(np.array_equal(a[k], b[k]) for k in ("y", "xS", "xI", "client", "alpha"))


def test_P2_joint_llr_additive_iff_assumption_a():
    for rho, additive in ((0.0, True), (0.8, False)):
        w = make_world(n=500, N=10, seed=1, rho=rho)
        i = np.arange(500)
        s = llr_gauss(w["xS"], w["params"]["mS"]) + llr_gauss(w["xI"], w["params"]["mI"])
        assert np.allclose(llr_joint(w, i), s, atol=1e-9) == additive


def test_P2_analytic_posterior_matches_frequency():
    w = make_world(n=60_000, N=50, seed=2)
    p = sigmoid(np.log(w["prev"] / (1 - w["prev"])) + llr_joint(w, np.arange(len(w["y"]))))
    for lo, hi in ((0, .02), (.02, .1), (.1, .3), (.3, 1.01)):
        m = (p >= lo) & (p < hi)
        se = np.sqrt(np.sum(p[m] * (1 - p[m]))) / m.sum()
        z = zscore(w["y"][m].mean(), p[m].mean(), se)
        print("bin", lo, hi, "z", z)
        assert z <= 4
