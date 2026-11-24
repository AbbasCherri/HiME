import numpy as np

from hime.experts import logit, sigmoid, softplus
from hime.fusion import fit_stacker, pattern_logit, stacker_objective
from hime.oracle import llr_gauss, make_world
from tests.conftest import zscore


def _data(n, rho, seed=0, pi=0.1):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < pi).astype(float); u = rng.normal(size=n)
    xs = rng.normal(size=n) + y + rho * u * y; xi = rng.normal(size=n) + 0.8 * y + rho * u * y
    return xs - 0.5, 0.8 * xi - 0.32, y, logit(y.mean())      # gS, gI = exact LLRs when rho = 0


def _ll(z, y):
    return np.mean(softplus(z) - y * z)


def test_P3p_unique_and_bounds():
    gS, gI, y, lp = _data(5000, 0.0)
    ps = [fit_stacker(gS, gI, y, lp, p_init=np.array(p)) for p in ([0, 1, 1, 0], [2, 0, 3, -1], [-1, 2, 0, 1])]
    assert all(np.allclose(ps[0], p, atol=1e-6) for p in ps)
    p = ps[0]; gam = 10 / len(y)
    L = _ll(lp + p[0] + p[1] * gS + p[2] * gI + p[3] * gS * gI, y)
    assert L <= _ll(lp + gS + gI, y) + 1e-12
    assert L <= _ll(lp + gS, y) + gam / 2 + 1e-12 and L <= _ll(lp + gI, y) + gam / 2 + 1e-12
    p3 = fit_stacker(gS, gI, y, lp, interaction=False)
    assert stacker_objective(p, gS, gI, y, lp, gam)[0] <= stacker_objective(p3, gS, gI, y, lp, gam, False)[0] + 1e-12


def test_P3p_active_bound():
    gS, gI, y, lp = _data(4000, 0.0)
    p = fit_stacker(gS, -gI, y, lp)                            # anti-informative image -> w_I at 0
    assert p[2] <= 1e-8, p


def test_P2_oracle_recovery_and_interaction():
    gS, gI, y, lp = _data(400_000, 0.0, seed=1)
    p = fit_stacker(gS, gI, y, lp)
    assert np.allclose(p, [0, 1, 1, 0], atol=0.05), p
    gS, gI, y, lp = _data(400_000, 1.0, seed=2)
    p = fit_stacker(gS, gI, y, lp)
    print("w_SI under violated (a):", p[3])
    assert abs(p[3]) > 0.02


def test_P2_general_fusion_calibrated_per_pattern():
    w = make_world(n=80_000, N=50, d_d=4, seed=5)
    pr = w["params"]; pG = pr["pG"]
    delta = np.log(pG[1] / pG[0])
    z = pattern_logit(logit(w["prev"]), llr_gauss(w["xS"], pr["mS"]), llr_gauss(w["xI"], pr["mI"]),
                      llr_gauss(w["xD"], pr["mD"]), w["pattern"], delta)
    p = sigmoid(z)
    for g in range(3):
        m = w["pattern"] == g
        se = np.sqrt(np.sum(p[m] * (1 - p[m]))) / m.sum()
        assert zscore(w["y"][m].mean(), p[m].mean(), se) <= 4
