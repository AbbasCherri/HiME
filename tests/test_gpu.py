"""GPU (torch) Newton fit must reproduce the CPU scipy reference fit of the central linear expert."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
from hime.experts import LinearExpert, logit  # noqa: E402
from hime.gpu import available, fit_newton  # noqa: E402

DEVICES = ["cpu"] + (["cuda"] if available() else [])


def _data(n=3000, d=40, prev=0.05, seed=0):
    rng = np.random.default_rng(seed)
    X = np.c_[np.ones(n), rng.standard_normal((n, d))]
    beta = rng.standard_normal(d + 1) * 0.4
    beta[0] = logit(prev)
    y = (rng.random(n) < 1 / (1 + np.exp(-X @ beta))).astype(float)
    return X, y, logit(y.mean())


@pytest.mark.parametrize("device", DEVICES)
def test_newton_matches_scipy(device):
    X, y, lp = _data()
    lams = [1e-2, 1e-5, 1e-3, 1e-4]                      # deliberately unsorted
    th = fit_newton(X, y, lp, lams, device=device)
    for k, lam in enumerate(lams):
        e = LinearExpert(X.shape[1], lp, lam=lam)
        ref = e.fit_central(X, y)
        assert np.abs(th[k] - ref).max() < 1e-6, (device, lam, np.abs(th[k] - ref).max())
        assert e.loss(th[k], X, y) <= e.loss(ref, X, y) + 1e-12
    print(f"newton vs scipy on {device}: agree to 1e-6 for lams {lams}")
